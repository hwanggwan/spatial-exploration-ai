"""Sequential mapper replay with simultaneous sensing-count and travel caps."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .baseline import write_csv
from .deepkriging import NeuralMapperConfig,calibration_multiplier,fit_ensemble_snapshots,standardized_features
from .deepkriging_study import prediction_metrics
from .estimated_gp import EstimatorConfig
from .resource_replay import ResourceConfig,acquired_states,design_routes,generate_fields,geometry
from .sample_size_study import bootstrap_mean,fit_snapshot


MODELS = ("oracle","estimated","deepkriging")
COLORS = {"oracle":"#27754d","estimated":"#376aa3","deepkriging":"#cf5656"}


def neural_states(observations,neural,seed_base,checkpoint_folder=None):
    """Fit only acquired prefixes, independently for every map and prefix."""
    features,_,_ = standardized_features(observations['points'],'deepkriging',neural)
    total,sites = len(observations['counts']),len(features)
    mean,variance = np.empty((total,sites)),np.empty((total,sites))
    members = np.empty((total,neural.members,sites))
    times = np.empty(total)
    epoch,decay = neural.epochs[0],neural.weight_decays[0]
    for count in np.unique(observations['counts']):
        chosen = np.flatnonzero(observations['counts']==count)
        for start in range(0,len(chosen),neural.chunk_maps):
            ids = chosen[start:start+neural.chunk_maps]
            seeds = [seed_base+int(state)*neural.members+member for state in ids for member in range(neural.members)]
            # Representative checkpoint per prefix length; all predictions/seeds are archived.
            checkpoint = checkpoint_folder/f'n{count}-states-first-chunk.pt' if checkpoint_folder and start==0 else None
            result,timing,_ = fit_ensemble_snapshots(features,observations['indices'][ids,:count],
                observations['values'][ids,:count],'deepkriging',neural,seeds,decay,(epoch,),save_path=checkpoint)
            mean[ids],variance[ids],members[ids] = result[epoch]['mean'],result[epoch]['raw_variance'],result[epoch]['members']
            times[ids] = timing[epoch]/len(ids)
    return {'mean':mean,'raw_variance':variance,'members':members,'fit_seconds':times}


def gp_states(observations,evaluator,config,estimator):
    """Fresh GP snapshot fits at every acquired prefix, matching prior mapper studies."""
    total,sites = len(observations['counts']),len(observations['points'])
    predictions = {model:{'mean':np.empty((total,sites)),'variance':np.empty((total,sites)),
                          'fit_seconds':np.empty(total)} for model in ('oracle','estimated')}
    fit_rows = []
    for state,count in enumerate(observations['counts']):
        for model in predictions:
            began = time.perf_counter()
            kwargs = {'oracle_prior':evaluator['true_covariance']} if model=='oracle' else {}
            belief = fit_snapshot(model,observations['points'],observations['indices'][state,:count],
                observations['values'][state,:count],config.noise_std,estimator,**kwargs)
            predictions[model]['fit_seconds'][state] = time.perf_counter()-began
            predictions[model]['mean'][state] = belief.mean
            predictions[model]['variance'][state] = np.diag(belief.covariance)
            if model=='estimated':
                fit_rows.append({'state':state,'field':int(observations['fields'][state]),'measurements':int(count),
                    'length_scale':belief.length_scale,'process_variance':belief.process_variance,
                    'fit_calls':len(belief.fit_history),'boundary_fit_calls':sum(e['boundary_hit'] for e in belief.fit_history)})
    return predictions,fit_rows


def summarize_endpoints(rows,config):
    summaries,comparisons = [],[]
    for cap in config.observation_caps:
        for budget in config.travel_budgets:
            for strategy in ('random','greedy'):
                common = {'observation_cap':cap,'travel_budget':budget,'strategy':strategy}
                by_model = {}
                for model in MODELS:
                    selected = sorted([r for r in rows if r['observation_cap']==cap and r['travel_budget']==budget
                        and r['strategy']==strategy and r['model']==model],key=lambda r:r['field'])
                    by_model[model] = selected
                    summary = {**common,'model':model,'fields':len(selected)}
                    for key in ('heldout_rmse','heldout_coverage_95','raw_heldout_coverage_95','heldout_mean_std',
                                'measurements','travel_distance','cumulative_fit_seconds'):
                        summary[key] = float(np.mean([r[key] for r in selected]))
                    summary['rmse_bootstrap_95_interval'] = bootstrap_mean([r['heldout_rmse'] for r in selected])
                    summaries.append(summary)
                gp = np.asarray([r['heldout_rmse'] for r in by_model['estimated']])
                dk = np.asarray([r['heldout_rmse'] for r in by_model['deepkriging']])
                comparisons.append({**common,'model':'deepkriging','reference':'estimated',
                    'mean_paired_rmse_difference':float((dk-gp).mean()),
                    'paired_bootstrap_95_interval':bootstrap_mean(dk-gp),
                    'rmse_advantage_percent':float(100*(gp.mean()-dk.mean())/gp.mean()),
                    'lower_rmse_fields':int(np.sum(dk<gp))})
    return summaries,comparisons


def make_curves(steps,episodes,config):
    """Carry the last available estimate after stopping; never invent observations."""
    lookup = {(r['episode'],r['model'],r['step']):r for r in steps}
    rows = []
    for cap in config.observation_caps:
        for budget in config.travel_budgets:
            for strategy in ('random','greedy'):
                chosen = [e for e in episodes if e['observation_cap']==cap and e['travel_budget']==budget and e['strategy']==strategy]
                for axis,values in (('measurement_allowance',range(1,cap+1)),('distance_allowance',np.linspace(0,budget,41))):
                    for value in values:
                        for model in MODELS:
                            samples = []
                            counts = []
                            for e in chosen:
                                step = min(int(value),len(e['route']))-1 if axis=='measurement_allowance' else int(
                                    np.searchsorted(e['distances'],value+1e-12,side='right')-1)
                                samples.append(lookup[e['episode'],model,step]['heldout_rmse'])
                                counts.append(step+1)
                            rows.append({'observation_cap':cap,'travel_budget':budget,'strategy':strategy,
                                'axis':axis,'allowance':float(value),'model':model,'mean_rmse':float(np.mean(samples)),
                                'mean_measurements_available':float(np.mean(counts))})
    return rows


def render_results(output,config,episodes,evaluator,predictions,summaries,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cap,budget = max(config.observation_caps),max(config.travel_budgets)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for model in MODELS:
        for strategy,style in (('random','--'),('greedy','-')):
            for axis,name in zip(axes,('measurement_allowance','distance_allowance')):
                selected = [r for r in curves if r['observation_cap']==cap and r['travel_budget']==budget
                    and r['model']==model and r['strategy']==strategy and r['axis']==name]
                axis.plot([r['allowance'] for r in selected],[r['mean_rmse'] for r in selected],style,
                    color=COLORS[model],label=f'{model}, {strategy} route',
                    drawstyle='steps-post' if name=='distance_allowance' else 'default')
    for axis,label in zip(axes,('Measurement allowance (start counts as one)','Distance allowance (unit-square coordinates)')):
        axis.set_xlabel(label)
        axis.set_ylabel('Mean held-out latent RMSE')
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(f'Sequential shared-route replay: at most {cap} observations and distance {budget:g}\n'
                 'Stop when either constraint prevents continuation; carry last prediction after stopping',fontsize=11)
    fig.savefig(output/'sequential_curves.png',dpi=150)
    plt.close(fig)

    fig,axes = plt.subplots(1,len(config.observation_caps),figsize=(11,4),constrained_layout=True,squeeze=False)
    for column,nmax in enumerate(config.observation_caps):
        for model in ('estimated','deepkriging'):
            for strategy,style in (('random','--'),('greedy','-')):
                selected = sorted([r for r in summaries if r['observation_cap']==nmax and r['model']==model
                    and r['strategy']==strategy],key=lambda r:r['travel_budget'])
                axes[0,column].plot([r['travel_budget'] for r in selected],[r['heldout_rmse'] for r in selected],style+'o',
                    color=COLORS[model],label=f'{model}, {strategy}')
        axes[0,column].set_title(f'Observation cap = {nmax}')
        axes[0,column].set_xlabel('Travel-distance cap')
        axes[0,column].set_ylabel('Mean final held-out RMSE')
        axes[0,column].set_xticks(config.travel_budgets)
        axes[0,column].grid(alpha=0.2)
        axes[0,column].legend(fontsize=8)
    fig.suptitle('Both budgets apply simultaneously — upper bounds, not forced equal resource use')
    fig.savefig(output/'budget_comparison.png',dpi=150)
    plt.close(fig)

    points,_ = geometry(config)
    fig,axes = plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    truth = evaluator['truth'][0]
    half = 0.5/(config.grid_size-1)
    for row,strategy in enumerate(('random','greedy')):
        episode = next(e for e in episodes if e['field']==0 and e['strategy']==strategy
                       and e['observation_cap']==cap and e['travel_budget']==budget)
        final = episode['state_ids'][-1]
        for column,(label,array) in enumerate((('Truth (evaluation only)',truth),
            ('Fitted GP',predictions['estimated']['mean'][final]),('DeepKriging-style',predictions['deepkriging']['mean'][final]))):
            axis = axes[row,column]
            plot = axis.imshow(array.reshape(config.grid_size,config.grid_size),origin='lower',
                extent=(-half,1+half,-half,1+half),vmin=truth.min(),vmax=truth.max(),cmap='viridis')
            route = points[episode['route']]
            axis.plot(route[:,0],route[:,1],'w.-',linewidth=1,markersize=4)
            axis.plot(*route[0],'wo',markeredgecolor='black',markersize=7)
            axis.plot(*route[-1],'wx',markersize=8)
            axis.set_title(f'{strategy}: {label}',fontsize=10)
            axis.set_xlabel('x')
            axis.set_ylabel('y')
            fig.colorbar(plot,ax=axis,shrink=0.7)
        axes[row,0].text(0.02,0.98,f"n={len(route)}, distance={episode['distances'][-1]:.2f}",
                        transform=axes[row,0].transAxes,va='top',color='white',fontsize=9,
                        bbox={'facecolor':'black','alpha':0.45,'edgecolor':'none'})
    fig.suptitle(f'Predefined field 0 — common start (0,0), caps n={cap}, distance={budget:g}\n'
                 'Each mapper receives exactly the same ordered observations within a route',fontsize=11)
    fig.savefig(output/'example_routes.png',dpi=150)
    plt.close(fig)


def run_study(output,config=None,neural=None,estimator=None):
    config = config or ResourceConfig()
    neural = neural or NeuralMapperConfig(epochs=(100,),weight_decays=(0.0001,))
    estimator = estimator or EstimatorConfig()
    for setting in (config,neural,estimator):setting.validate()
    if len(neural.epochs)!=1 or len(neural.weight_decays)!=1:
        raise ValueError('Use prespecified neural settings; no selection on this benchmark')
    output = Path(output)
    if (output/'protocol.json').exists():raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    (output/'checkpoints').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    protocol = {'resources':asdict(config),'neural':asdict(neural),'estimator':asdict(estimator),
        'objective':'final held-out latent RMSE under simultaneous observation-count and travel-distance upper bounds',
        'protocol':'shared sequential random/fixed-assumption-greedy routes; mapper replay, not mapper-driven closed loop',
        'start':'site 0 at (0,0); first reading consumes one observation and zero travel',
        'travel':'straight-line Euclidean distance; no obstacles or return-to-base requirement',
        'field':'gradual nonstationary long/short mixture, zero mean and unit variance',
        'truth_access':'simulator/evaluator only; mappers receive acquired prefix values only',
        'neural_settings':'fixed 100 epochs and decay 0.0001 by default, carried from the prior independent transition tuning; no new benchmark tuning',
        'early_prefixes':'same fixed architecture/settings at n=1..16; n<8 is an explicit sparse-data extrapolation',
        'updates':'both non-oracle mappers fit fresh from exactly the acquired prefix at every step; no look-ahead labels or warm starts',
        'calibration':'independent route-distributed maps; pooled per-prefix-length scalar; no posterior or conformal guarantee',
        'timing':'GP single-prefix fits; neural independent batched fits amortized over states; cumulative costs charge every prefix even when numerical results are cached',
        'seeds':{'calibration_field':3000000,'calibration_route':3010000,'calibration_members':3020000,
                 'benchmark_field':3100000,'benchmark_route':3110000,'benchmark_members':3120000},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
        'cpu_threads':1}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    calibration_episodes,calibration_states = design_routes(config,config.calibration_fields,3010000)
    calibration_evaluator = generate_fields(config,config.calibration_fields,3000000)
    calibration_observations = acquired_states(config,calibration_states,calibration_evaluator['potential_sensor_values'])
    np.savez_compressed(output/'calibration_acquired.npz',**calibration_observations)
    np.savez_compressed(output/'calibration_evaluation_only.npz',**calibration_evaluator)
    (output/'calibration_routes.json').write_text(json.dumps(calibration_episodes,indent=2)+'\n')
    began = time.perf_counter()
    calibrated = neural_states(calibration_observations,neural,3020000)
    calibration_wall = time.perf_counter()-began
    multipliers = {}
    mask = calibration_evaluator['evaluation_mask']
    for count in np.unique(calibration_observations['counts']):
        ids = np.flatnonzero(calibration_observations['counts']==count)
        truth = calibration_evaluator['truth'][calibration_observations['fields'][ids]]
        multipliers[int(count)] = calibration_multiplier(truth,calibrated['mean'][ids],calibrated['raw_variance'][ids],mask)
    (output/'frozen_protocol.json').write_text(json.dumps({'variance_multipliers':multipliers,
        'frozen_before_benchmark_generation':True,'neural':asdict(neural)},indent=2)+'\n')

    episodes,states = design_routes(config,config.fields,3110000)
    evaluator = generate_fields(config,config.fields,3100000)
    observations = acquired_states(config,states,evaluator['potential_sensor_values'])
    if not set(observations['counts']).issubset(multipliers):
        raise RuntimeError('Calibration must cover every evaluated prefix length')
    np.savez_compressed(output/'acquired_states.npz',**observations)
    np.savez_compressed(output/'evaluation_only.npz',**evaluator)
    (output/'routes.json').write_text(json.dumps(episodes,indent=2)+'\n')
    predictions,fit_rows = gp_states(observations,evaluator,config,estimator)
    deep = neural_states(observations,neural,3120000,output/'checkpoints')
    deep['variance'] = deep['raw_variance']*np.asarray([multipliers[int(n)] for n in observations['counts']])[:,None]
    predictions['deepkriging'] = deep
    np.savez_compressed(output/'predictions.npz',**{f'{model}_{key}':value for model,result in predictions.items() for key,value in result.items()})
    write_csv(output/'gp_fits.csv',fit_rows)
    state_scores = {}
    for state,field in enumerate(observations['fields']):
        for model,prediction in predictions.items():
            metrics = prediction_metrics(evaluator['truth'][field],prediction['mean'][state],prediction['variance'][state],mask)
            raw = prediction_metrics(evaluator['truth'][field],prediction['mean'][state],
                prediction.get('raw_variance',prediction['variance'])[state],mask)
            state_scores[state,model] = {**metrics,'raw_heldout_coverage_95':raw['heldout_coverage_95']}
    steps,endpoints = [],[]
    for episode in episodes:
        common = {key:episode[key] for key in ('episode','field','strategy','observation_cap','travel_budget')}
        for model,prediction in predictions.items():
            cumulative = 0.0
            for step,state in enumerate(episode['state_ids']):
                cumulative += prediction['fit_seconds'][state]
                row = {**common,'model':model,'step':step,'measurements':step+1,
                    'travel_distance':episode['distances'][step],**state_scores[state,model],
                    'cumulative_fit_seconds':float(cumulative)}
                steps.append(row)
            endpoints.append({**row,'stop_reason':episode['stop_reason']})
    summaries,comparisons = summarize_endpoints(endpoints,config)
    curves = make_curves(steps,episodes,config)
    write_csv(output/'step_metrics.csv',steps)
    write_csv(output/'final_metrics.csv',endpoints)
    write_csv(output/'curves.csv',curves)
    for name,value in (('summary.json',summaries),('paired_mapper_comparisons.json',comparisons)):
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    render_results(output,config,episodes,evaluator,predictions,summaries,curves)
    diagnostics = {'episodes':len(episodes),'unique_acquired_prefix_states':len(states),
        'neural_member_fits':len(states)*neural.members,'sequential_metric_rows':len(steps),'final_metric_rows':len(endpoints),
        'all_count_caps_respected':all(len(e['route'])<=e['observation_cap'] for e in episodes),
        'all_distance_caps_respected':all(e['distances'][-1]<=e['travel_budget']+1e-12 for e in episodes),
        'evaluation_sites_never_acquired':not any(mask[e['route']].any() for e in episodes),
        'same_ordered_observations_for_mappers':True,'frozen_before_benchmark_generation':True,
        'calibration_neural_wall_seconds':calibration_wall,
        'benchmark_neural_fit_seconds':float(deep['fit_seconds'].sum()),
        'benchmark_estimated_gp_fit_seconds':float(predictions['estimated']['fit_seconds'].sum())}
    (output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (output/'dataset_manifest.json').write_text(json.dumps({path.name:hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.glob('*.npz')},indent=2)+'\n')
    return summaries,diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/resource-study'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):summaries,diagnostics = run_study(args.output)
    for row in summaries:
        if row['model']!='oracle':
            print(f"{row['strategy']:6s} n<={row['observation_cap']:2d} d<={row['travel_budget']:.1f} {row['model']:12s} "
                  f"RMSE={row['heldout_rmse']:.4f} used n={row['measurements']:.2f}, d={row['travel_distance']:.3f}")
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':main()
