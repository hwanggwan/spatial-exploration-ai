import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from spatial_exploration.resource_replay import (ResourceConfig,acquired_states,design_routes,generate_fields,geometry)

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    from spatial_exploration.deepkriging import NeuralMapperConfig
    from spatial_exploration.estimated_gp import EstimatorConfig
    from spatial_exploration.resource_study import run_study


class ResourceReplayTests(unittest.TestCase):
    def test_routes_obey_both_budgets_and_identical_prefixes_share_states(self):
        config = ResourceConfig(grid_size=5,fields=2,calibration_fields=2,observation_caps=(2,4),travel_budgets=(0.5,1.0))
        episodes,states = design_routes(config,2,0)
        points,mask = geometry(config)
        self.assertEqual(len(episodes),16)
        prefixes = {}
        for episode in episodes:
            route = episode['route']
            self.assertEqual(route[0],0)
            self.assertFalse(mask[route].any())
            self.assertEqual(len(route),len(set(route)))
            self.assertLessEqual(len(route),episode['observation_cap'])
            distance = np.linalg.norm(np.diff(points[route],axis=0),axis=1).sum()
            self.assertAlmostEqual(distance,episode['distances'][-1])
            self.assertLessEqual(distance,episode['travel_budget']+1e-12)
            for n,state in enumerate(episode['state_ids'],start=1):
                key = (episode['field'],tuple(route[:n]))
                if key in prefixes:self.assertEqual(state,prefixes[key])
                prefixes[key] = state
                self.assertEqual(states[state]['indices'],route[:n])
        self.assertEqual(len(prefixes),len(states))

    def test_only_acquired_values_are_exposed_and_hidden_evaluation_values_do_not_affect_them(self):
        config = ResourceConfig(grid_size=5,fields=2,calibration_fields=2,observation_caps=(4,),travel_budgets=(1.0,))
        _,states = design_routes(config,2,10)
        evaluator = generate_fields(config,2,20)
        observed = acquired_states(config,states,evaluator['potential_sensor_values'])
        changed = evaluator['potential_sensor_values'].copy()
        changed[:,evaluator['evaluation_mask']] += 1e8
        repeated = acquired_states(config,states,changed)
        for key in observed:np.testing.assert_array_equal(observed[key],repeated[key])
        self.assertNotIn('truth',observed)
        self.assertNotIn('true_covariance',observed)
        for index,state in enumerate(states):
            n = len(state['indices'])
            np.testing.assert_array_equal(observed['values'][index,:n],evaluator['potential_sensor_values'][state['field'],state['indices']])
            self.assertTrue(np.isnan(observed['values'][index,n:]).all())


@unittest.skipUnless(HAS_TORCH,'Install PyTorch for the sequential neural replay')
class ResourceWorkflowTests(unittest.TestCase):
    @patch('spatial_exploration.resource_study.render_results')
    def test_frozen_new_fields_and_shared_ordered_data(self,*_):
        config = ResourceConfig(grid_size=5,fields=2,calibration_fields=2,observation_caps=(2,4),travel_budgets=(0.5,1.0))
        neural = NeuralMapperConfig(knot_grids=(2,3),basis_width=8,members=2,epochs=(2,),weight_decays=(0.0001,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            def checked_fields(config,fields,seed_base):
                if seed_base==3100000:self.assertTrue((output/'frozen_protocol.json').exists())
                return generate_fields(config,fields,seed_base)
            with patch('spatial_exploration.resource_study.generate_fields',side_effect=checked_fields):
                summaries,diagnostics=run_study(output,config,neural,EstimatorConfig(optimizer_restarts=0))
            self.assertEqual(len(summaries),24)
            self.assertEqual(diagnostics['final_metric_rows'],48)
            self.assertTrue(diagnostics['all_count_caps_respected'])
            self.assertTrue(diagnostics['all_distance_caps_respected'])
            self.assertTrue(diagnostics['evaluation_sites_never_acquired'])
            routes=json.loads((output/'routes.json').read_text())
            with np.load(output/'acquired_states.npz') as obs:
                for episode in routes:
                    for step,state in enumerate(episode['state_ids']):
                        self.assertEqual(obs['counts'][state],step+1)
                        np.testing.assert_array_equal(obs['indices'][state,:step+1],episode['route'][:step+1])
            with (output/'curves.csv').open() as stream:
                for row in csv.DictReader(stream):
                    available=float(row['mean_measurements_available'])
                    self.assertLessEqual(available,int(row['observation_cap']))
                    if row['axis']=='measurement_allowance':self.assertLessEqual(available,float(row['allowance']))
                    elif float(row['allowance'])==0:self.assertEqual(available,1)


if __name__=='__main__':unittest.main()
