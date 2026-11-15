// Distance-synchronized replay of the refined corridor benchmark.
const refinedMethods = [
  {key:"fine_candidates", label:"Greedy", path:"#b9603b", stop:"#ef8b64"},
  {key:"fine_rollout", label:"Rollout", path:"#328b7c", stop:"#6cccbb"}
];
const refinedNames = {short:"Short range", mixed:"Mixed by area", long:"Long range"};
const refinedZones = ["Lower-left room","Lower-right room","Corridor","Upper room"];
const refinedSlider = byId("refined-step");
const refinedPlay = byId("refined-play");
let refinedScene, refinedLimit, refinedExtent, refinedCeiling, refinedPosition=0, refinedAnimation=null;
let refinedViews=[], refinedTruth;

function stopRefined(){
  if(refinedAnimation!==null) cancelAnimationFrame(refinedAnimation);
  refinedAnimation=null;
  refinedPlay.textContent="▶ Play";
}

function mountRefined(){
  const container=byId("refined-cards");
  container.innerHTML=refinedMethods.map(method=>`
    <article class="duel-card" style="--route:${method.path}">
      <h2>${method.label}</h2>
      <div class="viewport">
        <canvas role="img" aria-label="${method.label} 3D route and reconstructed map"></canvas>
        <div class="viewport-ui"><span class="viewport-label" data-stat="layer"></span></div>
        <div class="viewport-bottom"><span></span><span class="sensor-state" data-stat="phase"></span></div>
      </div>
      <div class="stats">
        <div><span class="key">Hold-out RMSE</span><span class="number" data-stat="rmse"></span></div>
        <div><span class="key">Steps walked</span><span class="number" data-stat="steps"></span></div>
        <div><span class="key">Readings</span><span class="number" data-stat="readings"></span></div>
      </div>
      <div class="rooms" aria-label="Readings by area" data-stat="zones"></div>
      <section class="refined-uncertainty">
        <h3>Uncertainty</h3>
        <canvas width="288" height="288" role="img"
                aria-label="${method.label} top-down uncertainty map"></canvas>
        <div class="legend" aria-label="Posterior standard deviation color scale">
          <span>0</span><span class="ramp uncertainty"></span><span data-uncertainty-max></span><span>Std. dev.</span>
        </div>
      </section>
    </article>`).join("")+`
    <article class="duel-card refined-truth" style="--route:#475e53">
      <h2>True field</h2>
      <div class="viewport"><canvas role="img" aria-label="True 3D sensor field with held-out sites"></canvas></div>
      <p class="desc">Small dots mark the 100 hold-out sites used to score the maps.</p>
    </article>`;
  refinedViews=refinedMethods.map((method,i)=>{
    const card=container.children[i];
    return {method, canvas:card.querySelector(".viewport canvas"),
      uncertainty:card.querySelector(".refined-uncertainty canvas"),
      uncertaintyMax:card.querySelector("[data-uncertainty-max]"),
      stats:Object.fromEntries([...card.querySelectorAll("[data-stat]")].map(node=>[node.dataset.stat,node]))};
  });
  refinedTruth=container.querySelector(".refined-truth canvas");
  [...refinedViews.map(view=>view.canvas),refinedTruth].forEach(registerWorldCanvas);
}

function showRefinedResults(matched){
  const heading=matched ? ["Greedy","Rollout prefix"] : ["Greedy · original sites","Greedy · all sites","Rollout · all sites"];
  const rows=Object.entries(refinedData.results).map(([scenario,result])=>{
    const values=matched ? result.matched : result.caps;
    return `<tr aria-current="${scenario===refinedScene.scenario}"><th scope="row">${refinedNames[scenario]}</th>
      ${values.map(v=>`<td>${v.toFixed(4)}</td>`).join("")}</tr>`;
  }).join("");
  byId("refined-results").innerHTML=`<thead><tr><th scope="col">Field</th>
    ${heading.map(name=>`<th scope="col">${name}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
  byId("refined-result-note").textContent=matched
    ? "For each start, compare the readings taken by the time greedy stopped walking."
    : "Mean hold-out RMSE under the shared 52-step and 24-reading caps.";
  const result=refinedData.results[refinedScene.scenario];
  const signed=v=>`${v>=0?"+":""}${v.toFixed(4)}`;
  byId("refined-result-detail").textContent=matched
    ? `${result.fields.toLocaleString("en-US")} paired fields · Rollout − greedy 95% interval [${result.interval.map(signed).join(", ")}]`
    : `${result.fields.toLocaleString("en-US")} paired fields for the highlighted range; the 3D replay shows one field.`;
}

function loadRefined(){
  stopRefined();
  if(!refinedTruth) mountRefined();
  const layout=Number(byId("refined-layout").value), scenario=byId("refined-field").value;
  const matched=byId("refined-budget").value==="matched";
  refinedScene=refinedData.cases.find(c=>c.layout===layout && c.scenario===scenario);
  refinedLimit=matched ? refinedScene.cutoff : refinedData.travel;
  const plans=Object.values(refinedScene.plans);
  refinedExtent=Math.max(1, Math.ceil(Math.max(...refinedScene.truth.map(Math.abs),
    ...plans.flatMap(p=>p.mean.flat().map(Math.abs)))/1000));
  refinedCeiling=Math.max(.1, Math.ceil(Math.max(...plans.flatMap(p=>p.std.flat()))/100)/10);
  refinedPosition=0;
  refinedSlider.max=refinedLimit;
  byId("reading-cap").textContent=`${refinedData.cap} readings max`;
  byId("travel-cap").textContent=matched ? `${refinedLimit} walking steps compared` : `${refinedLimit} walking steps max`;
  byId("scale-min").textContent=(-refinedExtent).toFixed(1);
  byId("scale-max").textContent=refinedExtent.toFixed(1);
  byId("std-max").textContent=refinedCeiling.toFixed(1);
  for (const view of refinedViews) view.uncertaintyMax.textContent=refinedCeiling.toFixed(1);
  byId("refined-explanation").textContent=matched
    ? "The original rollout is replayed up to greedy’s walking distance. Drag any view to rotate together."
    : "Both routes start together and use the same caps. Drag any view to rotate together.";
  showRefinedResults(matched);
  paintRefined();
}

function drawRefinedUncertainty(view, layout, values, measured) {
  if(view.lastUncertainty===values) return;
  view.lastUncertainty=values;
  const canvas=view.uncertainty, ctx=canvas.getContext("2d");
  if(!ctx) return;
  const size=layout.size, unit=canvas.width/size;
  const cell=site=>[site%size,size-1-Math.floor(site/size)];
  for(let site=0;site<values.length;site++){
    const [col,row]=cell(site);
    const x=Math.round(col*unit), y=Math.round(row*unit);
    ctx.fillStyle=color(values[site]/1000,true,refinedCeiling);
    ctx.fillRect(x,y,Math.round((col+1)*unit)-x,Math.round((row+1)*unit)-y);
  }
  ctx.beginPath();
  for(const [x0,y0,x1,y1] of layout.walls){
    ctx.moveTo(x0*unit,y0*unit);
    ctx.lineTo(x1*unit,y1*unit);
  }
  ctx.strokeStyle="#243d3a";
  ctx.lineWidth=2;
  ctx.lineCap="round";
  ctx.stroke();
  for(const site of measured){
    const [col,row]=cell(site);
    ctx.beginPath();
    ctx.arc((col+.5)*unit,(row+.5)*unit,unit*.18,0,Math.PI*2);
    ctx.fillStyle="#fdfdfa";
    ctx.fill();
    ctx.strokeStyle=view.method.path;
    ctx.lineWidth=1.25;
    ctx.stroke();
  }
}

function paintRefined(){
  if(displayMode!=="refined" || !refinedScene) return;
  const layout=refinedData.layouts[refinedScene.layout];
  const std=layer==="std", step=Math.floor(refinedPosition);
  refinedSlider.value=step;
  refinedSlider.setAttribute("aria-valuetext",`${step} of ${refinedLimit} walking steps`);
  byId("refined-step-label").textContent=`Step ${step} / ${refinedLimit}`;
  byId("std-scale").hidden=!std;
  const snapshots=refinedViews.map(({method})=>{
    const plan=refinedScene.plans[method.key];
    const distance=Math.min(refinedPosition, plan.walk.length-1);
    const n=plan.endpoints.filter(i=>i-1<=distance+1e-9).length;
    return {plan,distance,n,measured:new Set(plan.sites.slice(0,n))};
  });
  refinedViews.forEach((view,i)=>{
    const {method,canvas,stats}=view;
    const {plan,distance,n,measured}=snapshots[i], other=snapshots[1-i].measured;
    const counts=[0,0,0,0];
    for(const site of measured) counts[layout.zones[site]]++;
    const next=plan.endpoints[n]-1 <= refinedLimit ? plan.sites[n] : undefined;
    let phase=distance===plan.endpoints[n-1]-1 ? `Reading ${n}` : "Walking";
    if(distance>=plan.walk.length-1) phase="Route complete";
    else if(distance>=refinedLimit) phase="Comparison distance reached";
    const values={
      layer:std ? "Model uncertainty" : "Predicted field", phase,
      rmse:plan.rmse[n-1].toFixed(4), steps:`${Math.floor(distance)} / ${refinedLimit}`,
      readings:`${n} / ${refinedData.cap}`
    };
    for(const [key,value] of Object.entries(values)){
      if(stats[key].textContent!==value) stats[key].textContent=value;
    }
    const zones=refinedZones.map((name,j)=>`<span>${name} <b>${counts[j]}</b></span>`).join("");
    if(stats.zones.innerHTML!==zones) stats.zones.innerHTML=zones;
    const highlight={path:method.path,fill:site=>other.has(site) ? "#fafff5" : method.stop};
    drawWorld(canvas,layout,plan,n,distance,next,plan[layer][n-1],highlight,
              v=>color(v,std,std ? refinedCeiling : refinedExtent));
    drawRefinedUncertainty(view,layout,plan.std[n-1],measured);
  });
  drawWorld(refinedTruth,layout,null,0,0,undefined,refinedScene.truth,null,
            v=>color(v,false,refinedExtent));
}

function seekRefined(step){
  stopRefined();
  refinedPosition=step;
  paintRefined();
}

["refined-layout","refined-field","refined-budget"].map(byId).forEach(
  element=>element.addEventListener("change",loadRefined));
refinedSlider.addEventListener("input",()=>seekRefined(Number(refinedSlider.value)));
byId("refined-reset").addEventListener("click",()=>seekRefined(0));
byId("refined-end").addEventListener("click",()=>seekRefined(refinedLimit));
refinedPlay.addEventListener("click",()=>{
  if(refinedAnimation!==null){stopRefined();return;}
  if(refinedPosition>=refinedLimit) seekRefined(0);
  const initial=refinedPosition, started=performance.now();
  const reduce=window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  refinedPlay.textContent="Ⅱ Pause";
  const tick=now=>{
    const position=Math.max(0, initial+(now-started)/150);
    refinedPosition=Math.min(refinedLimit,reduce ? Math.floor(position) : position);
    paintRefined();
    if(refinedPosition>=refinedLimit) stopRefined();
    else refinedAnimation=requestAnimationFrame(tick);
  };
  refinedAnimation=requestAnimationFrame(tick);
});
