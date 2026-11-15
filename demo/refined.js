// Distance-synchronized replay of the refined corridor benchmark.
const refinedMethods = [
  {key:"fine_candidates", label:"Greedy", path:"#663694", stop:"#d0a2f7"},
  {key:"fine_rollout", label:"Rollout", path:"#328b7c", stop:"#6cccbb"}
];
const coverageMethods = [
  {...refinedMethods[0], key:"quarter_156", label:"Existing rollout"},
  {...refinedMethods[1], key:"room_quota_two_156", label:"Minimum two per room"}
];
const refinedNames = {short:"Short range", mixed:"Mixed by area", long:"Long range"};
const refinedZones = ["Lower-left room","Lower-right room","Corridor","Upper room"];
const refinedSlider = byId("refined-step");
const refinedPlay = byId("refined-play");
const refinedStudy = byId("refined-study");
let refinedScene, refinedLayout, refinedLimit, refinedExtent, refinedCeiling, refinedPosition=0, refinedAnimation=null;
let showingRoomCoverage;
let refinedViews=[], refinedTruth;

function stopRefined(){
  if(refinedAnimation!==null) cancelAnimationFrame(refinedAnimation);
  refinedAnimation=null;
  refinedPlay.textContent="▶ Play";
}

function mountRefined(methods, coverage){
  const container=byId("refined-cards");
  container.querySelectorAll(".viewport canvas").forEach(canvas=>observer.unobserve(canvas));
  container.innerHTML=methods.map(method=>`
    <article class="duel-card" style="--route:${method.path}">
      <h2>${method.label}</h2>
      <div class="viewport">
        <canvas role="img" aria-label="${method.label} 3D route and reconstructed map"></canvas>
        <div class="viewport-ui"><span class="viewport-label">Predicted field</span></div>
        <div class="viewport-bottom"><span></span><span class="sensor-state" data-stat="phase"></span></div>
      </div>
      <div class="stats${coverage ? " coverage-stats" : ""}">
        <div><span class="key">Hold-out RMSE</span><span class="number" data-stat="rmse"></span></div>
        ${coverage ? '<div><span class="key">Left room interior RMSE</span><span class="number" data-stat="interior"></span></div>' : ""}
        <div><span class="key">Steps walked</span><span class="number" data-stat="steps"></span></div>
        <div><span class="key">Readings</span><span class="number" data-stat="readings"></span></div>
      </div>
      <div class="rooms" aria-label="Readings by area" data-stat="zones"></div>
      ${uncertaintyPanel(method.label)}
    </article>`).join("")+`
    <article class="duel-card refined-truth" style="--route:#475e53">
      <h2>True field</h2>
      <div class="viewport"><canvas role="img" aria-label="True 3D sensor field"></canvas></div>
      <p class="desc">${coverage
        ? "The lower-left room, upper room, and corridor have short sensor range. The lower-right room has long range."
        : "Small dots mark the 100 hold-out sites used to score the maps."}</p>
    </article>`;
  refinedViews=methods.map((method,i)=>{
    const card=container.children[i];
    return {method, canvas:card.querySelector(".viewport canvas"),
      uncertainty:bindUncertainty(card),
      stats:Object.fromEntries([...card.querySelectorAll("[data-stat]")].map(node=>[node.dataset.stat,node]))};
  });
  refinedTruth=container.querySelector(".refined-truth canvas");
  [...refinedViews.map(view=>view.canvas),refinedTruth].forEach(registerWorldCanvas);
}

function showCoverageResults(){
  const {fields,scores,evaluationCells,interiorCells}=refinedScene;
  const rows=scores.map(({budget,overall,short})=>`<tr aria-current="${budget===refinedLimit}">
    <th scope="row">${budget} steps</th>${[...overall,...short].map(value=>`<td>${value.toFixed(4)}</td>`).join("")}
    </tr>`).join("");
  byId("refined-results").innerHTML=`<thead><tr><th scope="col">Budget</th><th scope="col">Whole map · existing</th>
    <th scope="col">Whole map · min. two</th><th scope="col">Short areas · existing</th>
    <th scope="col">Short areas · min. two</th></tr></thead><tbody>${rows}</tbody>`;
  byId("refined-result-note").textContent=`Mean RMSE across ${fields} paired mixed-range fields with weak cross-wall dependence. The replay shows one field.`;
  byId("refined-result-detail").textContent=`This replay scores ${evaluationCells} cells unmeasured by both completed routes. The left-room interior score uses the ${interiorCells} of those cells at least five walking steps from its doorway.`;
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
  const coverage=Boolean(refinedData.roomCoverage && refinedStudy.value==="coverage");
  if(coverage!==showingRoomCoverage){
    showingRoomCoverage=coverage;
    mountRefined(coverage ? coverageMethods : refinedMethods, coverage);
  }
  byId("refined-selects").classList.toggle("coverage-controls",coverage);
  document.querySelectorAll("[data-greedy-control]").forEach(label=>label.hidden=coverage);
  byId("refined-left-label").textContent=coverage ? "Existing rollout only" : "Greedy only";
  byId("refined-right-label").textContent=coverage ? "Minimum two only" : "Rollout only";
  byId("refined-report").href=coverage
    ? "../docs/residential-floor-plan-study.md#increasing-the-walking-budget-with-a-fixed-reading-cap"
    : "../docs/residential-floor-plan-study.md#remaining-budget-rollout-on-the-refined-grid";
  const layout=Number(byId("refined-layout").value), scenario=byId("refined-field").value;
  const matched=!coverage && byId("refined-budget").value==="matched";
  refinedScene=coverage ? refinedData.roomCoverage : refinedData.cases.find(c=>c.layout===layout && c.scenario===scenario);
  refinedLayout=coverage ? refinedScene.layout : refinedData.layouts[layout];
  refinedLimit=coverage ? refinedScene.travel : matched ? refinedScene.cutoff : refinedData.travel;
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
  byId("refined-explanation").textContent=coverage
    ? "Both routes walk 156 steps and take 24 readings. Compare how they sample the lower-left room."
    : matched ? "The original rollout is replayed up to greedy’s walking distance. Drag any view to rotate together."
              : "Both routes start together and use the same caps. Drag any view to rotate together.";
  if(coverage) showCoverageResults();
  else showRefinedResults(matched);
  paintRefined();
}

function paintRefined(){
  if(displayMode!=="refined" || !refinedScene) return;
  const layout=refinedLayout;
  const step=Math.floor(refinedPosition);
  refinedSlider.value=step;
  refinedSlider.setAttribute("aria-valuetext",`${step} of ${refinedLimit} walking steps`);
  byId("refined-step-label").textContent=`Step ${step} / ${refinedLimit}`;
  const snapshots=refinedViews.map(({method})=>{
    const plan=refinedScene.plans[method.key];
    const distance=Math.min(refinedPosition, plan.walk.length-1);
    const n=plan.endpoints.filter(i=>i-1<=distance+1e-9).length;
    return {plan,distance,n,measured:new Set(plan.sites.slice(0,n))};
  });
  refinedViews.forEach((view,i)=>{
    const {method,canvas,stats,uncertainty}=view;
    const {plan,distance,n,measured}=snapshots[i], other=snapshots[1-i].measured;
    const counts=[0,0,0,0];
    for(const site of measured) counts[layout.zones[site]]++;
    const next=plan.endpoints[n]-1 <= refinedLimit ? plan.sites[n] : undefined;
    let phase=distance===plan.endpoints[n-1]-1 ? `Reading ${n}` : "Walking";
    if(distance>=plan.walk.length-1) phase="Route complete";
    else if(distance>=refinedLimit) phase="Comparison distance reached";
    const values={
      phase,
      rmse:plan.rmse[n-1].toFixed(4), steps:`${Math.floor(distance)} / ${refinedLimit}`,
      readings:`${n} / ${refinedData.cap}`
    };
    if(showingRoomCoverage) values.interior=plan.interior_rmse[n-1].toFixed(4);
    for(const [key,value] of Object.entries(values)){
      if(stats[key].textContent!==value) stats[key].textContent=value;
    }
    const zones=refinedZones.map((name,j)=>`<span>${name} <b>${counts[j]}</b></span>`).join("");
    if(stats.zones.innerHTML!==zones) stats.zones.innerHTML=zones;
    const highlight={path:method.path,fill:site=>other.has(site) ? "#fafff5" : method.stop};
    drawWorld(canvas,layout,plan,n,distance,next,plan.mean[n-1],highlight,
              v=>color(v,false,refinedExtent));
    paintUncertainty(uncertainty,layout,plan.std[n-1],refinedCeiling,measured,method.path);
  });
  drawWorld(refinedTruth,layout,null,0,0,undefined,refinedScene.truth,null,
            v=>color(v,false,refinedExtent));
}

function seekRefined(step){
  stopRefined();
  refinedPosition=step;
  paintRefined();
}

byId("refined-study-label").hidden=!refinedData?.roomCoverage;
byId("refined-selects").classList.toggle("has-study-selector",Boolean(refinedData?.roomCoverage));
refinedStudy.value=refinedData?.roomCoverage && location.hash!=="#greedy-rollout" ? "coverage" : "greedy";
["refined-study","refined-layout","refined-field","refined-budget"].map(byId).forEach(
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
    const position=Math.max(0, initial+(now-started)/(showingRoomCoverage ? 70 : 150));
    refinedPosition=Math.min(refinedLimit,reduce ? Math.floor(position) : position);
    paintRefined();
    if(refinedPosition>=refinedLimit) stopRefined();
    else refinedAnimation=requestAnimationFrame(tick);
  };
  refinedAnimation=requestAnimationFrame(tick);
});
