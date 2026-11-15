// Orthographic 3D replay drawn from the archived grid and walking path.
function uncertaintyPanel(label) {
  return `<section class="uncertainty-panel">
    <h3>Uncertainty</h3>
    <canvas width="288" height="288" role="img" aria-label="${label} top-down uncertainty map"></canvas>
    <div class="legend" aria-label="Posterior standard deviation color scale">
      <span>0</span><span class="ramp uncertainty"></span><span data-uncertainty-max></span><span>Std. dev.</span>
    </div>
  </section>`;
}

function bindUncertainty(container) {
  const panel=container.querySelector(".uncertainty-panel");
  return {canvas:panel.querySelector("canvas"), heading:panel.querySelector("h3"),
          maximum:panel.querySelector("[data-uncertainty-max]")};
}

function wallSegments(layout) {
  if(layout.walls) return layout.walls;
  return layout.doors.flatMap((door,i)=>{
    const x=i ? 9 : 3, opening=11-door;
    return [[x,0,x,opening],[x,opening+1,x,12]];
  });
}

function paintUncertainty(panel, layout, values, bound, measured, stopColor) {
  if(panel.values===values && panel.bound===bound) return;
  const {canvas}=panel, ctx=canvas.getContext("2d");
  if(!ctx) return;
  panel.values=values;
  panel.bound=bound;
  panel.maximum.textContent=bound.toFixed(1);
  const size=layout.size ?? 12, unit=canvas.width/size;
  const cell=site=>[site%size,size-1-Math.floor(site/size)];
  for(let site=0;site<values.length;site++){
    const [col,row]=cell(site);
    const x=Math.round(col*unit), y=Math.round(row*unit);
    ctx.fillStyle=color(values[site]/1000,true,bound);
    ctx.fillRect(x,y,Math.round((col+1)*unit)-x,Math.round((row+1)*unit)-y);
  }
  ctx.beginPath();
  for(const [x0,y0,x1,y1] of wallSegments(layout)){
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
    ctx.strokeStyle=stopColor;
    ctx.lineWidth=1.25;
    ctx.stroke();
  }
}

const worldCanvas = byId("world-canvas");
const worldUncertaintyContainer=byId("world-uncertainty");
worldUncertaintyContainer.innerHTML=uncertaintyPanel("3D world");
const worldUncertainty=bindUncertainty(worldUncertaintyContainer);
const duelCards = byId("duel-cards");
duelCards.innerHTML = rolloutMethods.map(m=>`
  <article class="duel-card" style="--route:${m.color}" aria-labelledby="${m.key}-title">
    <h2 id="${m.key}-title">${m.label}</h2>
    <div class="viewport">
      <canvas id="${m.key}-canvas" role="img" aria-label="${m.label} 3D measurement route"></canvas>
      <div class="viewport-ui"><span class="viewport-label">Predicted field</span></div>
      <div class="viewport-bottom"><span></span><span class="sensor-state" data-stat="stage"></span></div>
    </div>
    <div class="stats">
      <div><span class="key">Hold-out RMSE</span><span class="number" data-stat="rmse"></span></div>
      <div><span class="key">Walking steps</span><span class="number" data-stat="distance"></span></div>
      <div><span class="key">Readings</span><span class="number" data-stat="readings"></span></div>
    </div>
    <div class="rooms" data-stat="rooms"></div>
    ${uncertaintyPanel(m.label)}
  </article>`).join("");
const duelViews = rolloutMethods.map((method,i)=>{
  const card=duelCards.children[i];
  return {method,canvas:card.querySelector(".viewport canvas"),uncertainty:bindUncertainty(card),
    stats:Object.fromEntries([...card.querySelectorAll("[data-stat]")].map(el=>[el.dataset.stat,el]))};
});
const defaultYaw = -.54, pitch = .71;
let yaw = defaultYaw, drag = null, liveCursor, duelCursors, futureIndex = null;
const surfaceLabel = byId("surface-label");
const sensorState = byId("sensor-state");
const forecastPanel = byId("world-forecast");
const forecastValue = byId("forecast-value");
const forecastDetail = byId("forecast-detail");
const scenarioControls = byId("scenario-controls");
const scenarios = [...scenarioControls.querySelectorAll("[data-future]")];
const forecastSymbol = byId("forecast-symbol");
const worldOutput = Object.fromEntries(["rmse", "distance", "readings", "reading", "rooms"]
  .map(key => [key, byId("world-" + key)]));

function paintWorld(cursor) {
  liveCursor = cursor;
  const method = methods.find(m => m.key === worldMethod.value);
  const plan = replay.plans[method.key];
  const n = Math.min(frame, plan.sites.length);
  const distance = cursor ?? plan.endpoints[n-1]-1;
  const wasMoving = cursor !== undefined;
  const value = plan.readings[n-1];
  const counts = [0,0,0];
  for (const site of plan.sites.slice(0,n)) counts[roomOf(site)]++;
  const future = futureIndex===null ? undefined : plan.futures?.[n-1]?.[futureIndex];
  const label = future ? "Simulated prediction" : "Predicted field";
  if (surfaceLabel.textContent !== label) surfaceLabel.textContent = label;
  const status = wasMoving ? "Moving to next stop" : `Reading ${n} · ${value.toFixed(3)}`;
  if (sensorState.textContent !== status) sensorState.textContent = status;
  const fields = {
    rmse:plan.rmse[n-1].toFixed(3), distance:(distance/11).toFixed(3),
    readings:`${n} / ${data.count}`, reading:value.toFixed(3)
  };
  for (const [key, text] of Object.entries(fields)) {
    if (worldOutput[key].textContent !== text) worldOutput[key].textContent = text;
  }
  const rooms = ["Left","Middle","Right"].map((room,i)=>`<span>${room} <b>${counts[i]}</b></span>`).join("");
  if (worldOutput.rooms.innerHTML !== rooms) worldOutput.rooms.innerHTML = rooms;
  const forecast = plan.forecasts?.[n-1];
  forecastPanel.hidden = !plan.forecasts;
  forecastSymbol.hidden = !forecast;
  scenarioControls.hidden = !forecast || !plan.futures;
  for (const button of scenarios) {
    const index = button.dataset.future === "" ? null : Number(button.dataset.future);
    const pressed = String(index === futureIndex);
    if(button.getAttribute("aria-pressed") !== pressed) button.setAttribute("aria-pressed",pressed);
  }
  const detail = future ? `Simulated reading: ${future.reading.toFixed(3)}`
    : forecast ? "Mean ± one standard deviation" : "";
  if(forecastDetail.textContent !== detail) forecastDetail.textContent = detail;
  if (plan.forecasts) {
    const text = forecast ? `${forecast[0].toFixed(3)} ± ${forecast[1].toFixed(3)}` : "Route complete";
    if (forecastValue.textContent !== text) forecastValue.textContent = text;
  }
  const layout=data.layouts[replay.layout];
  drawWorld(worldCanvas,layout,plan,n,distance,forecast ? plan.sites[n] : undefined,future?.mean ?? plan.mean[n-1]);
  worldUncertainty.heading.textContent=future ? "Simulated uncertainty" : "Uncertainty";
  paintUncertainty(worldUncertainty,layout,future?.std ?? plan.std[n-1],ceiling,plan.sites.slice(0,n),method.color);
}

function paintDuel(cursors) {
  duelCursors=cursors;
  const snapshots=duelViews.map(({method})=>{
    const plan=replay.plans[method.key];
    const n=Math.min(frame,plan.sites.length);
    return {plan,n,measured:new Set(plan.sites.slice(0,n))};
  });
  duelViews.forEach(({method,canvas,stats,uncertainty},i)=>{
    const {plan,n,measured}=snapshots[i];
    const other=snapshots[1-i].measured;
    const cursor=cursors?.[i];
    const distance=cursor ?? plan.endpoints[n-1]-1;
    const counts=[0,0,0];
    for(const site of measured) counts[roomOf(site)]++;
    const values={
      stage:cursor!==undefined ? "Moving to next stop" : n===plan.sites.length ? "Route complete" : `Reading ${n}`,
      distance:`${cursor===undefined ? distance : distance.toFixed(1)} / 26`,
      rmse:plan.rmse[n-1].toFixed(3), readings:`${n} / ${data.count}`
    };
    for(const [key,value] of Object.entries(values)){
      if(stats[key].textContent!==value) stats[key].textContent=value;
    }
    const rooms=["Left","Middle","Right"].map((room,j)=>`<span>${room} <b>${counts[j]}</b></span>`).join("");
    if(stats.rooms.innerHTML!==rooms) stats.rooms.innerHTML=rooms;
    const highlight={path:method.path, fill:site=>other.has(site) ? "#fafff5" : method.stop};
    const layout=data.layouts[replay.layout];
    drawWorld(canvas,layout,plan,n,distance,plan.sites[n],plan.mean[n-1],highlight);
    paintUncertainty(uncertainty,layout,plan.std[n-1],ceiling,measured,method.path);
  });
}

function drawWorld(canvas, layout, plan, n, distance, target, floor, highlight,
                   floorColor = value => color(value, false)) {
  const {clientWidth:width, clientHeight:height} = canvas;
  const ctx = canvas.getContext("2d");
  if (!width || !height || !ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (canvas.width !== Math.round(width*dpr) || canvas.height !== Math.round(height*dpr)) {
    canvas.width = Math.round(width*dpr);
    canvas.height = Math.round(height*dpr);
  }
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const backdrop = ctx.createRadialGradient(width*.5,height*.45,20,width*.53,height*.57,width*.68);
  backdrop.addColorStop(0,"#2c4649"); backdrop.addColorStop(1,"#112229");
  ctx.fillStyle = backdrop; ctx.fillRect(0,0,width,height);
  const sin = Math.sin(yaw), cos = Math.cos(yaw);
  const high = Math.sin(pitch), depthScale = Math.cos(pitch);
  const span = 12*(Math.abs(sin)+Math.abs(cos));
  const scale = Math.min((width-30)/span, (height-84)/(span*high+1.5));
  const rotatedX = (x,z) => (x-6)*cos-(z-6)*sin;
  const depth = (x,z) => (x-6)*sin+(z-6)*cos;
  const project = ([x,y,z]) => [width*.5+rotatedX(x,z)*scale,
    height*.53+(depth(x,z)*high-y*depthScale)*scale];
  const size=layout.size ?? 12, unit=12/size;
  const gridCell=site=>[(site%size)*unit,(size-1-Math.floor(site/size))*unit];
  const gridCenter=site=>gridCell(site).map(value=>value+unit/2);

  function trace(vertices) {
    ctx.beginPath();
    vertices.forEach((v,i)=> {
      const point = project(v);
      if (i) ctx.lineTo(...point);
      else ctx.moveTo(...point);
    });
  }
  function poly(vertices, fill, stroke, weight=.75) {
    trace(vertices); ctx.closePath();
    ctx.fillStyle=fill; ctx.fill();
    if (stroke) {
      ctx.strokeStyle=stroke; ctx.lineWidth=weight; ctx.stroke();
    }
  }
  function line(vertices, stroke, weight) {
    trace(vertices); ctx.strokeStyle=stroke; ctx.lineWidth=weight;
    ctx.lineJoin="round"; ctx.lineCap="round"; ctx.stroke();
  }
  function rect(x0,x1,z0,z1,y) {
    return [[x0,y,z0],[x1,y,z0],[x1,y,z1],[x0,y,z1]];
  }
  function box(x0,x1,z0,z1,y0,y1, colors) {
    const [a,b,c,d]=rect(x0,x1,z0,z1,y0);
    const [e,f,g,h]=rect(x0,x1,z0,z1,y1);
    // Draw the two sides facing the camera, followed by the top.
    poly(sin>=0?[b,c,g,f]:[d,a,e,h],colors[0]);
    poly(cos>=0?[c,d,h,g]:[a,b,f,e],colors[1]);
    poly([e,f,g,h],colors[2]);
  }
  function marker(x,z,y,r,fill,stroke) {
    poly(Array.from({length:18},(_,i)=>{
      const t=i*Math.PI/9;
      return [x+r*Math.cos(t),y,z+r*Math.sin(t)];
    }),fill,stroke,1.25);
  }
  // A raised slab, the posterior raster and measurement marks share one geometry.
  box(-.15,12.15,-.15,12.15,-.42,0,["#243e42","#37535a","#c2d1c8"]);
  for (let site=0;site<floor.length;site++) {
    const [x,z]=gridCell(site);
    const fill=floorColor(floor[site]/1000);
    poly(rect(x,x+unit,z,z+unit,0),fill,fill,1);
  }
  for(let j=0;j<=size;j++) {
    const i=j*unit;
    line([[i,.012,0],[i,.012,12]],"#1a343027",.75);
    line([[0,.012,i],[12,.012,i]],"#1a343027",.75);
  }
  let robot;
  if(plan){
    const path=plan.walk.slice(0,Math.floor(distance)+1).map(gridCenter);
    const last=path[path.length-1], next=gridCenter(plan.walk[Math.min(Math.ceil(distance),plan.walk.length-1)]);
    const fraction=distance-Math.floor(distance);
    robot=last.map((v,i)=>v+(next[i]-v)*fraction);
    const positions=[...path,robot].map(([x,z])=>[x,.035,z]);
    line(positions,"#f8fff2dd",Math.max(2,scale*.123));
    line(positions,highlight?.path ?? "#205c50",Math.max(1.4,scale*.067));
    const stopScale=Math.sqrt(unit);
    plan.sites.slice(0,n).forEach((site,i)=>{
      const [x,z]=gridCenter(site);
      if (i===0) poly(rect(x-.15,x+.15,z-.15,z+.15,.06),"#ffc751","#203732");
      else marker(x,z,.06,(highlight ? .19 : .13)*stopScale,highlight?.fill(site) ?? "#fafff5",highlight?.path ?? "#275750");
    });
    if (target !== undefined) {
      const [x,z]=gridCenter(target);
      ctx.setLineDash([4,3]);
      marker(x,z,.065,.32*stopScale,"#f9f4c440","#234841");
      ctx.setLineDash([]);
    }
    if (distance===plan.endpoints[n-1]-1) {
      marker(robot[0],robot[1],.066,.44*stopScale,"#e1f49c64","#e8f8b6");
    }
  } else {
    for(const site of layout.held){
      const [x,z]=gridCenter(site);
      marker(x,z,.03,.045,"#31464a","#f5f6ec");
    }
  }

  const objects=[];
  const add=(x,z,draw)=>objects.push({depth:depth(x,z),draw});
  const wallColors=["#5c7473","#768e88","#c3d4c7"];
  if(layout.walls){
    for(const segment of layout.walls){
      const [x0,z0,x1,z1]=segment.map(value=>value*unit);
      const dx=x0===x1 ? .072 : 0, dz=z0===z1 ? .072 : 0;
      add((x0+x1)/2,(z0+z1)/2,()=>box(x0-dx,x1+dx,z0-dz,z1+dz,.006,.78,wallColors));
    }
  } else layout.doors.forEach((door,i)=>{
    const x=i?9:3, opening=11-door;
    for(let z=0;z<12;z++) {
      if (z===opening) {
        poly(rect(x-.14,x+.14,z+.045,z+.955,.038),"#f9f4c482");
        continue;
      }
      add(x,z+.5,()=>box(x-.095,x+.095,z,z+1,.006,1.17,wallColors));
    }
  });
  // Low borders keep the floor visible when the camera orbits.
  for (let i=0;i<12;i++) {
    add(i+.5,0,()=>box(i,i+1,-.11,0,0,.27,["#53716f","#698681","#bad2c7"]));
    add(i+.5,12,()=>box(i,i+1,12,12.11,0,.27,["#53716f","#698681","#bad2c7"]));
    add(0,i+.5,()=>box(-.11,0,i,i+1,0,.27,["#53716f","#698681","#bad2c7"]));
    add(12,i+.5,()=>box(12,12.11,i,i+1,0,.27,["#53716f","#698681","#bad2c7"]));
  }
  if(robot){
  const [rx,rz]=robot;
  add(rx,rz,()=>{
    marker(rx,rz,.07,.35,"#182e37a0");
    box(rx-.275,rx+.275,rz-.23,rz+.23,.09,.20,["#12252d","#192e35","#385054"]);
    box(rx-.235,rx+.235,rz-.195,rz+.195,.19,.44,["#51746f","#84aca0","#e2f1de"]);
    box(rx-.106,rx+.106,rz-.11,rz+.11,.44,.55,["#658e81","#a0bca2","#fae88a"]);
    marker(rx,rz,.557,.062,"#285e50");
  });
  }
  objects.sort((a,b)=>a.depth-b.depth);
  objects.forEach(object=>object.draw());
}

function redrawWorld() {
  if(displayMode==="refined"){paintRefined();return;}
  if(!replay) return;
  if(displayMode==="duel") paintDuel(duelCursors);
  if(displayMode==="world") paintWorld(liveCursor);
}

function orbit(delta) {
  yaw+=delta;
  redrawWorld();
}

const observer = new ResizeObserver(redrawWorld);
function registerWorldCanvas(canvas) {
  observer.observe(canvas);
  canvas.addEventListener("pointerdown",event=>{
    if (event.button!==0 || drag) return;
    drag={id:event.pointerId,x:event.clientX,canvas};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove",event=>{
    if(drag?.canvas!==canvas || drag.id!==event.pointerId) return;
    orbit((event.clientX-drag.x)*.009);
    drag.x=event.clientX;
  });
  canvas.addEventListener("lostpointercapture",event=>{
    if(drag?.canvas===canvas && drag.id===event.pointerId) drag=null;
  });
}
[worldCanvas, ...duelViews.map(view=>view.canvas)].forEach(registerWorldCanvas);
byId("orbit-left").addEventListener("click",()=>orbit(-.27));
byId("orbit-right").addEventListener("click",()=>orbit(.27));
byId("orbit-reset").addEventListener("click",()=>orbit(defaultYaw-yaw));
for(const button of document.querySelectorAll("[data-orbit]")){
  button.addEventListener("click",()=>orbit(button.dataset.orbit==="reset" ? defaultYaw-yaw : Number(button.dataset.orbit)));
}

for (const button of scenarios) {
  button.addEventListener("click",()=>{
    const index = Math.min(frame, replay.plans[worldMethod.value].sites.length)-1;
    if(!replay.plans[worldMethod.value].futures?.[index]) return;
    stop();
    futureIndex = button.dataset.future === "" ? null : Number(button.dataset.future);
    paint();
  });
}
