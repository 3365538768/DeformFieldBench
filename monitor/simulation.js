'use strict';
(() => {
  const el=id=>document.getElementById(id), fmt=(v,d=3)=>Number(v).toExponential(d);
  let catalog=[],views=[null,null],selection=0,rotation=[.65,-.35],zoom=1,playing=false,timer=null,request=0,scale=200;
  const canvases=[el('sim-canvas-a'),el('sim-canvas-b')];
  const axes=['X','Y','Z'];
  const norm=v=>Math.hypot(...v);
  const color=t=>{t=Math.max(0,Math.min(1,t));return `hsl(${220-205*t} 82% ${53+8*Math.sin(t*Math.PI)}%)`;};
  function frameFor(data){
    if(data.kind==='static_solution')return 0;
    return Math.round(selection/25*(data.displacement_frames_m.length-1));
  }
  function project(p,data,w,h){
    const dims=data.case.dimensions_m,center=dims.map(v=>v/2),q=p.map((v,i)=>v-center[i]);
    const [yaw,pitch]=rotation,c=Math.cos(yaw),s=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
    const x=c*q[0]+s*q[2],z=-s*q[0]+c*q[2],y=cp*q[1]-sp*z,depth=sp*q[1]+cp*z;
    const size=Math.min(w*.72,h*.72)/Math.max(...dims)*zoom;
    return [w/2+x*size,h/2-y*size,depth];
  }
  function draw(index){
    const canvas=canvases[index],data=views[index],rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;
    if(!w||!h)return;const dpr=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
    ctx.fillStyle='#0c1420';ctx.fillRect(0,0,w,h);
    if(!data){ctx.fillStyle='#92a3ba';ctx.font='14px sans-serif';ctx.fillText('选择案例后显示真实求解结果',25,40);return;}
    const displacement=data.displacement_frames_m[frameFor(data)],rest=data.rest_positions_m;
    const max=Math.max(...views.filter(Boolean).map(v=>v.max_displacement_m),1e-12);
    const positions=rest.map((p,i)=>project(p.map((v,j)=>v+displacement[i][j]*scale),data,w,h));
    if(el('sim-rest').checked){
      ctx.strokeStyle='#66768b';ctx.lineWidth=.65;ctx.globalAlpha=.32;
      const original=rest.map(p=>project(p,data,w,h));ctx.beginPath();
      for(const face of data.surface_triangles){const a=original[face[0]];ctx.moveTo(a[0],a[1]);for(let j=1;j<3;j++){const b=original[face[j]];ctx.lineTo(b[0],b[1]);}ctx.closePath();}ctx.stroke();ctx.globalAlpha=1;
    }
    const faces=data.surface_triangles.map(face=>({face,depth:face.reduce((s,i)=>s+positions[i][2],0)/3})).sort((a,b)=>a.depth-b.depth);
    for(const {face} of faces){
      const magnitude=face.reduce((s,i)=>s+norm(displacement[i]),0)/3;
      ctx.beginPath();ctx.moveTo(...positions[face[0]].slice(0,2));ctx.lineTo(...positions[face[1]].slice(0,2));ctx.lineTo(...positions[face[2]].slice(0,2));ctx.closePath();
      ctx.fillStyle=color(magnitude/max);ctx.globalAlpha=.83;ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle='#09101b88';ctx.lineWidth=.55;ctx.stroke();
    }
    ctx.fillStyle='#ff8692';for(let i=0;i<rest.length;i++)if(rest[i][0]===0){ctx.beginPath();ctx.arc(positions[i][0],positions[i][1],2,0,Math.PI*2);ctx.fill();}
    const origin=[.0,.0,.0],o=project(origin,data,w,h),len=Math.min(...data.case.dimensions_m)*.35;
    for(let i=0;i<3;i++){const p=[0,0,0];p[i]=len;const q=project(p,data,w,h);ctx.strokeStyle=['#ff8692','#59d6bb','#7cafff'][i];ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.font='11px sans-serif';ctx.fillText(axes[i],q[0]+4,q[1]-4);}
    ctx.fillStyle='#b1c2d9';ctx.font='12px sans-serif';ctx.fillText(`形变显示 ${Math.round(scale)}× · 颜色为实际位移`,14,h-16);
    const label=index?'b':'a',health=data.health;
    el('sim-meta-'+label).textContent=`Jmin ${health.min_jacobian.toFixed(6)} · 固定边界误差 ${fmt(health.fixed_vertex_max_error_m)} m · 最大位移 ${fmt(data.max_displacement_m)} m${health.actual_relative_residual!=null?' · ||Ax−b||/||b|| '+fmt(health.actual_relative_residual):''}`;
  }
  function render(){
    draw(0);draw(1);el('sim-amplify-label').textContent=Math.round(scale)+'×';
    const maximum=Math.max(...views.filter(Boolean).map(v=>v.max_displacement_m),0);el('sim-color-max').textContent=fmt(maximum)+' m';
    const dynamic=views.find(v=>v?.kind==='actual_solver_trajectory');
    if(dynamic){const i=frameFor(dynamic);el('sim-time-label').textContent=`实际时间 ${dynamic.frame_times_s[i].toFixed(3)} s · ${i+1}/${dynamic.frame_times_s.length} 帧`;}else el('sim-time-label').textContent='Warp 静态解（无动态帧）';
  }
  function stop(){playing=false;clearInterval(timer);el('sim-play').textContent='播放实际轨迹';}
  async function loadCase(){
    stop();const id=el('sim-case').value,row=catalog.find(r=>r.id===id);if(!row)return;
    const token=++request;el('sim-status').textContent='正在经 SSH 读取已保存的几何与位移场…';
    const pair=catalog.find(r=>r.fixture===row.fixture&&r.material_pair===row.material_pair&&r.axis===row.axis&&r.engine!==row.engine);
    try{
      const ids=[id,...(el('sim-compare').checked&&pair?[pair.id]:[])];
      const data=await Promise.all(ids.map(async cid=>{const r=await fetch('/api/simulation?case='+encodeURIComponent(cid));const v=await r.json();if(!r.ok||v.status!=='ready')throw new Error(v.error||'案例尚未准备完成');return v;}));
      if(token!==request)return;views=[data[0],data[1]||null];selection=25;el('sim-time').value=25;
      el('sim-pane-b').hidden=!views[1];el('sim-views').classList.toggle('single',!views[1]);
      const max=Math.max(...data.map(v=>v.max_displacement_m));scale=Math.max(1,Math.min(1000,Math.max(...row.case.dimensions_m)*.12/Math.max(max,1e-12)));el('sim-amplify').value=Math.log10(scale);
      data.forEach((v,i)=>{const suffix=i?'b':'a';el('sim-title-'+suffix).textContent=v.engine==='uipc'?'libuipc / SNK1':'NVIDIA Warp FEM / 线性弹性';el('sim-type-'+suffix).textContent=`${v.version} · ${v.kind==='static_solution'?'静态解':'实际求解轨迹'}`;});
      const dynamic=data.some(v=>v.kind==='actual_solver_trajectory');el('sim-play').disabled=!dynamic;el('sim-time').disabled=!dynamic;
      const c=row.case;el('sim-status').textContent=(data.some(v=>v.pair_validation==='pending')?'生成中：此案例数值检查通过，跨引擎配对验收尚未完成。 ':'')+`${catalog.length} 个已生成案例 · ${row.source||'G2'} · 夹具 ${row.fixture} / ${row.split==='train'?'训练':row.split==='val'?'验证':'标定审计'} · 尺寸 ${c.dimensions_m.map(v=>(v*1000).toFixed(0)).join(' × ')} mm · E ${(c.young_pa/1000).toFixed(0)} kPa / ν ${c.poisson} · ${axes[c.axis]} 向力 ${fmt(c.force_n,1)} N · 网格 ${c.cells.join(' × ')}。${views[1]?'两侧共享相同几何、载荷和色标。':''}${data[0].source==='G3v'?'理想 XY 投影运动 '+data[0].ideal_max_motion_px.toFixed(2)+' px（224px / 80mm）；单引擎载荷审计，不代表跨引擎等价。':''}`;
      render();
    }catch(e){if(token===request){el('sim-status').textContent='预览读取失败：'+e.message;}}
  }
  function filter(){
    const engine=el('sim-engine').value,previous=el('sim-case').value;
    const source=el('sim-source').value;const rows=catalog.filter(r=>(engine==='all'||r.engine===engine)&&(source==='all'||r.source===source));
    el('sim-case').replaceChildren(...rows.map(r=>{const o=document.createElement('option');o.value=r.id;const c=r.case;o.textContent=`${r.source} · ${r.engine} · 夹具${r.fixture} · ${axes[c.axis]}力 ${c.force_n}N · E=${c.young_pa/1000}kPa / ν=${c.poisson} · ${r.split}`;return o;}));
    if(rows.some(r=>r.id===previous))el('sim-case').value=previous;
    else {const defaultRow=rows.find(r=>r.engine==='uipc'&&r.axis===1&&r.case.young_pa===12000);if(defaultRow)el('sim-case').value=defaultRow.id;}
    loadCase();
  }
  async function refresh(){
    el('sim-refresh').disabled=true;
    try{const r=await fetch('/api/simulation');const data=await r.json();if(!r.ok)throw new Error(data.error||'SSH 读取失败');catalog=data.cases||[];if(!catalog.length){el('sim-status').textContent='响应矩阵生成中；案例通过检查并写入 manifest 后可预览。';return;}filter();}
    catch(e){el('sim-status').textContent='案例目录读取失败：'+e.message;}
    finally{el('sim-refresh').disabled=false;}
  }
  el('sim-refresh').addEventListener('click',refresh);el('sim-source').addEventListener('change',filter);el('sim-engine').addEventListener('change',filter);
  el('sim-case').addEventListener('change',loadCase);el('sim-compare').addEventListener('change',loadCase);
  el('sim-rest').addEventListener('change',render);el('sim-reset').addEventListener('click',()=>{rotation=[.65,-.35];zoom=1;render();});
  el('sim-amplify').addEventListener('input',()=>{scale=10**Number(el('sim-amplify').value);render();});
  el('sim-time').addEventListener('input',()=>{stop();selection=Number(el('sim-time').value);render();});
  el('sim-play').addEventListener('click',()=>{if(playing){stop();return;}playing=true;el('sim-play').textContent='暂停播放';if(selection>=25)selection=0;timer=setInterval(()=>{selection=(selection+1)%26;el('sim-time').value=selection;render();},160);});
  canvases.forEach(canvas=>{
    let drag=null;
    canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove',e=>{if(!drag)return;rotation[0]+=(e.clientX-drag[0])*.008;rotation[1]=Math.max(-1.5,Math.min(1.5,rotation[1]+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];render();});
    canvas.addEventListener('pointerup',()=>{drag=null;});canvas.addEventListener('pointercancel',()=>{drag=null;});
    canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));render();},{passive:false});
  });
  new ResizeObserver(render).observe(el('sim-views'));refresh();
})();
