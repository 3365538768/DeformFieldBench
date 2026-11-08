'use strict';
(() => {
  const el=id=>document.getElementById(id);
  let clips=[],playing=true,complete=false,busy=false,loaded='';
  function imageUrl(id,asset){return '/api/render?clip='+encodeURIComponent(id)+'&asset='+encodeURIComponent(asset);}
  function display(){
    const row=clips.find(r=>r.id===el('render-case').value);if(!row)return;
    el('render-images').hidden=false;el('render-play').disabled=false;
    el('render-video').src=imageUrl(row.id,playing?'preview.gif':'final.png');
    el('render-contact').src=imageUrl(row.id,'contact_sheet.png');
    loaded=row.id;
    const m=row.metrics;
    const sampling=row.sampling;
    if(row.motion_supervision){el('render-status').textContent=`位移监督示例 · ${row.object_id} · 仅训练集 · 对应覆盖 ${(m.mean_valid_object_fraction*100).toFixed(1)}% / 平均位移 ${m.mean_displacement_px.toFixed(2)} px。RGB 使用实际帧和时间；第三图为当前→首帧 DIS 像素位移，固定 0–32px 色标；红色区域无位移监督。标签使用原始物体掩码，不是三维速度、接触力或应力真值。`;return;}
    if(row.physical_tool_contact){el('render-status').textContent=`真实三维压头接触预检 · ${row.case.shape} / ${row.case.layout} · ${row.case.young_pa.map(x=>x/1000).join('/')} kPa · ${row.frame_times_s.length} 个真实帧 · 形变 1× · 峰值可见运动 P95 ${m.final_motion_p95_px.toFixed(2)} px。遮挡来自同一物理工具的深度测试；位移驱动不等于已知力，尚缺网格/时间步、接触力与能量验收，未准入训练。`;return;}
    el('render-status').textContent=`${clips.length} 个已完成视频 · E ${row.young_pa/1000} kPa / ν ${row.poisson} · 力 ${row.force_n} N · 网格 ${row.case.cells.join('×')} · ${row.camera.name} 相机 · ${row.frame_times_s.length} 个真实帧 / ${row.frame_times_s.at(-1).toFixed(2)} s · 形变 1×。${sampling?'原轨迹 '+sampling.source_duration_s.toFixed(2)+' s，裁去 '+sampling.dropped_static_tail_frames+' 个尾部静帧；有效运动转场 '+(sampling.selected_active_transition_fraction*100).toFixed(0)+'%。':''}可见对应点 ${m.visible_correspondence_pixels} 像素（${(m.visible_fraction*100).toFixed(1)}%）· ${m.motion_reference==='initial_to_peak_visible_correspondence'?'初始→峰值':'初始→最终'}运动 P50 / P95 / 最大值 ${m.final_motion_p50_px.toFixed(2)} / ${m.final_motion_p95_px.toFixed(2)} / ${m.final_motion_max_px.toFixed(2)} px。${m.observed_object_fraction!=null?'物体可见比例 '+(m.observed_object_fraction*100).toFixed(1)+'%；':''}${row.occlusion_variant&&row.occlusion_variant!=='clean'?'图像遮挡增强，尚无三维手/夹具接触仿真。':''}${row.eligible_for_video_pilot?'通过受控视频小样本筛查。':'运动或可见性不足，暂不进入视频训练。'}`;
  }
  async function refresh(){
    if(busy)return;busy=true;el('render-refresh').disabled=true;
    try{
      const r=await fetch('/api/render');const data=await r.json();if(!r.ok)throw new Error(data.error||'视频读取失败');
      complete=data.status==='complete';clips=data.clips||[];
      if(!clips.length){el('render-status').textContent=data.detail||'等待网格收敛验收；通过后才渲染视频。';return;}
      const previous=el('render-case').value;
      const occlusionNames={clean:'无遮挡',moderate:'约 28% 遮挡增强',heavy:'约 52% 遮挡增强'};
      el('render-case').replaceChildren(...clips.map(row=>{const o=document.createElement('option');o.value=row.id;o.textContent=row.motion_supervision?`位移伪标签 · ${row.object_id} · 对应覆盖 ${(row.metrics.mean_valid_object_fraction*100).toFixed(0)}%`:row.physical_tool_contact?`三维压头 · ${row.case.shape} · ${row.case.layout} · 物理预检/未准入训练`:`E ${row.young_pa/1000}kPa · 力 ${row.force_n}N · ${row.camera.name} · ${row.interaction==='load_hold_release'?'加载卸载':'恒载'} · ${occlusionNames[row.occlusion_variant]||'原版'} · ${row.eligible_for_video_pilot?'可用':'待改进'}`;return o;}));
      const preferred=clips.filter(row=>row.eligible_for_video_pilot&&(!row.occlusion_variant||row.occlusion_variant==='clean')).sort((a,b)=>b.metrics.final_motion_p95_px-a.metrics.final_motion_p95_px)[0]||clips[0];
      el('render-case').value=clips.some(row=>row.id===previous)?previous:preferred.id;
      if(loaded!==el('render-case').value)display();
    }catch(e){el('render-status').textContent='渲染预览读取失败：'+e.message;}
    finally{busy=false;el('render-refresh').disabled=false;}
  }
  el('render-case').addEventListener('change',display);
  el('render-refresh').addEventListener('click',()=>{loaded='';refresh();});
  el('render-play').addEventListener('click',()=>{playing=!playing;el('render-play').textContent=playing?'暂停动画':'播放实际轨迹';display();});
  for(const id of ['render-video','render-contact'])el(id).addEventListener('error',()=>{el('render-status').textContent='图像尚未读取成功，可点击刷新视频重试。';});
  setInterval(()=>{if(!complete&&!document.hidden)refresh();},15000);refresh();
})();
