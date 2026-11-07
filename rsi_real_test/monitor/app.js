'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (n, digits = 1) => n == null || !Number.isFinite(Number(n)) ? '—' : Number(n).toLocaleString('zh-CN', {maximumFractionDigits:digits, minimumFractionDigits:digits});
const integer = n => num(n, 0);
const pct = (a,b) => b ? Math.max(0,Math.min(100,a/b*100)) : 0;
const bytes = (n, digits = 1) => {if(n == null)return '—';const units=['B','KiB','MiB','GiB','TiB'];let i=0;while(n>=1024&&i<4){n/=1024;i++;}return `${num(n,digits)} ${units[i]}`;};
const duration = n => {if(n==null)return '—';n=Math.max(0,n);if(n<60)return `${Math.floor(n)} 秒`;if(n<3600)return `${Math.floor(n/60)} 分`;return `${Math.floor(n/3600)} 小时 ${Math.floor(n%3600/60)} 分`;};
const clock = seconds => new Date(seconds*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});
const colors = ['#7cafff','#ffb574','#59d6bb','#d3a3ff'];
const phaseNames={initializing:'加载模型',field_warmup:'场预测预热',joint_training:'联合训练',validation_or_checkpoint:'验证 / 保存检查点',validation:'分片验证',paused:'检查点已暂停',finished:'已完成'};
const statusNames={running:'运行中',finished:'已完成',failed:'异常退出',stopped:'进程未运行',paused:'已暂停',migrated:'已安全迁移',early_stopped:'停滞早停',queued:'等待资源',held_for_g1:'等待 G1 验收'};
let current=null,selectedRun='',busy=false,controlBusy=false,controlMessage='';

function experimentLabel(j){const code=j.name.includes('C_aux')?'C':j.name.includes('D_field')?'D':j.name;const desc={none:'无场反馈',fields:'场反馈',features:'特征反馈',oracle:'特权真场上界',oracle_shuffled:'跨物体真场置乱'}[j.feedback]||j.feedback;const suffix=j.purpose==='research'?'':j.purpose==='diagnostic'?' · G1 诊断':j.contact_loss==='balanced_bce'?' · 平衡试跑':' · 初始试跑';return `${code} · ${desc}${suffix}`;}
function visibleJobs(s){return s.experiments.filter(j=>$('experiment-filter').value==='all'||($('experiment-filter').value==='diagnostic'?j.purpose==='diagnostic':j.purpose==='research'));}

function renderOptimization(state){
  const o=state.snapshot?.optimization;
  const disabled=state.stale||controlBusy||!o?.version||(o.updated_unix&&state.snapshot.server_unix-o.updated_unix>20);
  $('pause-all').disabled=disabled||o?.desired==='paused';$('resume-all').disabled=disabled||o?.desired!=='paused';
  if(!o?.version){$('control-status').textContent='优化队列尚未注册；资源与旧实验仍可查看。';return;}
  const running=o.jobs.filter(j=>j.state==='running').length+o.tasks.filter(j=>j.state==='running').length+(o.drain?.jobs||[]).filter(j=>j.status==='waiting_checkpoint').length;
  const heartbeat=state.snapshot.server_unix-o.updated_unix;
  const managerStale=heartbeat>20;
  $('control-status').textContent=controlMessage||(managerStale?'队列管理器心跳已过期，请检查最近日志。':o.desired==='paused'?(running?'已请求暂停，等待当前检查点 / 验证任务结束。':'优化队列已暂停；GPU 训练进程已退出。'):`优化队列运行中 · ${running} 项任务执行中 · ${clock(o.updated_unix)}`);
  const stageName={passed:'验收通过',failed:'验收失败',running:'进行中',queued:'待开始',complete:'诊断完成',planned:'已规划'};
  $('stages').innerHTML=o.stages.map(s=>`<article class="panel stage-card"><div class="gpu-head"><strong>${esc(s.id)} · ${esc(s.title)}</strong><span class="badge ${esc(s.state)}">${stageName[s.state]||esc(s.state)}</span></div><p class="chart-note">${esc(s.detail)}</p>${s.total?`<progress max="${s.total}" value="${s.completed}" aria-label="${esc(s.id)} 验收项进度"></progress><div class="muted">${s.completed} / ${s.total} 项</div>`:''}</article>`).join('');
  const b=o.benchmark,cache=o.cache;
  $('optimization-evidence').innerHTML=`CPU 契约测试：<strong>${o.cpu_tests_passed?'14 项通过':'待验收'}</strong> · 旧实验迁移：${esc(o.drain?.status==='drained'?'完整检查点已保存':o.drain?.status||'等待中')}${b?`<br>G0 同检查点 ${b.samples} 条验证：FP32 最大预测差 ${num(b.checks.distributed_fp32.max_z_delta,6)} · BF16 MAE 差 ${num(b.checks.distributed_bf16.mae_delta,6)} · 本次验证加速 ${num(b.checks.distributed_bf16.validation_speedup,2)}×（小样本实测）`:''}${cache?`<br>G1 冻结 RGB 特征缓存：${integer(cache.completed)} / ${integer(cache.total)} · ${esc(cache.status)}${cache.bytes?' · '+bytes(cache.bytes):''}`:''}`+o.tasks.filter(t=>t.state==='running').map(t=>`<br>执行任务：${esc(t.id)} · GPU ${esc(t.devices)}${t.progress?.length?' · 验证 '+t.progress.reduce((a,v)=>a+v.processed,0)+' / '+t.progress.reduce((a,v)=>a+v.total,0):''}`).join('');
  $('queue').innerHTML=o.jobs.map(j=>{const exp=state.snapshot.experiments.find(e=>e.id===j.id),r=exp?.runtime||j.runtime||{},p=exp?.timing?.at(-1)?.plateau?.material;
    const desired=j.control?.desired||'running',canPause=['running','queued','held_for_g1'].includes(j.state)&&desired!=='paused',canResume=['paused','queued','held_for_g1'].includes(j.state)||desired==='paused';
    const phase=r.phase?` · ${phaseNames[r.phase]||r.phase}`:'';
    const label=j.name.includes('oracle')?' · 特权真场诊断':j.stage==='G1'?' · 单种子诊断':'';
    const pausePending=(desired==='paused'||o.desired==='paused')&&j.state==='running';
    const progress=exp?.phase==='validation'?(exp.val_progress||[]).filter(v=>v.epoch===exp.epoch):[];
    return `<article class="queue-row"><div><strong>${esc(j.name)}${esc(label)}</strong><div class="exp-id">${esc(j.stage)} · ${esc(j.id)}</div><div class="muted">${pausePending?'等待本轮检查点后暂停':desired==='paused'&&j.state!=='running'?'已暂停（等待继续）':esc(statusNames[j.state]||j.state)}${esc(phase)}${r.epoch?' · 第 '+r.epoch+' 轮':''}${progress.length?' · 验证 '+progress.reduce((a,v)=>a+v.processed,0)+' / '+exp.split_counts.val:''}${j.stop_reason?' · '+esc(j.stop_reason):''}${p?' · 无实质改善 '+p.bad_epochs+' 轮':''}</div></div><div class="queue-actions"><button data-id="${esc(j.id)}" data-action="pause" ${disabled||!canPause?'disabled':''}>暂停</button><button data-id="${esc(j.id)}" data-action="resume" ${disabled||!canResume||j.state==='running'&&desired!=='paused'?'disabled':''}>${j.state==='held_for_g1'?'继续（排队）':'继续'}</button></div></article>`;}).join('');
  const feedback=o.diagnostic;
  $('feedback').innerHTML=(feedback?`<strong>G1 反馈：</strong>${esc(feedback.route)}。单种子诊断，不能作为显著提升结论。<br>`:'')+o.events.slice(-4).reverse().map(e=>`${clock(e.time_unix)} · ${esc(e.message)}`).join('<br>');
}

async function sendControl(id,action){
  if(controlBusy||!current)return;controlBusy=true;controlMessage='正在提交'+(action==='pause'?'暂停':'继续')+'请求…';renderOptimization(current);
  try{const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':current.control_token},body:JSON.stringify({id,action})});const value=await response.json();if(!response.ok||!value.ok)throw new Error(value.error||`HTTP ${response.status}`);controlMessage=action==='pause'?'暂停请求已保存，等待当前轮检查点或当前验证任务完成。':'继续请求已保存，队列将在资源可用后恢复。';await sync();}
  catch(e){controlMessage='控制失败：'+e.message;}finally{controlBusy=false;renderOptimization(current);setTimeout(()=>{controlMessage='';if(current)renderOptimization(current);},8000);}
}
function stat(label,value,unit,sub){return `<article class="stat"><div class="stat-label">${label}</div><div class="stat-value">${value}<span class="stat-unit">${unit}</span></div><div class="stat-sub">${sub}</div></article>`;}
function row(label,value,klass=''){return `<div class="row"><span class="row-label">${label}</span><span class="row-value ${klass}">${value}</span></div>`;}
function spark(points){const valid=points.filter(p=>p!=null);if(valid.length<2)return '';const path=valid.map((v,i)=>`${i?'L':'M'}${(i/(valid.length-1)*126+2).toFixed(1)},${(31-v/100*28).toFixed(1)}`).join(' ');return `<svg class="spark" viewBox="0 0 130 34" aria-hidden="true"><path d="${path}" stroke="#59d6bb" stroke-width="1.6" fill="none"/></svg>`;}

function chart(series,xLabel){
  const usable=series.map(s=>({...s,points:s.points.filter(p=>p.x!=null&&p.y!=null&&Number.isFinite(p.y))})).filter(s=>s.points.length);
  if(!usable.length)return '<div class="empty">暂无完整验证轮次或训练数据<br>采集会继续，数据产生后自动显示。</div>';
  const points=usable.flatMap(s=>s.points),W=620,H=220,L=49,R=16,T=15,B=33;
  let minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x));
  let minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));
  const pad=(maxY-minY)*.14||Math.max(.01,maxY*.05);minY=Math.max(0,minY-pad);maxY+=pad;if(maxX===minX){minX=Math.max(0,minX-1);maxX++;}
  const x=v=>L+(v-minX)/(maxX-minX)*(W-L-R),y=v=>T+(maxY-v)/(maxY-minY)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(xLabel)}曲线">`;
  for(let i=0;i<=4;i++){const yy=T+i*(H-T-B)/4,v=maxY-i*(maxY-minY)/4;svg+=`<line class="gridline" x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}"/><text x="${L-9}" y="${yy+4}" text-anchor="end">${num(v,3)}</text>`;}
  for(let i=0;i<=4;i++){const v=minX+i*(maxX-minX)/4;svg+=`<text x="${x(v)}" y="${H-12}" text-anchor="middle">${integer(v)}</text>`;}
  for(const s of usable){const path=s.points.map((p,i)=>`${i?'L':'M'}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join(' ');svg+=`<path d="${path}" stroke="${s.color}" stroke-width="2" fill="none"/>`;for(const p of s.points){svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="${s.points.length<50?3:1.5}" fill="${s.color}"><title>${esc(s.label)} · ${esc(xLabel)} ${p.x} · ${num(p.y,5)}</title></circle>`;}}
  return svg+'</svg>';
}
function averageCurve(curve,key){return curve.map((p,i)=>{const tail=curve.slice(Math.max(0,i-9),i+1).map(v=>v[key]).filter(v=>v!=null);return {x:p.step,y:tail.length?tail.reduce((a,b)=>a+b,0)/tail.length:null};});}

function render(state){
  current=state;const s=state.snapshot;
  const isStale=state.stale;document.body.classList.toggle('stale-data',Boolean(isStale));
  const conn=state.connection==='online'&&!isStale?'online':state.connection==='connecting'?'connecting':'offline';
  $('connection').className='badge '+conn;$('connection').textContent={online:'● SSH 已连接',connecting:'◌ 正在连接',offline:'● 数据已过期'}[conn];
  $('updated').textContent=state.last_success_unix?`最近采集 ${clock(state.last_success_unix)} · 北京时间`:'等待首次采集…';
  $('notice').className='notice'+(state.last_error?' error':isStale?' warning':'');
  $('notice').textContent=state.last_error?`SSH 采集失败，保留最后一次成功数据。${state.last_error}`:!s?'正在通过 SSH 读取 A800，首次采集需要几秒钟。':isStale?`最近数据已超过 ${Math.floor(state.age_seconds)} 秒，请查看连接状态。`:`实时采集正常 · 每 ${state.interval_seconds} 秒通过 SSH 更新 · 支持检查点暂停 / 继续`;
  $('footer-time').textContent=state.collecting?'正在采集新快照…':`采集间隔 ${state.interval_seconds} 秒 · 窗口数据每 5 秒同步`;
  renderOptimization(state);if(!s)return;
  $('project').textContent=s.project;
  const gpu=s.gpus,utils=gpu.map(g=>g.utilization).filter(v=>v!=null),avg=utils.length?utils.reduce((a,b)=>a+b,0)/utils.length:null;
  const used=gpu.reduce((a,g)=>a+(g.memory_used_mib||0),0),total=gpu.reduce((a,g)=>a+(g.memory_total_mib||0),0);
  const active=s.experiments.filter(j=>j.status==='running'&&j.purpose==='research');
  $('overview').innerHTML=stat('GPU 平均利用率',num(avg,0),'%',`${gpu.length} 张 GPU · 忙碌 ${gpu.filter(g=>g.utilization>10).length} 张`)+stat('GPU 显存占用',num(used/1024,1),'GiB',`总容量 ${num(total/1024,0)} GiB · ${num(pct(used,total),1)}%`)+stat('CPU 使用率',num(s.cpu.percent,1),'%',`${s.cpu.cores} 核 · 1 分钟负载 ${num(s.cpu.load[0],1)}`)+stat('主机内存',num(s.memory.used/1024**3,1),'GiB',`可用 ${bytes(s.memory.available)} / ${bytes(s.memory.total)}`)+stat('正式实验运行',integer(active.length),'组',`已完成 ${s.experiments.filter(j=>j.status==='finished'&&j.purpose==='research').length} 组 · 主机运行 ${duration(s.uptime_seconds)}`);
  $('gpus').innerHTML=gpu.length?gpu.map(g=>{const owner=s.experiments.find(j=>j.status==='running'&&j.devices.split(',').includes(String(g.index)));const task=s.optimization?.tasks.find(j=>j.state==='running'&&j.devices.split(',').includes(String(g.index)));return `<article class="gpu-card"><div class="gpu-head"><span class="gpu-name">GPU ${g.index} <span class="muted">A800</span></span><span class="gpu-group">${owner?esc(experimentLabel(owner)):task?esc(task.id):'空闲 / 未分配'}</span></div><div class="gpu-head"><span class="gpu-value">${num(g.utilization,0)}<small>%</small></span>${spark(state.history.map(h=>h.gpus[String(g.index)]))}</div><progress max="100" value="${pct(g.memory_used_mib,g.memory_total_mib)}" aria-label="GPU ${g.index} 显存使用率"></progress><div class="gpu-line"><span>显存 ${num(g.memory_used_mib/1024,1)} / ${num(g.memory_total_mib/1024,0)} GiB</span><span>${num(g.temperature_c,0)}°C</span></div><div class="gpu-line"><span>${num(g.power_w,0)} W / ${num(g.power_limit_w,0)} W</span><span>${g.utilization>10?(g.power_w<150?'忙碌 / 可能等待通信':'GPU 忙碌'):'低负载'}</span></div></article>`;}).join(''):`<div class="empty">GPU 指标暂不可用：${esc(s.gpu_error||'没有返回设备')}</div>`;
  const jobs=visibleJobs(s);
  $('experiments').innerHTML=jobs.length?jobs.map(j=>{const progress=j.status==='finished'?100:pct(j.step,j.total_steps);const age=j.last_update_unix?s.server_unix-j.last_update_unix:null;return `<article class="experiment"><div class="exp-heading"><div><div class="exp-name">${esc(experimentLabel(j))}</div><div class="exp-id">${esc(j.id)}</div></div><span class="badge ${esc(j.status)}">${esc(statusNames[j.status]||j.status)}</span></div><div class="exp-grid"><div><div class="exp-number">${j.epoch}<small> / ${j.epochs}</small></div><div class="exp-label">当前轮次</div></div><div><div class="exp-number">${integer(j.step)}</div><div class="exp-label">累计优化步 / ${integer(j.total_steps)}</div></div><div><div class="exp-number">${num(progress,1)}<small>%</small></div><div class="exp-label">按优化步计</div></div><div><div class="exp-number">${num(j.best_val_mae,4)}</div><div class="exp-label">最佳联合验证 MAE</div></div></div><progress max="100" value="${progress}" aria-label="${esc(j.name)} 训练进度"></progress><div class="exp-meta"><span class="exp-phase">${esc(phaseNames[j.phase]||j.phase)}${j.phase==='field_warmup'?` · 前 ${j.warmup_epochs} 轮`:''}</span><span>GPU ${esc(j.devices)}</span></div><div class="exp-meta"><span>已运行 ${duration(j.elapsed_seconds)}</span><span>${j.status==='running'?`预计剩余 ${duration(j.eta_seconds)}${j.eta_seconds!=null?'（按平均步速）':''}`:'最终记录已保存'}</span></div><div class="exp-meta"><span>训练记录：${age==null?'尚无':duration(age)+'前'}</span><span>${age>900&&j.status==='running'?'较久未更新，请检查验证或日志':''}</span></div><div class="exp-metrics"><span class="chip">${esc(j.backbone)}</span><span class="chip">${j.views} 视角 × ${j.frames} 帧</span><span class="chip">seed ${j.seed}</span><span class="chip">${esc(j.contact_loss)}</span><span class="chip">${j.purpose==='research'?'正式对照':j.purpose==='diagnostic'?'单种子诊断':'工程试跑'}</span></div></article>`;}).join(''):'<div class="empty">没有符合筛选条件的实验。</div>';
  $('protocol').textContent=`数据按物体隔离：训练 ${s.data.split?.train??'—'} / 验证 ${s.data.split?.val??'—'} / 测试 ${s.data.split?.test??'—'}。正式对照保持相同数据、骨干和预算；进程存活与训练记录是否更新分别展示。`;
  const prev=selectedRun||$('run-select').value;const chosen=jobs.find(j=>j.id===prev)||jobs[0];selectedRun=chosen?.id||'';
  $('run-select').innerHTML=jobs.map(j=>`<option value="${esc(j.id)}">${esc(experimentLabel(j))} · ${esc(j.purpose==='research'?'正式':'试跑')} · ${esc(j.id.replace('engineering_pilot_240_','').replace('full_giant16_',''))}</option>`).join('');$('run-select').value=selectedRun;
  $('val-chart').innerHTML=chart(jobs.map((j,i)=>({label:experimentLabel(j),color:colors[i%colors.length],points:j.validation.map(v=>({x:v.epoch,y:v.mae}))})),'轮次');
  $('val-legend').innerHTML=jobs.map((j,i)=>`<span class="legend-item ${['blue','orange','teal','purple'][i%4]}">${esc(experimentLabel(j))}</span>`).join('');
  renderTrain(chosen);
  $('host-details').innerHTML=row('CPU 负载 · 1 / 5 / 15 分钟',s.cpu.load.map(v=>num(v,1)).join(' / '))+row('可用内存 / 总内存',`${bytes(s.memory.available)} / ${bytes(s.memory.total)}`)+row('网络接收 / 发送',`${bytes(s.network_rates.rx_bps)}/s · ${bytes(s.network_rates.tx_bps)}/s`)+s.disks.map(d=>`<div class="disk"><div class="row"><span class="row-label">${esc(d.mount)}</span><span class="row-value ${pct(d.used,d.total)>90?'bad':''}">${num(pct(d.used,d.total),1)}% 已用</span></div><progress max="100" value="${pct(d.used,d.total)}" aria-label="${esc(d.mount)} 磁盘使用率"></progress><div class="gpu-line"><span>已用 ${bytes(d.used)}</span><span>剩余 ${bytes(d.free)}</span></div></div>`).join('');
  const d=s.data,a=d.audit;
  $('data-details').innerHTML=row('全量数据阶段',d.phase.phase==='p0_data_ready'?'准备完成':esc(d.phase.phase||'未知'),d.phase.phase==='p0_data_ready'?'good':'warm')+row('CRC / 格式 / 时间审计',`${integer(a.audited_samples)} 条 · ${d.audit_failures} 失败`,d.audit_failures?'bad':'good')+row('物体 / 本构种类',`${integer(a.objects)} 个 / ${Object.keys(a.material_counts||{}).length} 类`)+row('几何资产覆盖',`${d.geometry.covered_dataset_objects??'—'} / ${d.geometry.dataset_objects??'—'}`)+row('16 / 64 帧资源测量',d.profile.phase==='profile_finished'?'已通过':'待完成',d.profile.phase==='profile_finished'?'good':'warm')+row('测试集评估','未纳入监控曲线')+`<p class="chart-note">数据审计通过不等于物理标定完成；场预测和参数误差改善仍需独立实验验证。</p>`;
  const opened=new Set([...document.querySelectorAll('#logs details[open]')].map(el=>el.dataset.id));
  $('logs').innerHTML=jobs.map(j=>`<details class="logs-panel" data-id="${esc(j.id)}" ${opened.has(j.id)?'open':''}><summary>${esc(experimentLabel(j))} · ${esc(j.output)}</summary><pre>${esc(j.log_tail.join('\n')||'暂无日志输出')}</pre></details>`).join('');
}
function renderTrain(j){$('train-scope').textContent=j?.managed?'跨卡均值 · 最近记录':'rank 0 · 最近记录';$('train-chart').innerHTML=chart(j?[['loss_rank0','总损失'],['parameter','参数损失'],['contact','接触损失']].map(([key,label],i)=>({label,color:colors[i],points:averageCurve(j.train_curve,key)})):[],'优化步');}
async function sync(){if(busy)return;busy=true;$('refresh').disabled=true;try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw new Error(`HTTP ${r.status}`);render(await r.json());}catch(e){$('connection').className='badge offline';$('connection').textContent='● 本地服务断开';$('notice').className='notice error';$('notice').textContent=`无法连接本机监控服务，页面保留旧数据：${e.message}`;document.body.classList.add('stale-data');}finally{busy=false;$('refresh').disabled=false;}}
$('refresh').addEventListener('click',sync);
$('pause-all').addEventListener('click',()=>sendControl('pipeline','pause'));
$('resume-all').addEventListener('click',()=>sendControl('pipeline','resume'));
$('queue').addEventListener('click',event=>{const button=event.target.closest('button[data-action]');if(button&&!button.disabled)sendControl(button.dataset.id,button.dataset.action);});
$('experiment-filter').addEventListener('change',()=>{selectedRun='';if(current)render(current);});
$('run-select').addEventListener('change',()=>{selectedRun=$('run-select').value;renderTrain(current?.snapshot?.experiments.find(j=>j.id===selectedRun));});
sync();setInterval(sync,5000);
