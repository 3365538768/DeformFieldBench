"""Force schedules, conservative solver stopping and motion-aware frame selection."""
import numpy as np


def force_scale(time_seconds, history=None):
    if history is None:return 1.
    knots=np.asarray(history,dtype=float)
    if knots.ndim!=2 or knots.shape[1]!=2 or len(knots)<2 or not np.isfinite(knots).all() or np.any(np.diff(knots[:,0])<=0):
        raise ValueError('Force history must contain increasing finite [time,scale] knots')
    return float(np.interp(time_seconds,knots[:,0],knots[:,1]))


def should_stop(trace, max_speed, time_seconds, case):
    cfg=case.get('stop_criteria')
    if not cfg:return False
    window=cfg.get('window',20)
    if len(trace)<max(window,cfg.get('minimum_steps',40)):return False
    history=case.get('force_history')
    if history and time_seconds<history[-1][0]:return False  # Never truncate a future action.
    recent=np.asarray(trace[-window:])
    # Normalize to peak response, including an unload-to-zero sequence.
    peak=max(max(float(np.linalg.norm(d,axis=1).max()) for d in trace),1e-12)
    return bool(max_speed<cfg.get('maximum_speed_m_s',1e-6)
        and np.ptp(recent,axis=0).max()/peak<cfg.get('relative_displacement_tolerance',.001))


def select_motion_frames(trace,dt,image_size=224,field_of_view_m=.08,max_frames=17,threshold_px=.05,post_frames=2):
    """Select actual saved frames; trim visually static tail without changing time.

    Motion arc-length sampling retains load transients rather than evenly
    spending most frames on a long hold. Includes rest, peak, endpoint and two
    frames after the last visible increment. No interpolation or time warping.
    """
    frames=np.concatenate((np.zeros_like(trace[:1]),trace),axis=0)
    increments=np.linalg.norm(np.diff(frames,axis=0)[...,:2],axis=2).max(axis=1)*image_size/field_of_view_m
    active=np.flatnonzero(increments>=threshold_px)
    end=min(len(frames)-1,int(active[-1]+1+post_frames)) if len(active) else min(len(frames)-1,post_frames)
    candidates=np.arange(end+1)
    peak=int(np.argmax(np.linalg.norm(frames[:end+1],axis=2).max(axis=1)))
    arc=np.r_[0.,np.cumsum(increments[:end])]
    chosen={0,end,peak}
    if arc[-1]>0:
        chosen.update(int(i) for i in np.searchsorted(arc,np.linspace(0,arc[-1],max_frames)))
    if len(chosen)>max_frames:
        other=sorted(chosen-{0,end,peak})
        selected=np.linspace(0,len(other)-1,max_frames-len({0,end,peak})).round().astype(int)
        chosen={0,end,peak}|{other[i] for i in selected}
    for i in np.linspace(0,end,min(len(candidates),max_frames)).round().astype(int):
        if len(chosen)>=max_frames:break
        chosen.add(int(i))
    indices=np.array(sorted(chosen),dtype=int)
    selected_increments=np.linalg.norm(np.diff(frames[indices],axis=0)[...,:2],axis=2).max(axis=1)*image_size/field_of_view_m
    return indices,dict(source_frames=len(frames),selected_frames=len(indices),source_duration_s=len(trace)*dt,
        retained_duration_s=float(end*dt),dropped_static_tail_frames=len(frames)-1-end,
        source_active_transition_fraction=float((increments>=threshold_px).mean()),
        selected_active_transition_fraction=float((selected_increments>=threshold_px).mean()) if len(selected_increments) else 0.,
        motion_threshold_px=threshold_px,selection='actual frame indices by projected-motion arc length; timestamps unchanged')
