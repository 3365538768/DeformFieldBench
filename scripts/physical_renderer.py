"""Small deterministic, depth-tested SI rasterizer for geometry/flow diagnostics.

No displacement amplification, GT-dependent appearance, or photorealism claim.
RGB uses a fixed material-coordinate texture and simple two-sided shading.
"""
import numpy as np


def exterior_triangles(tets):
    faces=np.concatenate([tets[:,indices] for indices in ([0,1,2],[0,1,3],[0,2,3],[1,2,3])])
    unique,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
    return unique[counts==1]


def camera_basis(yaw, pitch):
    c,s,cp,sp=np.cos(yaw),np.sin(yaw),np.cos(pitch),np.sin(pitch)
    return np.array([[c,0,s],[sp*s,cp,-sp*c],[-cp*s,sp,cp*c]])


def project(points, camera):
    xyz=(points-np.asarray(camera['center_m']))@camera_basis(camera['yaw'],camera['pitch']).T
    scale=camera['image_size']/camera['field_of_view_m']
    return np.column_stack((camera['image_size']/2+xyz[:,0]*scale,
                            camera['image_size']/2-xyz[:,1]*scale,xyz[:,2]))


def rasterize(rest, positions, triangles, camera, texture_seed=17):
    size=camera['image_size'];projected=project(positions,camera)
    depth=np.full((size,size),-np.inf);face_index=np.full((size,size),-1,dtype=np.int32)
    barycentric=np.zeros((size,size,3),dtype=np.float32)
    for index,face in enumerate(triangles):
        p=projected[face]
        xmin=max(0,int(np.ceil(p[:,0].min()-.5)));xmax=min(size-1,int(np.floor(p[:,0].max()-.5)))
        ymin=max(0,int(np.ceil(p[:,1].min()-.5)));ymax=min(size-1,int(np.floor(p[:,1].max()-.5)))
        if xmin>xmax or ymin>ymax:continue
        x,y=np.meshgrid(np.arange(xmin,xmax+1)+.5,np.arange(ymin,ymax+1)+.5)
        a,b,c=p[:,:2];den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12:continue
        w0=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        w1=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        weights=np.stack((w0,w1,1-w0-w1),axis=-1)
        z=weights@p[:,2]
        target=depth[ymin:ymax+1,xmin:xmax+1]
        valid=(weights.min(axis=-1)>=-1e-8)&(z>target)
        target[valid]=z[valid]
        face_index[ymin:ymax+1,xmin:xmax+1][valid]=index
        barycentric[ymin:ymax+1,xmin:xmax+1][valid]=weights[valid]
    mask=face_index>=0
    yy,xx=np.mgrid[:size,:size]
    rgb=np.stack((24+4*xx/size,30+4*yy/size,39+4*xx/size),axis=-1)
    faces=triangles[face_index[mask]];weights=barycentric[mask]
    material=np.einsum('ni,nij->nj',weights,rest[faces])
    rng=np.random.default_rng(texture_seed)
    phase=rng.uniform(0,2*np.pi,3)
    # Texture depends only on reference coordinates and an independent seed.
    tex=.52+.18*np.sin(material[:,0]*1800+phase[0])*np.sin(material[:,1]*2100+phase[1])
    tex+=.12*np.sin((material[:,0]+material[:,2])*2700+phase[2])
    normal=np.cross(positions[faces[:,1]]-positions[faces[:,0]],positions[faces[:,2]]-positions[faces[:,0]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-15)
    light=np.array([.3,.5,.8]);light/=np.linalg.norm(light)
    shade=.65+.35*np.abs(normal@light)
    base=np.array([105.,170.,205.])+rng.uniform(-15,15,3)
    rgb[mask]=base[None,:]*(.55+tex[:,None]*.65)*shade[:,None]
    return dict(rgb=np.clip(rgb,0,255).astype(np.uint8),depth=depth,mask=mask,
                face_index=face_index,barycentric=barycentric,projected=projected)


def correspondence_flow(source, target, triangles, target_positions, camera):
    """Forward correspondence on visible source material points, with z visibility."""
    size=camera['image_size'];flow=np.zeros((size,size,2),dtype=np.float32)
    valid=np.zeros((size,size),dtype=bool)
    y,x=np.where(source['mask'])
    faces=triangles[source['face_index'][y,x]]
    positions=np.einsum('ni,nij->nj',source['barycentric'][y,x],target_positions[faces])
    q=project(positions,camera)
    u=np.floor(q[:,0]).astype(int);v=np.floor(q[:,1]).astype(int)
    inside=(u>=0)&(u<size)&(v>=0)&(v<size)
    safe_u=np.clip(u,0,size-1);safe_v=np.clip(v,0,size-1)
    target_faces=target['face_index'][safe_v,safe_u]
    # Conservative visibility. Same triangle or close depth on its neighbours.
    delta=np.abs(target['depth'][safe_v,safe_u]-q[:,2])
    visible=inside&(target_faces>=0)&((target_faces==source['face_index'][y,x])|(delta<camera['field_of_view_m']/size))
    flow[y,x]=q[:,:2]-np.column_stack((x+.5,y+.5))
    valid[y,x]=visible
    return flow,valid


def overlay_occluder(rgb, object_mask, variant):
    """Paired image-space occlusion augmentation, NOT a contact/hand simulation."""
    mask=np.zeros(object_mask.shape,dtype=bool)
    if variant=='clean':return rgb,mask
    if variant not in {'moderate','heavy'}:raise ValueError('Unknown occlusion variant')
    y,x=np.where(object_mask)
    if not len(x):raise ValueError('No reference object pixels')
    width=x.max()-x.min()+1
    band=max(1,round(width*(.28 if variant=='moderate' else .52)))
    center=round(x.min()+width*.62)
    lo=max(0,center-band//2);hi=min(rgb.shape[1],lo+band)
    mask[:min(rgb.shape[0],y.max()+5),lo:hi]=True
    result=rgb.copy();result[mask]=[145,124,108]
    result[:min(rgb.shape[0],y.max()+5),lo:min(hi,lo+2)]=[188,163,139]
    return result,mask
