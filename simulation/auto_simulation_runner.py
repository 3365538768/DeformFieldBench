import argparse
import ctypes
import json
import os
import random
import shutil
import subprocess
import sys
import time
import re
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


def _ensure_maca_runtime_libs() -> None:
    maca_root = os.environ.get("MACA_PATH", "/opt/maca-3.3.0")
    cand = [
        os.path.join(maca_root, "tools", "cu-bridge", "lib"),
        os.path.join(maca_root, "lib"),
        os.path.join(maca_root, "ompi", "lib"),
        os.path.join(maca_root, "ucx", "lib"),
    ]
    old = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in old.split(os.pathsep) if p]
    for p in reversed(cand):
        if os.path.exists(p) and p not in parts:
            parts.insert(0, p)
    if parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    # 进程内预加载关键 MPI/UCX 依赖，避免后续 import torch 时找不到 libmpi.so.40
    preload_libs = [
        os.path.join(maca_root, "ompi", "lib", "libopen-pal.so.40"),
        os.path.join(maca_root, "ompi", "lib", "libopen-rte.so.40"),
        os.path.join(maca_root, "ompi", "lib", "libmpi.so.40"),
        os.path.join(maca_root, "ucx", "lib", "libucs.so.0"),
        os.path.join(maca_root, "ucx", "lib", "libuct.so.0"),
        os.path.join(maca_root, "ucx", "lib", "libucp.so.0"),
    ]
    for lib in preload_libs:
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_ensure_maca_runtime_libs()


def _nvidia_smi_gpu_states() -> Optional[List[Tuple[int, int, int]]]:
    """
    查询各物理 GPU 的 index、空闲显存与利用率（动态占卡仅使用利用率）。
    返回 [(index, memory_free_mib, utilization_gpu_pct), ...]；失败返回 None。
    """
    def _from_nvidia_smi() -> Optional[List[Tuple[int, int, int]]]:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=60,
            text=True,
        )
        out: List[Tuple[int, int, int]] = []
        for line in result.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            out.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return out if out else None

    def _from_mx_smi() -> Optional[List[Tuple[int, int, int]]]:
        # mx-smi 不支持 --query-gpu，改为解析文本：GPU#N + vram total/used + Utilization GPU%
        result = subprocess.check_output(
            ["mx-smi", "--show-memory", "--show-usage"],
            stderr=subprocess.DEVNULL,
            timeout=60,
            text=True,
        )
        blocks = re.split(r"\nGPU#(\d+)\s+", result)
        out: List[Tuple[int, int, int]] = []
        for i in range(1, len(blocks), 2):
            gpu_id = int(blocks[i])
            block = blocks[i + 1]
            m_total = re.search(r"vram total\s*:\s*(\d+)\s*KB", block)
            m_used = re.search(r"vram used\s*:\s*(\d+)\s*KB", block)
            m_util = re.search(r"GPU\s*:\s*(\d+)\s*%", block)
            if not (m_total and m_used and m_util):
                continue
            total_kb = int(m_total.group(1))
            used_kb = int(m_used.group(1))
            util = int(m_util.group(1))
            free_mib = max(total_kb - used_kb, 0) // 1024
            out.append((gpu_id, free_mib, util))
        return out if out else None

    # 优先 MetaX，再回退 NVIDIA
    try:
        states = _from_mx_smi()
        if states:
            return states
    except Exception:
        pass
    try:
        return _from_nvidia_smi()
    except Exception:
        return None


def _all_schedulable_physical_gpu_ids(torch_device_count: int) -> List[int]:
    """
    本机可参与调度的全部物理 GPU 编号（与 nvidia-smi index 一致）。
    不在此列表截断：--num_gpus 表示「最多同时跑几个任务」，而不是「只用前 N 张卡」，
    否则会出现在 GPU0–2 挤爆时无法去占 GPU5/7 上大量空闲显存的情况。
    若已设置 CUDA_VISIBLE_DEVICES，则仅使用其中的编号（逗号分隔整数）。
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        ids: List[int] = []
        for p in raw.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                ids.append(int(p))
            except ValueError:
                continue
        return ids
    states = _nvidia_smi_gpu_states()
    if states:
        return [s[0] for s in states]
    n = max(int(torch_device_count), 1)
    return list(range(n))


def _physical_gpu_looks_idle(
    gpu_id: int,
    states: Optional[List[Tuple[int, int, int]]],
    max_util_pct: int,
    min_free_mib: int,
    *,
    relax_util_when_free_mib: int,
    smi_failed_assume_idle: bool,
) -> bool:
    """
    是否可占卡：须满足显存下限；利用率一般须 ≤max_util_pct。
    若剩余显存 ≥ relax_util_when_free_mib（如 40GiB 级），则不再卡利用率——
    适用于 MPS/多进程共享显存、他人在跑算力但仍有大块空闲显存的 A800。
    min_free_mib==0 时不检查显存下限（不推荐）。
    """
    if states is None:
        return smi_failed_assume_idle
    for g, free_mib, util in states:
        if g == gpu_id:
            mem_ok = int(min_free_mib) <= 0 or free_mib >= int(min_free_mib)
            if not mem_ok:
                return False
            relax = int(relax_util_when_free_mib)
            if relax > 0 and free_mib >= relax:
                return True
            return util <= max_util_pct
    return False


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(path: str, *, base_dir: str) -> str:
    """Resolve user/config paths deterministically instead of relying on cwd."""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _list_ply_files(model_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(model_dir)):
        if name.lower().endswith(".ply"):
            files.append(os.path.join(model_dir, name))
    if not files:
        raise FileNotFoundError(f"未在 model_path 找到 .ply: {model_dir}")
    return files


def _sample_from_space(spec: Any, rng: random.Random) -> Any:
    """
    支持 3 种参数空间写法：
    1) list: 从列表均匀采样一个
    2) {"type": "uniform", "min": a, "max": b}: 连续均匀
    3) {"type": "log_uniform", "min": a, "max": b}: 对数均匀（a,b>0）
    """
    if isinstance(spec, list):
        if not spec:
            raise ValueError("空间列表为空")
        return rng.choice(spec)
    if isinstance(spec, dict):
        t = spec.get("type", "uniform")
        if t == "uniform":
            a = float(spec["min"])
            b = float(spec["max"])
            return a + (b - a) * rng.random()
        if t == "log_uniform":
            import math

            a = float(spec["min"])
            b = float(spec["max"])
            if a <= 0 or b <= 0:
                raise ValueError("log_uniform 需要 min/max > 0")
            la = math.log(a)
            lb = math.log(b)
            return float(math.exp(la + (lb - la) * rng.random()))
    # 直接常量
    return spec


def _grid_values_from_space(spec: Any, num: int) -> List[Any]:
    """
    将参数空间转换成“遍历用”的取值列表：
    - list: 直接返回该列表（保持顺序）
    - {"type":"uniform","min":a,"max":b}: 等间距 num 点（含端点，num=1 时取中点）
    - {"type":"log_uniform","min":a,"max":b}: 对数等间距 num 点（含端点，num=1 时取几何均值）
    - 常量: 返回 [常量]
    """
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, dict):
        t = spec.get("type", "uniform")
        if t == "uniform":
            a = float(spec["min"])
            b = float(spec["max"])
            if num <= 1:
                return [0.5 * (a + b)]
            step = (b - a) / float(num - 1)
            return [a + i * step for i in range(num)]
        if t == "log_uniform":
            import math

            a = float(spec["min"])
            b = float(spec["max"])
            if a <= 0 or b <= 0:
                raise ValueError("log_uniform 需要 min/max > 0")
            if num <= 1:
                return [float(math.sqrt(a * b))]
            la = math.log(a)
            lb = math.log(b)
            step = (lb - la) / float(num - 1)
            return [float(math.exp(la + i * step)) for i in range(num)]
    return [spec]


def _filter_material_params_for_type(material_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    只保留对该 material 有意义/被 solver 接受的参数。
    solver 接受的键来自 mpm_solver_warp.MPM_Simulator_WARP.set_parameters_dict
    """
    allowed_common = {
        "material",
        "grid_lim",
        "n_grid",
        "E",
        "nu",
        "density",
        "g",
        "rpic_damping",
        "plastic_viscosity",
        "softening",
        "grid_v_damping_scale",
        "hardening",
        "xi",
        "friction_angle",
        "yield_stress",
    }
    out = {k: v for k, v in params.items() if k in allowed_common}

    # material 专属参数约束（经验规则，避免无意义键）
    if material_type != "metal":
        out.pop("yield_stress", None)
        out.pop("hardening", None)
    if material_type != "sand":
        out.pop("friction_angle", None)
    return out


def _safe_model_dir_name(ply_stem: str) -> str:
    """一级输出目录名：与 PLY 物体名一致，去掉路径不安全字符。"""
    s = ply_stem.replace("/", "_").replace("\\", "_").strip()
    if not s or s in (".", ".."):
        return "unknown_object"
    return s


def _safe_dataset_segment(name: str, kind: str = "dataset") -> str:
    """数据集名或 split 子目录（如 train），禁止路径穿越。"""
    s = str(name).replace("/", "_").replace("\\", "_").strip()
    if not s or s in (".", ".."):
        raise ValueError(f"非法{kind}目录名: {name!r}（需非空且非 . / ..）")
    return s


@dataclass(frozen=True)
class Job:
    ply_path: str
    sim_type: str
    material_params: Dict[str, Any]
    output_root: str
    output_layout: str  # by_model | by_action
    num_views: int
    num_render_views: int  # -1 表示与 num_views 相同
    random_render_views: bool
    random_render_views_min_gap_deg: float
    random_render_views_seed: int
    num_render_timesteps: int  # 0 表示每仿真帧都输出渲染/视角辅助
    render_outputs_per_sim_second: float  # >0 时覆盖为「每秒仿真时长 N 张」均匀采样 + 视频帧率
    render_img: bool
    compile_video: bool
    delete_png_sequences_after_compile_video: bool
    output_deformation: bool
    output_stress: bool
    output_view_stress_heatmap: bool
    output_view_stress_gaussian: bool
    stress_gaussian_colormap_steps: int
    stress_gaussian_vm_pct_low: float
    stress_gaussian_vm_pct_high: float
    stress_gaussian_single_channel: bool
    output_view_flow_gaussian: bool
    output_view_object_mask: bool
    output_view_force_mask: bool
    force_mask_single_channel: bool
    flow_gaussian_max_gaussians: int
    flow_gaussian_seed: int
    flow_gaussian_depth_gamma: float
    flow_gaussian_depth_eps: float
    flow_gaussian_opacity_power: float
    flow_gaussian_vis_max_motion: float
    output_view_tracks_gaussian: bool
    tracks_gaussian_max_tracks: int
    tracks_gaussian_sigma_scale: float
    tracks_gaussian_point_opacity: float
    tracks_gaussian_intensity: float
    tracks_gaussian_seed: int
    tracks_gaussian_accum_mode: str
    tracks_gaussian_accum_frame_weight: float
    tracks_gaussian_accum_decay: float
    tracks_gaussian_accum_midpoint: bool
    tracks_gaussian_accum_no_normalize_save: bool
    output_subsampled_world_tracks: bool
    subsampled_tracks_num: int
    subsampled_tracks_seed: int
    subsampled_tracks_ortho_axes: str
    subsampled_tracks_video_size: int
    no_volumetric_stress_deformation: bool
    field_output_interval: int
    output_bc_info: bool
    output_force_info: bool
    output_initial_force_mask_arrow: bool
    pack_arch4_tensors: bool
    pack_arch4_lmdb: bool
    pack_arch4_lmdb_include_object_mask: bool
    pack_sample_pack: bool
    pack_sample_pack_include_object_mask: bool
    arch4_lmdb_resize: int
    arch4_lmdb_map_size_gb: float
    arch4_lmdb_name: str
    sample_pack_name: str
    compress_render_pngs: bool
    png_compression_level: int
    render_export_max_side: int
    render_export_scale: float
    camera_distance_scale: float
    arch4_tensor_dtype: str
    white_bg: bool
    debug: bool
    # 非 None 时：样本写到 output_root/dataset_name/[dataset_split]/000000/（6 位），并写 gt.json
    dataset_name: Optional[str]
    dataset_split: str  # 空字符串表示不再套一层子目录（仅 output_root/dataset_name/000000/）


def _build_job_list(
    ply_files: List[str],
    material_space: Dict[str, Any],
    train_cfg: Dict[str, Any],
    rng: random.Random,
) -> List[Job]:
    sim_types = train_cfg.get("sim_types", ["bend", "drop", "press", "shear", "stretch"])
    n_total = int(train_cfg.get("num_simulations", 10))

    # 输出根目录默认固定为 PhysGaussian/auto_output（相对本脚本目录）
    output_root = train_cfg.get("output_root", os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_output"))
    # by_model：auto_output/<物体名>/<obj__params__action>/... ；by_action：auto_output/<action>/...
    _ol = str(train_cfg.get("output_layout", "by_model")).strip().lower()
    output_layout = _ol if _ol in ("by_model", "by_action") else "by_model"
    _dn_raw = train_cfg.get("dataset_name", None)
    if _dn_raw is None:
        dataset_name: Optional[str] = None
    else:
        _dn = str(_dn_raw).strip()
        dataset_name = _dn if _dn else None
    _ds_raw = train_cfg.get("dataset_split", "train")
    dataset_split = str(_ds_raw).strip() if _ds_raw is not None else ""
    num_views = int(train_cfg.get("num_views", 1))
    # 渲染视角数：默认同 num_views；可单独设 num_render_views（与 rules/dataset_task 对齐）
    nr_v = train_cfg.get("num_render_views", None)
    num_render_views = int(nr_v) if nr_v is not None else -1
    random_render_views = bool(train_cfg.get("random_render_views", False))
    random_render_views_min_gap_deg = float(
        train_cfg.get("random_render_views_min_gap_deg", 20.0)
    )
    random_render_views_seed = int(train_cfg.get("random_render_views_seed", 0))
    num_render_timesteps = int(train_cfg.get("num_render_timesteps", 0))
    render_outputs_per_sim_second = float(
        train_cfg.get("render_outputs_per_sim_second", 0.0) or 0.0
    )
    render_img = bool(train_cfg.get("render_img", True))
    # 统一控制：有 PNG 后是否用 ffmpeg 合成 mp4（images/、stress_gaussian/、tracks_gaussian/、flow_gaussian/；另含 subsampled 正交轨迹 mp4）
    if "compile_multiview_videos" in train_cfg:
        compile_video = bool(train_cfg["compile_multiview_videos"])
    else:
        compile_video = bool(train_cfg.get("compile_video", True))
    delete_png_sequences_after_compile_video = bool(
        train_cfg.get("delete_png_sequences_after_compile_video", False)
    )
    output_deformation = bool(train_cfg.get("output_deformation", False))
    output_stress = bool(train_cfg.get("output_stress", False))
    output_view_stress_heatmap = bool(train_cfg.get("output_view_stress_heatmap", False))
    output_view_stress_gaussian = bool(
        train_cfg.get("output_view_stress_gaussian", False)
    )
    stress_gaussian_colormap_steps = int(
        train_cfg.get("stress_gaussian_colormap_steps", 24)
    )
    stress_gaussian_vm_pct_low = float(
        train_cfg.get(
            "stress_gaussian_vm_pct_low",
            train_cfg.get("stress_gaussian_log_pct_low", 1.0),
        )
    )
    stress_gaussian_vm_pct_high = float(
        train_cfg.get(
            "stress_gaussian_vm_pct_high",
            train_cfg.get("stress_gaussian_log_pct_high", 99.0),
        )
    )
    stress_gaussian_single_channel = bool(
        train_cfg.get("stress_gaussian_single_channel", False)
    )
    output_view_flow_gaussian = bool(
        train_cfg.get("output_view_flow_gaussian", False)
    )
    output_view_object_mask = bool(train_cfg.get("output_view_object_mask", False))
    output_view_force_mask = bool(train_cfg.get("output_view_force_mask", False))
    force_mask_single_channel = bool(train_cfg.get("force_mask_single_channel", False))
    flow_gaussian_max_gaussians = int(
        train_cfg.get("flow_gaussian_max_gaussians", 8192)
    )
    flow_gaussian_seed = int(train_cfg.get("flow_gaussian_seed", 0))
    flow_gaussian_depth_gamma = float(
        train_cfg.get("flow_gaussian_depth_gamma", 1.0)
    )
    flow_gaussian_depth_eps = float(
        train_cfg.get("flow_gaussian_depth_eps", 1e-2)
    )
    flow_gaussian_opacity_power = float(
        train_cfg.get("flow_gaussian_opacity_power", 1.0)
    )
    flow_gaussian_vis_max_motion = float(
        train_cfg.get("flow_gaussian_vis_max_motion", 0.0)
    )
    output_view_tracks_gaussian = bool(
        train_cfg.get("output_view_tracks_gaussian", False)
    )
    tracks_gaussian_max_tracks = int(train_cfg.get("tracks_gaussian_max_tracks", 2048))
    tracks_gaussian_sigma_scale = float(
        train_cfg.get("tracks_gaussian_sigma_scale", 0.0012)
    )
    tracks_gaussian_point_opacity = float(
        train_cfg.get("tracks_gaussian_point_opacity", 1.0)
    )
    tracks_gaussian_point_opacity = max(
        0.01, min(1.0, tracks_gaussian_point_opacity)
    )
    tracks_gaussian_intensity = float(
        train_cfg.get("tracks_gaussian_intensity", 1.0)
    )
    tracks_gaussian_seed = int(train_cfg.get("tracks_gaussian_seed", 0))
    _tgm = str(train_cfg.get("tracks_gaussian_accum_mode", "max")).strip().lower()
    tracks_gaussian_accum_mode = _tgm if _tgm in ("max", "add") else "max"
    tracks_gaussian_accum_frame_weight = float(
        train_cfg.get("tracks_gaussian_accum_frame_weight", 1.0)
    )
    tracks_gaussian_accum_decay = float(
        train_cfg.get("tracks_gaussian_accum_decay", 1.0)
    )
    tracks_gaussian_accum_midpoint = bool(
        train_cfg.get("tracks_gaussian_accum_midpoint", False)
    )
    tracks_gaussian_accum_no_normalize_save = bool(
        train_cfg.get("tracks_gaussian_accum_no_normalize_save", False)
    )
    output_subsampled_world_tracks = bool(
        train_cfg.get("output_subsampled_world_tracks", False)
    )
    subsampled_tracks_num = int(train_cfg.get("subsampled_tracks_num", 1024))
    subsampled_tracks_seed = int(train_cfg.get("subsampled_tracks_seed", 0))
    _sa = str(train_cfg.get("subsampled_tracks_ortho_axes", "xz")).strip().lower()
    subsampled_tracks_ortho_axes = _sa if _sa in ("xy", "xz", "yz") else "xz"
    subsampled_tracks_video_size = max(
        32, int(train_cfg.get("subsampled_tracks_video_size", 512))
    )
    no_volumetric_stress_deformation = bool(
        train_cfg.get("no_volumetric_stress_deformation", False)
    )
    field_output_interval = int(train_cfg.get("field_output_interval", 1))
    output_bc_info = bool(train_cfg.get("output_bc_info", True))
    output_force_info = bool(train_cfg.get("output_force_info", True))
    output_initial_force_mask_arrow = bool(
        train_cfg.get("output_initial_force_mask_arrow", False)
    )
    pack_arch4_lmdb = bool(train_cfg.get("pack_arch4_lmdb", False))
    pack_arch4_lmdb_include_object_mask = bool(
        train_cfg.get("pack_arch4_lmdb_include_object_mask", True)
    )
    pack_sample_pack = bool(train_cfg.get("pack_sample_pack", False))
    pack_sample_pack_include_object_mask = bool(
        train_cfg.get("pack_sample_pack_include_object_mask", True)
    )
    pack_arch4_tensors = bool(train_cfg.get("pack_arch4_tensors", True)) and (
        not pack_arch4_lmdb
    )
    compress_render_pngs = bool(train_cfg.get("compress_render_pngs", True))
    png_compression_level = max(
        0, min(9, int(train_cfg.get("png_compression_level", 6)))
    )
    render_export_max_side = max(0, int(train_cfg.get("render_export_max_side", 0) or 0))
    _res_scale = float(train_cfg.get("render_export_scale", 1.0) or 1.0)
    render_export_scale = _res_scale if _res_scale > 0.0 else 1.0
    _cam_dist_scale = float(train_cfg.get("camera_distance_scale", 1.0) or 1.0)
    camera_distance_scale = _cam_dist_scale if _cam_dist_scale > 0.0 else 1.0
    _adtp = str(train_cfg.get("arch4_tensor_dtype", "float32")).strip().lower()
    if _adtp in ("fp16", "half"):
        _adtp = "float16"
    if _adtp in ("bf16",):
        _adtp = "bfloat16"
    if _adtp not in ("float32", "float16", "bfloat16"):
        _adtp = "float32"
    arch4_tensor_dtype = _adtp
    arch4_lmdb_resize = max(8, int(train_cfg.get("arch4_lmdb_resize", 224)))
    arch4_lmdb_map_size_gb = max(0.25, float(train_cfg.get("arch4_lmdb_map_size_gb", 8.0)))
    _lmn = str(train_cfg.get("arch4_lmdb_name", "arch4_data.lmdb")).strip() or "arch4_data.lmdb"
    arch4_lmdb_name = _lmn.replace("/", "_").replace("\\", "_").strip()
    if not arch4_lmdb_name or arch4_lmdb_name in (".", ".."):
        arch4_lmdb_name = "arch4_data.lmdb"
    _spn = str(train_cfg.get("sample_pack_name", "sample_pack.npz")).strip() or "sample_pack.npz"
    sample_pack_name = _spn.replace("/", "_").replace("\\", "_").strip()
    if not sample_pack_name or sample_pack_name in (".", ".."):
        sample_pack_name = "sample_pack.npz"
    white_bg = bool(train_cfg.get("white_bg", False))
    debug = bool(train_cfg.get("debug", False))

    jobs: List[Job] = []

    mat_spaces = material_space.get("material_spaces", {})
    if not mat_spaces:
        raise ValueError("material_space_config 需要提供 material_spaces，按材质拆分参数空间。")

    material_values = sorted(mat_spaces.keys())

    def _make_job(ply_path: str, sim_type: str, material_params: Dict[str, Any]) -> Job:
        return Job(
            ply_path=ply_path,
            sim_type=sim_type,
            material_params=material_params,
            output_root=output_root,
            output_layout=output_layout,
            num_views=num_views,
            num_render_views=num_render_views,
            random_render_views=random_render_views,
            random_render_views_min_gap_deg=random_render_views_min_gap_deg,
            random_render_views_seed=random_render_views_seed,
            num_render_timesteps=num_render_timesteps,
            render_outputs_per_sim_second=render_outputs_per_sim_second,
            render_img=render_img,
            compile_video=compile_video,
            delete_png_sequences_after_compile_video=delete_png_sequences_after_compile_video,
            output_deformation=output_deformation,
            output_stress=output_stress,
            output_view_stress_heatmap=output_view_stress_heatmap,
            output_view_stress_gaussian=output_view_stress_gaussian,
            stress_gaussian_colormap_steps=stress_gaussian_colormap_steps,
            stress_gaussian_vm_pct_low=stress_gaussian_vm_pct_low,
            stress_gaussian_vm_pct_high=stress_gaussian_vm_pct_high,
            stress_gaussian_single_channel=stress_gaussian_single_channel,
            output_view_flow_gaussian=output_view_flow_gaussian,
            output_view_object_mask=output_view_object_mask,
            output_view_force_mask=output_view_force_mask,
            force_mask_single_channel=force_mask_single_channel,
            flow_gaussian_max_gaussians=flow_gaussian_max_gaussians,
            flow_gaussian_seed=flow_gaussian_seed,
            flow_gaussian_depth_gamma=flow_gaussian_depth_gamma,
            flow_gaussian_depth_eps=flow_gaussian_depth_eps,
            flow_gaussian_opacity_power=flow_gaussian_opacity_power,
            flow_gaussian_vis_max_motion=flow_gaussian_vis_max_motion,
            output_view_tracks_gaussian=output_view_tracks_gaussian,
            tracks_gaussian_max_tracks=tracks_gaussian_max_tracks,
            tracks_gaussian_sigma_scale=tracks_gaussian_sigma_scale,
            tracks_gaussian_point_opacity=tracks_gaussian_point_opacity,
            tracks_gaussian_intensity=tracks_gaussian_intensity,
            tracks_gaussian_seed=tracks_gaussian_seed,
            tracks_gaussian_accum_mode=tracks_gaussian_accum_mode,
            tracks_gaussian_accum_frame_weight=tracks_gaussian_accum_frame_weight,
            tracks_gaussian_accum_decay=tracks_gaussian_accum_decay,
            tracks_gaussian_accum_midpoint=tracks_gaussian_accum_midpoint,
            tracks_gaussian_accum_no_normalize_save=tracks_gaussian_accum_no_normalize_save,
            output_subsampled_world_tracks=output_subsampled_world_tracks,
            subsampled_tracks_num=subsampled_tracks_num,
            subsampled_tracks_seed=subsampled_tracks_seed,
            subsampled_tracks_ortho_axes=subsampled_tracks_ortho_axes,
            subsampled_tracks_video_size=subsampled_tracks_video_size,
            no_volumetric_stress_deformation=no_volumetric_stress_deformation,
            field_output_interval=field_output_interval,
            output_bc_info=output_bc_info,
            output_force_info=output_force_info,
            output_initial_force_mask_arrow=output_initial_force_mask_arrow,
            pack_arch4_tensors=pack_arch4_tensors,
            pack_arch4_lmdb=pack_arch4_lmdb,
            pack_arch4_lmdb_include_object_mask=pack_arch4_lmdb_include_object_mask,
            pack_sample_pack=pack_sample_pack,
            pack_sample_pack_include_object_mask=pack_sample_pack_include_object_mask,
            arch4_lmdb_resize=arch4_lmdb_resize,
            arch4_lmdb_map_size_gb=arch4_lmdb_map_size_gb,
            arch4_lmdb_name=arch4_lmdb_name,
            sample_pack_name=sample_pack_name,
            compress_render_pngs=compress_render_pngs,
            png_compression_level=png_compression_level,
            render_export_max_side=render_export_max_side,
            render_export_scale=render_export_scale,
            camera_distance_scale=camera_distance_scale,
            arch4_tensor_dtype=arch4_tensor_dtype,
            white_bg=white_bg,
            debug=debug,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )

    boundary_test_mode = bool(train_cfg.get("boundary_test_mode", False))
    if boundary_test_mode:
        fixed_raw = train_cfg.get("boundary_fixed_material")
        if not isinstance(fixed_raw, dict):
            raise ValueError(
                "boundary_test_mode=true 时 train_config.boundary_fixed_material 必须为材质参数字典（含 material 与各物理量）"
            )
        mt = fixed_raw.get("material")
        if mt is None or str(mt) not in mat_spaces:
            raise ValueError(
                "boundary_fixed_material.material 必须存在于 material_space 的 material_spaces 中，"
                f"得到: {mt!r}"
            )
        mt = str(mt)
        sampled_base = _filter_material_params_for_type(mt, dict(fixed_raw))
        for ply_path in sorted(ply_files):
            for sim_type in sim_types:
                jobs.append(_make_job(ply_path, sim_type, dict(sampled_base)))
        return jobs

    # 完全随机采样：每次仿真独立随机选 model / sim_type / 材质类型 / 该材质的参数组合
    for _ in range(n_total):
        ply_path = rng.choice(ply_files)
        sim_type = rng.choice(sim_types)

        material_type = rng.choice(material_values)
        space = mat_spaces[material_type]

        sampled: Dict[str, Any] = {"material": material_type}
        for k, spec in space.items():
            sampled[k] = _sample_from_space(spec, rng)

        # 仍然跑一遍过滤器，防止 future 扩展时混入不支持的键
        sampled = _filter_material_params_for_type(material_type, sampled)

        jobs.append(_make_job(ply_path, sim_type, sampled))

    return jobs


def _make_run_config(
    base_cfg: Dict[str, Any],
    sim_type: str,
    material_params: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    # 直接覆盖 material 参数
    for k, v in material_params.items():
        # g 只对 drop 生效：非 drop 时不覆盖模板里的重力设置
        if k == "g" and sim_type != "drop":
            continue
        cfg[k] = v
    # boundary_conditions 会在 modified_simulation.py 内按 sim_type 自动生成
    cfg["boundary_conditions"] = cfg.get("boundary_conditions", [])
    return cfg


def _job_output_dir(job: Job, sample_index: int) -> str:
    """
    命名数据集（job.dataset_name 非空）：
      <output_root>/<dataset_name>/[<dataset_split>/]<000000>/  （6 位全局序号，与 dataset_400 一致）
    否则（每种 model 四位序号）：
    - by_model:   <output_root>/<model_safe>/<NNNN>/
    - by_action:  <output_root>/<sim_type>/<model_safe>/<NNNN>/
    """
    if getattr(job, "dataset_name", None):
        ds_name = _safe_dataset_segment(job.dataset_name, kind="数据集")
        parts = [job.output_root, ds_name]
        split = (getattr(job, "dataset_split", "") or "").strip()
        if split:
            parts.append(_safe_dataset_segment(split, kind="split"))
        parts.append(f"{int(sample_index):06d}")
        return os.path.join(*parts)

    ply_stem = os.path.splitext(os.path.basename(job.ply_path))[0]
    model_safe = _safe_model_dir_name(ply_stem)
    sub = f"{int(sample_index):04d}"
    if getattr(job, "output_layout", "by_model") == "by_action":
        return os.path.join(job.output_root, job.sim_type, model_safe, sub)
    return os.path.join(job.output_root, model_safe, sub)


def _boundary_merge_staging_dir(train_cfg: Dict[str, Any], script_dir: str) -> str:
    """边界测试：按 action 合并后的 mp4 输出目录。"""
    output_root = train_cfg.get("output_root", os.path.join(script_dir, "auto_output"))
    sub = str(train_cfg.get("boundary_merge_subdir", "boundary_merged_by_action")).strip()
    if not sub or sub in (".", ".."):
        sub = "boundary_merged_by_action"
    dn_raw = train_cfg.get("dataset_name", None)
    if dn_raw is not None:
        _dn = str(dn_raw).strip()
        if _dn:
            ds_split_raw = train_cfg.get("dataset_split", "train")
            ds_split = str(ds_split_raw).strip() if ds_split_raw is not None else ""
            parts = [output_root, _safe_dataset_segment(_dn, kind="数据集")]
            if ds_split:
                parts.append(_safe_dataset_segment(ds_split, kind="split"))
            parts.append(sub)
            return os.path.join(*parts)
    return os.path.join(output_root, sub)


def _boundary_dataset_root(train_cfg: Dict[str, Any], script_dir: str) -> Optional[str]:
    """边界测试数据集根目录（样本目录的父目录）。"""
    output_root = train_cfg.get("output_root", os.path.join(script_dir, "auto_output"))
    dn_raw = train_cfg.get("dataset_name", None)
    if dn_raw is None:
        return None
    dn = str(dn_raw).strip()
    if not dn:
        return None
    ds_split_raw = train_cfg.get("dataset_split", "train")
    ds_split = str(ds_split_raw).strip() if ds_split_raw is not None else ""
    parts = [output_root, _safe_dataset_segment(dn, kind="数据集")]
    if ds_split:
        parts.append(_safe_dataset_segment(ds_split, kind="split"))
    return os.path.join(*parts)


def _ffmpeg_concat_mp4s(clip_paths: List[str], out_mp4: str, ffmpeg_bin: str) -> bool:
    """用 concat demuxer 串联多个 mp4；stream copy 失败时退回 libx264 重编码。"""
    if not clip_paths:
        return False
    list_path: Optional[str] = None
    try:
        fd, list_path = tempfile.mkstemp(suffix=".txt", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for p in clip_paths:
                ap = os.path.abspath(p).replace("'", "'\\''")
                f.write(f"file '{ap}'\n")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            out_mp4,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return True
        cmd2 = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            out_mp4,
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
        return proc2.returncode == 0
    finally:
        if list_path and os.path.isfile(list_path):
            try:
                os.unlink(list_path)
            except OSError:
                pass


def _boundary_collect_existing_successes(
    train_cfg: Dict[str, Any],
    *,
    script_dir: str,
) -> List[Tuple[int, str]]:
    """
    从现有输出目录扫描成功样本，返回 [(sample_idx, out_dir), ...]。
    成功样本定义为：存在 gt.json 且 videos/ 下至少有一个 mp4。
    """
    dataset_root = _boundary_dataset_root(train_cfg, script_dir)
    if not dataset_root or not os.path.isdir(dataset_root):
        return []

    out: List[Tuple[int, str]] = []
    for name in sorted(os.listdir(dataset_root)):
        sample_dir = os.path.join(dataset_root, name)
        if not os.path.isdir(sample_dir):
            continue
        if not str(name).isdigit():
            continue
        gt_json = os.path.join(sample_dir, "gt.json")
        videos_dir = os.path.join(sample_dir, "videos")
        if not os.path.isfile(gt_json) or not os.path.isdir(videos_dir):
            continue
        has_mp4 = any(f.lower().endswith(".mp4") for f in os.listdir(videos_dir))
        if not has_mp4:
            continue
        out.append((int(name), sample_dir))
    return out


def _boundary_merge_output_dirs_by_action(
    action_dirs: Dict[str, List[Tuple[int, str]]],
    train_cfg: Dict[str, Any],
    *,
    script_dir: str,
    quiet: bool,
) -> None:
    if not bool(train_cfg.get("boundary_merge_videos_by_action", True)):
        return
    if not action_dirs:
        return
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        if not quiet:
            print(
                "[auto_simulation_runner] boundary_merge_videos_by_action: "
                "未找到 ffmpeg，跳过按 action 合并。"
            )
        return

    merge_root = _boundary_merge_staging_dir(train_cfg, script_dir)
    os.makedirs(merge_root, exist_ok=True)

    for sim_type in sorted(action_dirs.keys()):
        ordered = sorted(action_dirs[sim_type], key=lambda x: x[0])
        out_dirs = [od for _i, od in ordered]
        if not out_dirs:
            continue

        first_videos = ""
        for od in out_dirs:
            cand = os.path.join(od, "videos")
            if os.path.isdir(cand):
                first_videos = cand
                break
        if not first_videos:
            continue

        mp4_names = sorted(
            f for f in os.listdir(first_videos) if f.lower().endswith(".mp4")
        )
        if not mp4_names:
            continue
        clips: List[str] = []
        for od in out_dirs:
            vdir = os.path.join(od, "videos")
            if not os.path.isdir(vdir):
                continue
            for name in mp4_names:
                p = os.path.join(vdir, name)
                if os.path.isfile(p):
                    clips.append(p)
                elif not quiet:
                    tqdm.write(
                        f"[auto_simulation_runner] boundary_merge: 缺少片段，已跳过: {p}"
                    )
        if not clips:
            continue
        out_merged = os.path.join(merge_root, f"{sim_type}_all_samples.mp4")
        ok = _ffmpeg_concat_mp4s(clips, out_merged, ffmpeg_bin)
        if ok and not quiet:
            print(
                f"[auto_simulation_runner] boundary_merge: 已写出 {out_merged} "
                f"（{len(clips)} 段 clip，action={sim_type}）"
            )
        elif not ok and not quiet:
            print(
                f"[auto_simulation_runner] boundary_merge: 合并失败 action={sim_type} -> {out_merged}",
                file=sys.stderr,
            )


def _boundary_merge_videos_by_action(
    tasks: List[Tuple[int, Job, List[str], str]],
    train_cfg: Dict[str, Any],
    *,
    script_dir: str,
    quiet: bool,
) -> None:
    """
    边界测试模式：对每个 sim_type，将各样本目录下 videos/ 中同名的 mp4 按 job 顺序串联，
    生成一个长视频（多视角时按视角名排序后，逐物体依次拼接：obj1 的 view_a, view_b, …，再 obj2 …）。
    """
    if not bool(train_cfg.get("boundary_merge_videos_by_action", True)):
        return
    if not tasks:
        return
    if not tasks[0][1].compile_video:
        if not quiet:
            print(
                "[auto_simulation_runner] boundary_merge_videos_by_action: "
                "compile_video 关闭，跳过。"
            )
        return

    by_action: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for idx, job, _cmd, out_dir in tasks:
        by_action[job.sim_type].append((idx, out_dir))
    _boundary_merge_output_dirs_by_action(
        by_action,
        train_cfg,
        script_dir=script_dir,
        quiet=quiet,
    )


def _boundary_merge_existing_outputs_by_action(
    train_cfg: Dict[str, Any],
    *,
    script_dir: str,
    quiet: bool,
) -> None:
    """离线扫描当前 boundary_test 输出，对已有成功样本按 action 合并视频。"""
    existing = _boundary_collect_existing_successes(train_cfg, script_dir=script_dir)
    by_action: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for idx, out_dir in existing:
        gt_json = os.path.join(out_dir, "gt.json")
        try:
            with open(gt_json, "r", encoding="utf-8") as f:
                gt = json.load(f)
        except Exception as e:
            if not quiet:
                tqdm.write(
                    f"[auto_simulation_runner] boundary_merge_existing: 读取失败，已跳过 {gt_json}: {e}"
                )
            continue
        action = str(gt.get("action", "")).strip()
        if not action:
            continue
        by_action[action].append((idx, out_dir))
    _boundary_merge_output_dirs_by_action(
        by_action,
        train_cfg,
        script_dir=script_dir,
        quiet=quiet,
    )


def _write_gt_parameters_json(
    out_dir: str,
    job: Job,
    job_index: int,
    *,
    sample_index_for_output: int,
) -> None:
    """地面真值物理参数（与目录名解耦，便于下游读取）。"""
    ply_stem = os.path.splitext(os.path.basename(job.ply_path))[0]
    payload = {
        "ply_stem": ply_stem,
        "ply_path": job.ply_path,
        "sim_type": job.sim_type,
        "material_params": dict(job.material_params),
        "job_index": int(job_index),
        "output_layout": job.output_layout,
        "sample_index": int(sample_index_for_output),
        "folder_digits": 6 if getattr(job, "dataset_name", None) else 4,
    }
    if getattr(job, "dataset_name", None):
        payload["dataset_name"] = job.dataset_name
        payload["dataset_split"] = (job.dataset_split or "").strip()
    _write_json(os.path.join(out_dir, "gt_parameters.json"), payload)


def _material_params_to_str_dict(mp: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in mp.items()}


def _build_gt_json_inline(
    object_name: str,
    action: str,
    params: Dict[str, str],
    *,
    sim_types_ref: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    内置版 gt.json 构造逻辑（避免依赖 transform_dataset.py）。

    字段尽量对齐历史 dataset_deformation_stress_500 的格式：
    - object / action / action_index
    - params（字符串）
    - regression（E/nu/density/yield_stress 的 float）
    """
    # action_index：优先按当前配置里的 sim_types 顺序；否则回退默认顺序
    sim_types = list(sim_types_ref or ["bend", "drop", "press", "shear", "stretch"])
    action_l = str(action).strip().lower()
    if action_l in sim_types:
        action_index = int(sim_types.index(action_l))
    else:
        action_index = -1

    def _to_float_or_none(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    reg = {
        "E": _to_float_or_none(params.get("E", None)),
        "nu": _to_float_or_none(params.get("nu", None)),
        "density": _to_float_or_none(params.get("density", None)),
        "yield_stress": _to_float_or_none(params.get("yield_stress", None)),
    }

    return {
        "object": str(object_name),
        "action": str(action),
        "action_index": int(action_index),
        "params": dict(params),
        "regression": reg,
    }


def _write_gt_json_dataset_layout(
    out_dir: str,
    job: Job,
    job_index: int,
    sample_index: int,
    *,
    train_seed: int,
) -> None:
    """
    与 transform_dataset 产出的 dataset_400/train/000000/gt.json 字段对齐，
    并附加 dataset 元信息，便于后续划分 train/test 或统一训练。
    """
    ply_stem = os.path.splitext(os.path.basename(job.ply_path))[0]
    params_str = _material_params_to_str_dict(job.material_params)
    gt: Dict[str, Any] = _build_gt_json_inline(
        ply_stem,
        job.sim_type,
        params_str,
        sim_types_ref=["bend", "drop", "press", "shear", "stretch"],
    )
    gt["sample_index"] = int(sample_index)
    gt["job_index"] = int(job_index)
    gt["train_seed"] = int(train_seed)
    gt["dataset_name"] = job.dataset_name
    gt["dataset_split"] = (job.dataset_split or "").strip()
    gt["ply_path"] = job.ply_path
    gt["output_layout_config"] = job.output_layout
    _write_json(os.path.join(out_dir, "gt.json"), gt)


def _run_one_job(
    idx: int,
    job: Job,
    cmd: List[str],
    out_dir: str,
    script_dir: str,
    run_env: Dict[str, str],
    gpu_id: int,
) -> Tuple[int, int, str]:
    """在指定 GPU 上跑一个任务，返回 (idx, returncode, out_dir)。"""
    env = dict(run_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    proc = subprocess.run(cmd, cwd=script_dir, env=env)
    return (idx, proc.returncode, out_dir)


def run() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    release_dir = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="包含多个 .ply 的文件夹")
    parser.add_argument("--material_space_config", type=str, required=True, help="材质空间采样配置 JSON")
    parser.add_argument("--train_config", type=str, required=True, help="训练/批量运行配置 JSON")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="默认 1 单卡顺序。≥2 时为「最多同时并行几个仿真任务」；"
        "候选卡为全部可见物理 GPU（或 CUDA_VISIBLE_DEVICES 列表），在整机范围内选最空的卡，"
        "而不是只用编号 0..N-1。0 表示并行数=可见 GPU 数量。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="与 train_config 中 quiet 任一开启：子进程加 --quiet；本进程用 tqdm 显示总进度与 ETA。",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="若指定则覆盖 train_config.dataset_name：样本写入 output_root/<名称>/[split]/000000/ 并写 gt.json。",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="若指定则覆盖 train_config.dataset_split（默认 train；设为空字符串则不要 train|test 中间目录）。",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="只校验输入和配置并打印任务数，不加载 CUDA 或启动仿真。",
    )
    args = parser.parse_args()

    model_dir = _resolve_path(args.model_path, base_dir=release_dir)
    material_space_path = _resolve_path(args.material_space_config, base_dir=release_dir)
    train_config_path = _resolve_path(args.train_config, base_dir=release_dir)
    material_space = _read_json(material_space_path)
    train_cfg = dict(_read_json(train_config_path))
    if args.dataset_name is not None:
        train_cfg["dataset_name"] = args.dataset_name
    if args.dataset_split is not None:
        train_cfg["dataset_split"] = args.dataset_split
    quiet_run = bool(args.quiet) or bool(train_cfg.get("quiet", False))

    seed = int(train_cfg.get("seed", 0))
    rng = random.Random(seed)

    ply_files = _list_ply_files(model_dir)

    base_cfg_by_sim: Dict[str, str] = train_cfg.get("base_config_by_sim_type", {})
    if not base_cfg_by_sim:
        raise ValueError(
            "train_config 需要提供 base_config_by_sim_type，指向各 sim_type 的模板 config json（例如 press/drop/...）。"
        )

    # Scene templates are historically written as config/<name>.json and live
    # next to modified_simulation.py. Absolute paths remain untouched.
    base_cfg_by_sim = {
        str(k): _resolve_path(str(v), base_dir=script_dir)
        for k, v in base_cfg_by_sim.items()
    }
    train_cfg["base_config_by_sim_type"] = base_cfg_by_sim

    output_root = str(train_cfg.get("output_root", "auto_output"))
    train_cfg["output_root"] = _resolve_path(output_root, base_dir=release_dir)

    jobs = _build_job_list(ply_files, material_space, train_cfg, rng)
    if bool(train_cfg.get("boundary_test_mode", False)) and not quiet_run:
        n_ply = len(ply_files)
        n_act = len(train_cfg.get("sim_types", ["bend", "drop", "press", "shear", "stretch"]))
        print(
            f"boundary_test_mode: {len(jobs)} jobs = {n_ply} ply × {n_act} sim_types（固定材质 boundary_fixed_material）"
        )

    modified_sim = os.path.join(script_dir, "modified_simulation.py")

    if args.validate_only:
        missing_scene_configs = [
            path for path in base_cfg_by_sim.values() if not os.path.isfile(path)
        ]
        if missing_scene_configs:
            raise FileNotFoundError(
                "missing base simulation config(s): "
                + ", ".join(missing_scene_configs)
            )
        for path in base_cfg_by_sim.values():
            _read_json(path)
        if not os.path.isfile(modified_sim):
            raise FileNotFoundError(f"simulation entry point not found: {modified_sim}")
        print(
            f"configuration valid: {len(ply_files)} PLY files, "
            f"{len(jobs)} simulation jobs, output_root={train_cfg['output_root']}"
        )
        return

    # 子进程需能加载 torch 与 MACA/cu-bridge 运行库：
    # - torch lib: 保障 simple_knn/diff_gaussian_rasterization 等扩展可链接
    # - maca/cu-bridge lib: 保障 warp 可找到 libcuda.so（MetaX 兼容层）
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    _maca_root = os.environ.get("MACA_PATH")
    _maca_lib_dirs = (
        [
            os.path.join(_maca_root, "tools", "cu-bridge", "lib"),
            os.path.join(_maca_root, "lib"),
        ]
        if _maca_root
        else []
    )
    _run_env = os.environ.copy()
    _ld_parts: List[str] = []
    for p in [_torch_lib] + _maca_lib_dirs + (_run_env.get("LD_LIBRARY_PATH") or "").split(os.pathsep):
        if not p:
            continue
        if p not in _ld_parts:
            _ld_parts.append(p)
    if _ld_parts:
        _run_env["LD_LIBRARY_PATH"] = os.pathsep.join(_ld_parts)

    tmp_cfg_dir = os.path.join(train_cfg.get("output_root", os.path.join(script_dir, "auto_output")), "_tmp_configs")
    os.makedirs(tmp_cfg_dir, exist_ok=True)

    # 确定并行 GPU 数
    num_gpus = args.num_gpus
    if num_gpus == 0:
        num_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
        if not quiet_run:
            print(f"自动检测到 {num_gpus} 张 GPU，将并行运行。")
    if num_gpus < 1:
        num_gpus = 1

    # 每种 model（及 by_action 下每种 model×动作）独立递增四位序号
    per_key_next: Dict[Tuple[str, ...], int] = {}

    def _next_sample_index(job: Job) -> int:
        ply_stem = os.path.splitext(os.path.basename(job.ply_path))[0]
        model_safe = _safe_model_dir_name(ply_stem)
        if job.output_layout == "by_action":
            key = (job.sim_type, model_safe)
        else:
            key = (model_safe,)
        n = per_key_next.get(key, 0)
        per_key_next[key] = n + 1
        return n

    # 为所有 job 生成配置并构建 cmd
    tasks: List[Tuple[int, Job, List[str], str]] = []
    for idx, job in enumerate(jobs):
        if job.sim_type not in base_cfg_by_sim:
            raise ValueError(f"base_config_by_sim_type 缺少 sim_type={job.sim_type}")
        base_cfg = _read_json(base_cfg_by_sim[job.sim_type])
        run_cfg = _make_run_config(base_cfg, job.sim_type, job.material_params)

        if job.dataset_name:
            path_idx = idx
        else:
            path_idx = _next_sample_index(job)
        out_dir = _job_output_dir(job, path_idx)
        os.makedirs(out_dir, exist_ok=True)
        # 命名数据集布局下统一使用 gt.json，避免与 gt_parameters.json 重复。
        if not job.dataset_name:
            _write_gt_parameters_json(
                out_dir, job, idx, sample_index_for_output=path_idx
            )
        if job.dataset_name:
            _write_gt_json_dataset_layout(
                out_dir,
                job,
                idx,
                path_idx,
                train_seed=seed,
            )

        cfg_path = os.path.join(tmp_cfg_dir, f"{idx:06d}_{job.sim_type}.json")
        _write_json(cfg_path, run_cfg)

        cmd = [
            sys.executable,
            modified_sim,
            "--ply_path",
            job.ply_path,
            "--config",
            cfg_path,
            "--output_path",
            out_dir,
            "--ply_flat_output",
            "--sim_type",
            job.sim_type,
            "--num_views",
            str(job.num_views),
            "--num_render_views",
            str(job.num_render_views),
            "--random_render_views_min_gap_deg",
            str(job.random_render_views_min_gap_deg),
            "--random_render_views_seed",
            str(job.random_render_views_seed),
            "--num_render_timesteps",
            str(job.num_render_timesteps),
        ]
        if job.random_render_views:
            cmd.append("--random_render_views")
        if float(job.render_outputs_per_sim_second) > 0.0:
            cmd.extend(
                [
                    "--render_outputs_per_sim_second",
                    str(float(job.render_outputs_per_sim_second)),
                ]
            )
        cmd.extend(
            [
                "--field_output_interval",
                str(job.field_output_interval),
            ]
        )
        if job.render_img:
            cmd.append("--render_img")
        if job.compile_video:
            cmd.append("--compile_video")
        if job.delete_png_sequences_after_compile_video:
            cmd.append("--delete_png_sequences_after_compile_video")
        if job.output_deformation:
            cmd.append("--output_deformation")
        if job.output_stress:
            cmd.append("--output_stress")
        if job.output_view_stress_heatmap:
            cmd.append("--output_view_stress_heatmap")
        if job.output_view_stress_gaussian:
            cmd.append("--output_view_stress_gaussian")
            cmd.extend(
                [
                    "--stress_gaussian_colormap_steps",
                    str(job.stress_gaussian_colormap_steps),
                    "--stress_gaussian_vm_pct_low",
                    str(job.stress_gaussian_vm_pct_low),
                    "--stress_gaussian_vm_pct_high",
                    str(job.stress_gaussian_vm_pct_high),
                ]
            )
            if job.stress_gaussian_single_channel:
                cmd.append("--stress_gaussian_single_channel")
        if job.output_view_flow_gaussian:
            cmd.append("--output_view_flow_gaussian")
            cmd.extend(
                [
                    "--flow_gaussian_max_gaussians",
                    str(job.flow_gaussian_max_gaussians),
                    "--flow_gaussian_seed",
                    str(job.flow_gaussian_seed),
                    "--flow_gaussian_depth_gamma",
                    str(job.flow_gaussian_depth_gamma),
                    "--flow_gaussian_depth_eps",
                    str(job.flow_gaussian_depth_eps),
                    "--flow_gaussian_opacity_power",
                    str(job.flow_gaussian_opacity_power),
                    "--flow_gaussian_vis_max_motion",
                    str(job.flow_gaussian_vis_max_motion),
                ]
            )
        if job.output_view_object_mask:
            cmd.append("--output_view_object_mask")
        if job.output_view_force_mask:
            cmd.append("--output_view_force_mask")
            if job.force_mask_single_channel:
                cmd.append("--force_mask_single_channel")
        if job.output_view_tracks_gaussian:
            cmd.append("--output_view_tracks_gaussian")
            cmd.extend(
                [
                    "--tracks_gaussian_max_tracks",
                    str(job.tracks_gaussian_max_tracks),
                    "--tracks_gaussian_sigma_scale",
                    str(job.tracks_gaussian_sigma_scale),
                    "--tracks_gaussian_point_opacity",
                    str(job.tracks_gaussian_point_opacity),
                    "--tracks_gaussian_intensity",
                    str(job.tracks_gaussian_intensity),
                    "--tracks_gaussian_seed",
                    str(job.tracks_gaussian_seed),
                    "--tracks_gaussian_accum_mode",
                    str(job.tracks_gaussian_accum_mode),
                    "--tracks_gaussian_accum_frame_weight",
                    str(job.tracks_gaussian_accum_frame_weight),
                    "--tracks_gaussian_accum_decay",
                    str(job.tracks_gaussian_accum_decay),
                ]
            )
            if job.tracks_gaussian_accum_midpoint:
                cmd.append("--tracks_gaussian_accum_midpoint")
            if job.tracks_gaussian_accum_no_normalize_save:
                cmd.append("--tracks_gaussian_accum_no_normalize_save")
        if job.output_subsampled_world_tracks:
            cmd.append("--output_subsampled_world_tracks")
            cmd.extend(
                [
                    "--subsampled_tracks_num",
                    str(job.subsampled_tracks_num),
                    "--subsampled_tracks_seed",
                    str(job.subsampled_tracks_seed),
                    "--subsampled_tracks_ortho_axes",
                    str(job.subsampled_tracks_ortho_axes),
                    "--subsampled_tracks_video_size",
                    str(job.subsampled_tracks_video_size),
                ]
            )
        if job.no_volumetric_stress_deformation:
            cmd.append("--no_volumetric_stress_deformation")
        if job.output_bc_info:
            cmd.append("--output_bc_info")
        if job.output_force_info:
            cmd.append("--output_force_info")
        if job.output_initial_force_mask_arrow:
            cmd.append("--output_initial_force_mask_arrow")
        if job.pack_arch4_lmdb:
            cmd.append("--pack_arch4_lmdb")
            cmd.extend(
                [
                    "--arch4_lmdb_resize",
                    str(int(job.arch4_lmdb_resize)),
                    "--arch4_lmdb_map_size_gb",
                    str(float(job.arch4_lmdb_map_size_gb)),
                    "--arch4_lmdb_name",
                    str(job.arch4_lmdb_name),
                ]
            )
            if job.pack_arch4_lmdb_include_object_mask:
                cmd.append("--pack_arch4_lmdb_include_object_mask")
        if job.pack_sample_pack:
            cmd.append("--pack_sample_pack")
            cmd.extend(
                [
                    "--sample_pack_name",
                    str(job.sample_pack_name),
                ]
            )
            if job.pack_sample_pack_include_object_mask:
                cmd.append("--pack_sample_pack_include_object_mask")
        if job.pack_arch4_tensors:
            cmd.append("--pack_arch4_tensors")
            cmd.extend(
                [
                    "--arch4_tensor_dtype",
                    str(job.arch4_tensor_dtype),
                ]
            )
        if not job.compress_render_pngs:
            cmd.append("--no_compress_render_pngs")
        cmd.extend(
            [
                "--png_compression_level",
                str(int(job.png_compression_level)),
            ]
        )
        cmd.extend(
            [
                "--render_export_max_side",
                str(int(job.render_export_max_side)),
                "--render_export_scale",
                str(float(job.render_export_scale)),
                "--camera_distance_scale",
                str(float(job.camera_distance_scale)),
            ]
        )
        if job.white_bg:
            cmd.append("--white_bg")
        if job.debug:
            cmd.append("--debug")
        # 多卡时子进程并行写 stderr 会打乱总进度条；默认对子进程加 --quiet
        _verbose_multi = bool(train_cfg.get("verbose_subprocess_multi_gpu", False))
        if quiet_run or (num_gpus > 1 and not _verbose_multi):
            cmd.append("--quiet")

        tasks.append((idx, job, cmd, out_dir))

    if num_gpus <= 1:
        job_pbar = tqdm(
            tasks,
            desc="批量仿真",
            unit="job",
            dynamic_ncols=True,
            smoothing=0.05,
        )
        for idx, job, cmd, out_dir in job_pbar:
            short = os.path.basename(job.ply_path)[:24]
            job_pbar.set_postfix_str(
                f"#{idx + 1}/{len(tasks)} {job.sim_type} {short}",
                refresh=False,
            )
            proc = subprocess.run(cmd, cwd=script_dir, env=_run_env)
            if proc.returncode != 0:
                tqdm.write(f"Job failed (exit={proc.returncode}) out={out_dir}")
                raise RuntimeError(f"Job failed (exit={proc.returncode}): {cmd}")
    else:
        # 多卡并行：按 nvidia-smi 找「空闲」物理 GPU 动态派单；全忙则等待轮询
        max_util_pct = int(train_cfg.get("idle_gpu_max_util_pct", 10))
        # 默认同卡需约 ≥20GiB 空闲再派单，避免「util 低但显存已满」时 Taichi ti.init OOM
        min_free_mib = int(train_cfg.get("idle_gpu_min_free_mib", 20480))
        relax_util_free_mib = int(
            train_cfg.get("idle_gpu_relax_util_when_free_mib", 0)
        )
        poll_wait_s = float(train_cfg.get("idle_gpu_poll_seconds", 2.0))
        smi_failed_assume_idle = bool(
            train_cfg.get("idle_gpu_if_smi_fails_assume_idle", True)
        )

        torch_n = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
        gpu_pool = _all_schedulable_physical_gpu_ids(torch_n)
        if not gpu_pool:
            raise RuntimeError(
                "多卡模式未解析到任何候选 GPU（请检查 CUDA_VISIBLE_DEVICES 或 nvidia-smi）。"
            )
        max_parallel_jobs = min(int(num_gpus), len(gpu_pool))
        if max_parallel_jobs < 1:
            max_parallel_jobs = 1

        if not quiet_run:
            mem_rule = (
                f"free≥{min_free_mib} MiB"
                if min_free_mib > 0
                else "不检查显存（仅 util）"
            )
            relax_rule = (
                f"；free≥{relax_util_free_mib} MiB 时不卡 util"
                if relax_util_free_mib > 0
                else ""
            )
            print(
                f"\n多卡动态调度: 候选池 {len(gpu_pool)} 张卡 {gpu_pool}，"
                f"最多同时 {max_parallel_jobs} 个任务；"
                f"占卡 util≤{max_util_pct}% 且 {mem_rule}{relax_rule}；"
                f"优先选剩余显存最大的卡；不满足则每 {poll_wait_s:g}s 重试。"
            )

        pending = deque(tasks)
        active: Dict[int, Tuple[subprocess.Popen, int, str]] = {}
        failed: List[Tuple[int, int, str]] = []
        smi_warned = False
        last_wait_log_t = 0.0

        pbar = tqdm(
            total=len(tasks),
            desc="批量仿真 (动态GPU)",
            unit="job",
            dynamic_ncols=True,
            smoothing=0.05,
        )

        while pending or active:
            for gid in list(active.keys()):
                popen, idx, out_dir = active[gid]
                ret = popen.poll()
                if ret is not None:
                    del active[gid]
                    pbar.update(1)
                    if ret != 0:
                        failed.append((idx, ret, out_dir))

            states = _nvidia_smi_gpu_states()
            if states is None and not smi_warned and not quiet_run:
                tqdm.write(
                    "[auto_simulation_runner] 无法查询 nvidia-smi；"
                    f"空闲检测改为 assume_idle={smi_failed_assume_idle}（见 train_config idle_gpu_if_smi_fails_assume_idle）。"
                )
                smi_warned = True

            assume_idle = smi_failed_assume_idle or (states is None)
            launched_this_round = 0
            while pending:
                if len(active) >= max_parallel_jobs:
                    break
                chosen: Optional[int] = None
                best_free = -1
                for gid in gpu_pool:
                    if gid in active:
                        continue
                    if not _physical_gpu_looks_idle(
                        gid,
                        states,
                        max_util_pct,
                        min_free_mib,
                        relax_util_when_free_mib=relax_util_free_mib,
                        smi_failed_assume_idle=assume_idle,
                    ):
                        continue
                    free_here = 0
                    if states is not None:
                        for g, f_mib, _u in states:
                            if g == gid:
                                free_here = f_mib
                                break
                    if free_here > best_free:
                        best_free = free_here
                        chosen = gid
                if chosen is None:
                    break
                idx, job, cmd, out_dir = pending.popleft()
                full_cmd = [
                    "env",
                    f"CUDA_VISIBLE_DEVICES={chosen}",
                    sys.executable,
                ] + cmd[1:]
                popen = subprocess.Popen(full_cmd, cwd=script_dir, env=_run_env)
                active[chosen] = (popen, idx, out_dir)
                launched_this_round += 1

            if pending:
                if active:
                    time.sleep(0.2)
                else:
                    now = time.time()
                    if now - last_wait_log_t >= 15.0 and not quiet_run:
                        tqdm.write(
                            f"[auto_simulation_runner] 等待可占卡… 剩余任务 {len(pending)}，"
                            f"已占 {len(active)}/{max_parallel_jobs}；"
                            f"条件 util≤{max_util_pct}%"
                            + (
                                f", free≥{min_free_mib} MiB"
                                if min_free_mib > 0
                                else ""
                            )
                            + (
                                f"（或 free≥{relax_util_free_mib} 忽略 util）"
                                if relax_util_free_mib > 0
                                else ""
                            )
                            + ")"
                        )
                        last_wait_log_t = now
                    time.sleep(poll_wait_s)
            elif active:
                time.sleep(0.2)

        pbar.close()

        if bool(train_cfg.get("boundary_test_mode", False)):
            failed_idx = {idx for idx, _ret, _out_dir in failed}
            successful_tasks = [t for t in tasks if t[0] not in failed_idx]
            _boundary_merge_videos_by_action(
                successful_tasks,
                train_cfg,
                script_dir=script_dir,
                quiet=quiet_run,
            )

        if failed:
            for idx, ret, out_dir in failed:
                tqdm.write(
                    f"失败: job_idx={idx} returncode={ret} out={out_dir}",
                    file=sys.stderr,
                )
            raise RuntimeError(f"有 {len(failed)} 个任务失败，见上方 stderr。")


if __name__ == "__main__":
    run()
