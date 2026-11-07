from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from .paths import path_map, relocate_path


@dataclass
class ModelConfig:
    backbone: str = "tiny"  # tiny is for software checks only
    repo: str = ""
    weights: str = ""
    hub_name: str = "dinov3_vitl16"
    width: int = 256
    heads: int = 8
    layers: int = 2
    field_size: int = 56
    field_mode: str = "legacy_visual"
    feedback: str = "fields"  # fields / none / features
    use_prior: bool = False
    freeze_backbone: bool = True
    train_last_blocks: int = 0
    frame_chunk: int = 8
    detach_feedback: bool = False
    architecture: str = "legacy"
    residual_rgb_input: bool = True
    bridge_pooling: str = "legacy"  # normalized_anchor cancels zero-field shortcuts


@dataclass
class DataConfig:
    manifest: str = "data/manifest.jsonl"
    stats: str = "data/train_stats.json"
    frames: int = 16
    max_views: int = 3
    image_size: int = 224
    view_dropout: float = 0.0
    prepared_dir: str = ""


@dataclass
class TrainConfig:
    output: str = "runs/smoke"
    seed: int = 17
    epochs: int = 2
    batch_size: int = 1
    accumulation: int = 1
    workers: int = 0
    lr: float = 0.0003
    backbone_lr: float = 0.000003
    weight_decay: float = 0.05
    field_epochs: int = 0
    field_weight: float = 1.0
    contact_loss: str = "bce"  # bce / balanced_bce / bce_dice; fixed in each comparison
    parameter_weight: float = 1.0
    nll_weight: float = 0.05
    law_weight: float = 0.1
    clip_grad: float = 1.0
    bf16: bool = True
    device: str = "cuda"
    max_steps: int = 0
    resume: str = ""
    initialize: str = ""
    initialize_rgb_baseline: str = ""
    purpose: str = "research"  # smoke / pilot / research
    val_batch_size: int = 2
    val_workers: int = 2
    val_bf16: bool = True
    log_interval: int = 20
    early_stop_patience: int = 5
    early_stop_min_joint_epochs: int = 6
    early_stop_relative_delta: float = 0.002
    field_plateau_patience: int = 3
    compact_checkpoint: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        m, d, t = self.model, self.data, self.train
        if m.architecture not in {"legacy", "local_residual"}:
            raise ValueError("Unknown model architecture")
        if m.architecture == 'local_residual' and (m.width % 4 or not m.freeze_backbone or m.train_last_blocks):
            raise ValueError('Local residual requires width divisible by four and a frozen backbone')
        if m.bridge_pooling not in {'legacy', 'normalized_anchor'}:
            raise ValueError('Unknown bridge pooling')
        if m.bridge_pooling != 'legacy' and (m.architecture != 'local_residual' or m.residual_rgb_input):
            raise ValueError('Anchored pooling requires local residual without direct RGB input')
        if m.feedback not in {"none", "fields", "features", "oracle", "oracle_shuffled"}:
            raise ValueError("Unknown feedback mode")
        if m.field_mode not in {"legacy_visual", "metric"}:
            raise ValueError("Do not mix visual and SI field targets")
        if m.width % m.heads or m.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if min(m.width, m.heads, m.layers, m.field_size, m.frame_chunk) < 1 or m.train_last_blocks < 0:
            raise ValueError("Invalid model dimensions")
        if min(d.frames, d.max_views, d.image_size, t.batch_size, t.accumulation) < 1:
            raise ValueError("Invalid data or batch dimensions")
        if not 0 <= d.view_dropout < 1:
            raise ValueError("Invalid dropout")
        if m.backbone != "tiny" and (not m.repo or not m.weights):
            raise ValueError("Pretrained runs require an explicit local repository and checkpoint")
        if sum(bool(x) for x in (t.resume, t.initialize, t.initialize_rgb_baseline)) > 1:
            raise ValueError("resume and initialization options are mutually exclusive")
        if t.initialize_rgb_baseline and m.architecture != "local_residual":
            raise ValueError("RGB baseline initialization requires local_residual architecture")
        if t.purpose not in {"smoke", "pilot", "research", "diagnostic"} or t.epochs < 1 or min(t.workers, t.val_workers) < 0:
            raise ValueError("Invalid training purpose or schedule")
        if m.feedback.startswith("oracle") and t.purpose != "diagnostic":
            raise ValueError("GT fields are privileged diagnostics only")
        if min(t.val_batch_size, t.log_interval) < 1 or t.early_stop_patience < 0 or t.field_plateau_patience < 0:
            raise ValueError("Invalid validation/logging/early-stop schedule")
        if not 0 <= t.early_stop_relative_delta < 1:
            raise ValueError("Invalid early-stop improvement threshold")
        if t.contact_loss not in {"bce", "balanced_bce", "bce_dice"}:
            raise ValueError("Unknown contact loss")
        return self

    def dump(self):
        return asdict(self)


def load_config(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) - {"model", "data", "train"}:
        raise ValueError("Unknown config section")
    mappings = path_map()
    for section, fields in {'model': ('repo', 'weights'),
                            'data': ('manifest', 'stats', 'prepared_dir'),
                            'train': ('output', 'resume', 'initialize', 'initialize_rgb_baseline')}.items():
        for key in fields:
            if key in raw.get(section, {}):
                raw[section][key] = relocate_path(raw[section][key], mappings)
    return Config(ModelConfig(**raw.get("model", {})), DataConfig(**raw.get("data", {})),
                  TrainConfig(**raw.get("train", {}))).validate()
