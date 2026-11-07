import importlib.util
from pathlib import Path
import json
import pytest
import torch
import numpy as np
from fieldbridge.config import Config, ModelConfig
from fieldbridge.data import MaterialDataset, ParameterCodec, collate, fit_stats, model_inputs, read_manifest
from fieldbridge.model import MaterialFieldModel
from fieldbridge.losses import compute_loss, balanced_contact_bce

spec = importlib.util.spec_from_file_location("fixture_generator", Path(__file__).parents[1] / "scripts/make_fixture.py")
fixture_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture_module)


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    manifest = fixture_module.create(tmp_path)
    stats_path = tmp_path / "stats.json"
    fit_stats(manifest, stats_path)
    cfg = Config(model=ModelConfig(width=32, heads=4, layers=1, field_size=8))
    cfg.data.manifest, cfg.data.stats = str(manifest), str(stats_path)
    cfg.data.image_size, cfg.data.frames = 32, 4
    ds = MaterialDataset(cfg.data, "train", 8, "legacy_visual")
    return cfg, ds


def test_leakage_is_rejected(setup, tmp_path):
    cfg, _ = setup
    rows = [json.loads(l) for l in Path(cfg.data.manifest).read_text().splitlines()]
    rows[-1]["object_id"] = rows[0]["object_id"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(bad)


def test_statistics_use_train_only(setup):
    cfg, _ = setup
    stats = json.loads(Path(cfg.data.stats).read_text())
    assert stats["count"] == [6, 6, 6, 3]
    codec = ParameterCodec(stats)
    y, mask = codec.encode({"E": 1234, "nu": .3, "rho": 998, "sigma_y": None})
    assert not mask[-1] and y[-1] == 0
    decoded = codec.decode(y)
    torch.testing.assert_close(decoded[:3], torch.tensor([1234., .3, 998.]), rtol=1e-5, atol=1e-4)


def test_no_labels_enter_inference_and_padding_is_invariant(setup):
    cfg, ds = setup
    torch.manual_seed(3)
    model = MaterialFieldModel(cfg.model).eval()
    a, b = ds[0], ds[1]
    single = collate([a]); padded = collate([a, b])
    with torch.no_grad():
        baseline = model(**model_inputs(single))["mean"]
        together = model(**model_inputs(padded))["mean"][:1]
        padded["target"] += 100
        padded["fields"] *= 0
        altered = model(**model_inputs(padded))["mean"][:1]
    torch.testing.assert_close(baseline, together, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(altered, together)
    assert set(model_inputs(single)) == {"rgb", "valid", "times", "prior", "prior_mask"}


def test_field_feedback_isolates_parameter_gradient(setup):
    cfg, ds = setup
    batch = collate([ds[0], ds[1]])
    cfg.model.feedback = "none"
    model = MaterialFieldModel(cfg.model)
    out = model(**model_inputs(batch))
    out["mean"].sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.contact_flow.parameters())
    model.zero_grad(set_to_none=True)
    loss, _ = compute_loss(model(**model_inputs(batch)), batch, cfg)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.contact_flow.parameters())
    cfg.model.feedback = "fields"
    model = MaterialFieldModel(cfg.model)
    model(**model_inputs(batch))["mean"].sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.contact_flow.parameters())


def test_view_permutation_and_matched_parameter_counts(setup):
    cfg, ds = setup
    batch = collate([ds[1]])
    torch.manual_seed(6)
    m = MaterialFieldModel(cfg.model).eval()
    with torch.no_grad():
        a = m(**model_inputs(batch))["mean"]
        for key in ("rgb", "valid", "times"):
            batch[key] = batch[key].flip(1)
        b = m(**model_inputs(batch))["mean"]
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    counts = []
    for feedback in ("none", "fields", "features"):
        cfg.model.feedback = feedback
        counts.append(sum(p.numel() for p in MaterialFieldModel(cfg.model).parameters()))
    assert len(set(counts)) == 1


def test_metric_fields_keep_binary_contact_and_explicit_units(tmp_path):
    root = tmp_path
    rgb = np.zeros((1, 3, 3, 32, 32), dtype=np.uint8)
    fields = np.ones((1, 3, 4, 32, 32), np.float32)
    fields[:, :, :1, :, :16] = 0
    fields[:, :, 1:3] = -2
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        pack = root / f"metric_{i}.npz"
        np.savez(pack, rgb=rgb, fields=fields, field_mask=np.ones_like(fields), timestamps=np.array([0., .1, .2]))
        rows.append({"sample_id": str(i), "object_id": str(i), "source": "metric_fixture", "split": split,
                     "pack": str(pack), "parameters": {"E": 1234, "nu": .3}, "field_mode": "metric",
                     "field_units": ["binary", "pixel_per_second", "pixel_per_second", "pascal"], "field_scales": [2, 1]})
    manifest = root / "manifest.jsonl"; manifest.write_text("\n".join(json.dumps(r) for r in rows))
    stats = root / "stats.json"; fit_stats(manifest, stats)
    cfg = Config(); cfg.data.manifest = str(manifest); cfg.data.stats = str(stats)
    cfg.data.image_size = 32
    ds = MaterialDataset(cfg.data, "train", 7, "metric")
    sample = ds[0]
    assert set(sample["fields"][:, :, 0].unique().tolist()) == {0., 1.}
    torch.testing.assert_close(sample["fields"][:, :, 1:3], -torch.ones_like(sample["fields"][:, :, 1:3]))
    with pytest.raises(ValueError, match="Mixed field"):
        MaterialDataset(cfg.data, "train", 7, "legacy_visual")[0]


def test_physical_timestamps_are_not_normalized(setup):
    cfg, _ = setup
    path = Path(cfg.data.manifest)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows[0]["timestamps_seconds"] = [0., 3.99]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    fit_stats(path, cfg.data.stats)
    sample = MaterialDataset(cfg.data, "train", 8, "legacy_visual")[0]
    torch.testing.assert_close(sample["times"], torch.tensor([[0., 3.99]]))
    rows[0]["timestamps_seconds"] = [0., 0.]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    fit_stats(path, cfg.data.stats)
    with pytest.raises(ValueError, match="timestamps"):
        MaterialDataset(cfg.data, "train", 8, "legacy_visual")[0]


def test_published_test_ids_preserved_and_missing_pack_fails(tmp_path):
    spec = importlib.util.spec_from_file_location("importer", Path(__file__).parents[1]/"scripts/import_deformfield.py")
    importer = importlib.util.module_from_spec(spec); spec.loader.exec_module(importer)
    root = tmp_path / "samples"
    for i in range(5):
        folder = root / f"{i:06d}"; folder.mkdir(parents=True)
        (folder / "gt.json").write_text(json.dumps({"object": f"o{i}", "params": {"material": "jelly", "E": 1000, "nu": .3, "density": 900}}))
        if i != 4:
            (folder / "sample_pack.npz").write_bytes(b"index-only fixture")
    split = tmp_path / "published.json"
    split.write_text(json.dumps({"train_ids": ["000000", "000001", "000002"], "test_ids": ["000003"], "test_objects": ["o3"]}))
    output = tmp_path / "manifest.jsonl"
    importer.build(root, output, original_split=split)
    rows = read_manifest(output)
    assert {r["object_id"] for r in rows if r["split"] == "test"} == {"o3"}
    assert sum(r["split"] == "val" for r in rows) == 1
    assert sum(r["split"] == "train" for r in rows) == 2
    (root / "000002/sample_pack.npz").unlink()
    with pytest.raises(ValueError, match="no pack"):
        importer.build(root, tmp_path / "missing.jsonl", original_split=split)


def test_sparse_contact_receives_balanced_gradient_and_empty_masks_are_safe():
    # A uniform background shortcut must not overwhelm one rare contact pixel.
    logits = torch.zeros(10000, requires_grad=True)
    target = torch.zeros_like(logits); target[0] = 1
    loss = balanced_contact_bce(logits, target, torch.ones_like(target))
    loss.backward()
    assert logits.grad[0] < 0 and (logits.grad[1:] > 0).all()
    torch.testing.assert_close(-logits.grad[0], logits.grad[1:].sum())
    background_only = balanced_contact_bce(torch.zeros(10), torch.zeros(10), torch.ones(10))
    torch.testing.assert_close(background_only, torch.tensor(2.).log())
    assert balanced_contact_bce(torch.zeros(10), torch.zeros(10), torch.zeros(10)) == 0
