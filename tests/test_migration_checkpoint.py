import copy
import json

import pytest
import torch

from scripts.restore_migration_checkpoint import reconstruct, sha256


def fixture(tmp_path):
    backbone = tmp_path / 'backbone.pt'
    torch.save({'test.weight': torch.arange(4.)}, backbone)
    value = {
        'model': {'head.weight': torch.tensor([8.])},
        'config': {'model': {'freeze_backbone': True, 'train_last_blocks': 0}},
        'optimizer': {'state': {4: {'exp_avg': torch.tensor([.25])}}},
        'rng_states': [torch.arange(5)], 'step': 123,
    }
    embedded = {
        'original_relative_path': 'runs/test/last.pt',
        'frozen_backbone_sha256': sha256(backbone),
        'frozen_backbone_exact_tensor_equality': True,
        'excluded_keys': ['frame_encoder.rgb_mean', 'frame_encoder.rgb_std', 'frame_encoder.net.test.weight'],
        'original_model_key_order': ['frame_encoder.rgb_mean', 'frame_encoder.rgb_std',
                                     'frame_encoder.net.test.weight', 'head.weight'],
    }
    original = copy.deepcopy(value)
    value['migration_frozen_backbone_export'] = embedded
    export = tmp_path / 'last.pt'
    torch.save(value, export)
    receipt = {**embedded, 'export_sha256': sha256(export)}
    export.with_suffix('.migration.json').write_text(json.dumps(receipt))
    return export, backbone, original


def test_reconstruction_retains_optimizer_rng_and_key_order(tmp_path):
    export, backbone, original = fixture(tmp_path)
    value, receipt = reconstruct(export, backbone)
    assert list(value['model']) == receipt['original_model_key_order']
    assert torch.equal(value['model']['frame_encoder.net.test.weight'], torch.arange(4.))
    assert value['model']['frame_encoder.rgb_mean'].shape == (1, 3, 1, 1)
    assert torch.equal(value['optimizer']['state'][4]['exp_avg'], original['optimizer']['state'][4]['exp_avg'])
    assert torch.equal(value['rng_states'][0], original['rng_states'][0])
    assert value['step'] == 123 and 'migration_frozen_backbone_export' not in value


@pytest.mark.parametrize('target', ['export', 'backbone', 'sidecar'])
def test_changed_content_fails_closed(tmp_path, target):
    export, backbone, _ = fixture(tmp_path)
    if target == 'sidecar':
        path = export.with_suffix('.migration.json')
        data = json.loads(path.read_text())
        data['original_relative_path'] = 'changed'
        path.write_text(json.dumps(data))
    else:
        path = export if target == 'export' else backbone
        with path.open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError):
        reconstruct(export, backbone)
