# DeformFieldBench

This repository contains the runnable code used in the paper: sample-pack simulation, supervised and logic-model training, parameter and field evaluation, ambiguity replay, and VLM benchmark runners. The release is intentionally compact and keeps only the code paths used for the reported experiments.

## Resources

- Dataset: [Physical Field Material Parameter 5000 on Kaggle](https://www.kaggle.com/datasets/anonymous336/physical-field-material-parameter-5000)
- Checkpoints: [HandsomeHusky/DeformFieldBench on Hugging Face](https://huggingface.co/HandsomeHusky/DeformFieldBench/tree/main)
- Checkpoint manifest: `pretrained/models.yaml`

## Artifact Checklist

This release is organized for reproducibility:

- Training code for the supervised baseline and the final logic model.
- Logic-model ablation entry points used in the paper.
- Evaluation code for parameter metrics, field metrics, and parameter ambiguity replay.
- Sample-pack simulation pipeline aligned with the released dataset format.
- VLM benchmark code with API keys loaded only from environment variables.
- Download scripts and manifests for public checkpoints.

## Environment

Create or activate a Python environment, then install the core dependencies:

```bash
pip install -r requirements.txt
```

For VLM evaluation, also install the VLM dependencies and provide API keys through environment variables:

```bash
pip install -r requirements-vlm.txt
export DASHSCOPE_API_KEY=your_key
export OPENAI_API_KEY=your_key
export ARK_API_KEY=your_key
```

API keys are never stored in the codebase. VLM clients read them only from environment variables.

## Dataset

Download the dataset from Kaggle:

[https://www.kaggle.com/datasets/anonymous336/physical-field-material-parameter-5000](https://www.kaggle.com/datasets/anonymous336/physical-field-material-parameter-5000)

The training and evaluation code expects the dataset to follow this layout:

```text
auto_output/dataset_5000/train/<sample_id>/sample_pack.npz
auto_output/dataset_5000/train_test_split.cleaned.json
```

Each `sample_pack.npz` stores the rendered RGB sequence, stress field, flow field, force mask, and optionally object mask arrays. The released scripts use this package as the common interface between simulation, model training, evaluation, and VLM prompting.

## Checkpoints

Large checkpoint files are hosted separately on Hugging Face:

[https://huggingface.co/HandsomeHusky/DeformFieldBench/tree/main](https://huggingface.co/HandsomeHusky/DeformFieldBench/tree/main)

To download the checkpoints listed in `pretrained/models.yaml`, run:

```bash
bash pretrained/download_models.sh
```

Expected files:

```text
pretrained/logic_baseline_epoch_0320.pt
pretrained/supervised_baseline_last.pt
pretrained/logic_abl_no_boundary.pt
pretrained/logic_abl_no_field_aux.pt
pretrained/logic_abl_no_multistage.pt
pretrained/logic_plus_phys_01.pt
```

## Quick Start

After installing dependencies, downloading the dataset, and downloading checkpoints, run the main parameter evaluation:

```bash
MODEL=logic \
CHECKPOINT=pretrained/logic_baseline_epoch_0320.pt \
OUT_DIR=outputs/param_eval/logic \
bash scripts/eval_param_metrics.sh
```

For the supervised baseline:

```bash
MODEL=my_model \
WEIGHTS=pretrained/supervised_baseline_last.pt \
OUT_DIR=outputs/param_eval/my_model \
bash scripts/eval_param_metrics.sh
```

Evaluation outputs include `summary.json`, `score.json`, `param_metrics/`, and `field_metrics/`.

## Simulation

The simulation entry point generates a `sample_pack.npz` compatible with the released training and evaluation code:

```bash
PLY_PATH=/path/to/object.ply \
CONFIG=configs/simulation/train_config_dataset_full.json \
OUTPUT_PATH=outputs/simulation_sample \
SIM_TYPE=press \
bash scripts/run_simulation_sample_pack.sh
```

The exported sample pack contains RGB frames, stress Gaussians, flow Gaussians, force masks, object masks, and metadata needed by downstream modules.

## Training

Train the supervised baseline:

```bash
NUM_GPUS=8 bash scripts/train_my_model_param_only.sh
```

Train the final logic model:

```bash
NUM_GPUS=8 bash scripts/train_logic.sh
```

Train a logic-model ablation:

```bash
CONFIG=configs/logic_model/final_logic_413_5000_new.json \
NUM_GPUS=8 \
bash scripts/train_logic_ablation.sh
```

For paper ablations, use the corresponding config and checkpoint names listed in `pretrained/models.yaml`.

## Parameter Ambiguity Replay

```bash
DATASET_ROOT=auto_output/dataset_5000/train \
RECORDS=outputs/param_eval/logic/param_metrics/sample_records.jsonl \
OUT_DIR=outputs/param_ambiguity/logic \
REPLAY_NUM_SAMPLES=0 \
bash scripts/eval_param_ambiguity.sh
```

Outputs include `replay_metrics.csv`, `replay_metrics.jsonl`, `replay_summary.json`, `replay_selection.json`, and rendered replay samples. `REPLAY_NUM_SAMPLES=0` replays all available records.

## VLM Benchmark

```bash
export DASHSCOPE_API_KEY=your_key
DATASET_ROOT=auto_output/dataset_5000/train \
OUTPUT_DIR=outputs/vlm_benchmark \
bash scripts/run_vlm_benchmark.sh
```

Additional model providers can be enabled with `OPENAI_API_KEY` and `ARK_API_KEY`. The benchmark code samples 64-frame inputs for released VLM evaluation settings.

## Repository Layout

```text
configs/          Released training, simulation, and evaluation configs
simulation/       Sample-pack simulation pipeline
my_model/         Supervised material-parameter baseline
logic_model/      Logic model and ablation training code
eval_abalation/   Parameter, field, and ambiguity evaluation
vlm_benchmark/    VLM benchmark runners, prompts, and clients
scripts/          Reproducible command-line entry points
pretrained/       Checkpoint manifest and download script
outputs/          Default local output directory
```

