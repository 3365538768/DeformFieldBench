# Rebuilding frozen video features on a new host

Derived caches are disposable accelerators, not additional training data. Rebuild
them from the verified raw packs and official backbone weights instead of copying
partial caches between hosts. Keep the object-isolated manifest, statistics,
sampling settings and backbone fingerprint unchanged.

Make a runtime copy of the original cache configuration and change only
`data.prepared_dir` to a fresh directory under `data/prepared`. For example:

```sh
.venv/bin/python scripts/rebuild_feature_cache.py \
  --config configs/runtime/a8002_rebuild_dinov3.json \
  --control-output runs/a8002_rebuild_dinov3 \
  --status-output docs/CACHE_REBUILD_DINOV3.json \
  --prefetch-workers 2 \
  --activate data/prepared/g1f_dinov3l16 \
  --activate data/prepared/g1f_dinov3l16_local
```

Use `configs/g1f_cache.json` as the DINOv3 source and
`configs/g1_B_rgb_only_seed17.json` as the DINOv2 source. Supply the explicit
runtime path map when raw manifests refer to a previous host. Each configuration
uses 512 training and 128 validation samples; no test samples enter the cache.

The supervisor resumes only artifacts produced in its own fresh directory. It
retains one child process, honors the existing global and per-job controls at
sample boundaries, and waits for the dashboard's resume command. Register it as
an `engine_pipeline` job with `progress_report` pointing at its status JSON and an
explicit free-GPU-memory requirement. This reuses the persistent queue and its
pause/resume controls without requeueing completed training runs.

Acceptance requires extraction success, exact fields/masks/targets/times/validity
on four raw-versus-cached train/validation examples, and maximum prediction error
at most 0.001 standardized units. The supervisor additionally checks all cache
IDs and reads every file to check identity, finite feature/supervision values,
view/time dimensions and SHA256. Hashes are recorded in the run directory.
Ground-truth fields remain supervision or privileged diagnostic targets; normal
inference receives RGB features only.

Aliases switch only after acceptance. Existing copied directories are renamed
to `*_migration_retained_<timestamp>` alongside the cache, preserving their
contents. An atomic symlink replacement publishes the rebuilt cache. These
acceptance checks establish cache integrity and inference equivalence, not an
improvement in material estimation or physical simulation accuracy.
