#!/usr/bin/env python3
"""Validate a DeformFieldBench environment without loading data or weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import shutil
import sys
from typing import Iterable


COMMON = ("numpy", "cv2", "PIL", "yaml", "torch", "tensorboard")
SIMULATION = ("taichi", "warp", "mcubes", "pymeshlab", "diff_gaussian_rasterization", "simple_knn")
VLM = ("dashscope", "openai", "volcenginesdkarkruntime")


def _version(module: object, import_name: str) -> str:
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    package_names = {
        "cv2": "opencv-python-headless",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "warp": "warp-lang",
        "mcubes": "PyMCubes",
        "volcenginesdkarkruntime": "volcengine-python-sdk",
    }
    try:
        return importlib.metadata.version(package_names.get(import_name, import_name))
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _check_imports(names: Iterable[str]) -> list[str]:
    failures: list[str] = []
    for name in names:
        try:
            module = importlib.import_module(name)
            print(f"[ok] {name} {_version(module, name)}")
        except Exception as exc:  # import errors often include actionable ABI details
            failures.append(name)
            print(f"[failed] {name}: {type(exc).__name__}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("train", "simulation", "vlm", "all"), default="train"
    )
    args = parser.parse_args()

    names = list(COMMON)
    if args.profile in ("simulation", "all"):
        names.extend(SIMULATION)
    if args.profile in ("vlm", "all"):
        names.extend(VLM)
    failures = _check_imports(dict.fromkeys(names))

    if args.profile in ("simulation", "all"):
        path = shutil.which("ffmpeg")
        if path:
            print(f"[ok] ffmpeg {path}")
        else:
            failures.append("ffmpeg")
            print("[failed] ffmpeg: executable not found")

    print(f"python={sys.version.split()[0]} executable={sys.executable}")
    if failures:
        print("missing or broken: " + ", ".join(failures))
        return 1
    print(f"environment profile '{args.profile}' is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
