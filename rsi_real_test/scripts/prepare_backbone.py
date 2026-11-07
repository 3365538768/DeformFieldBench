"""Pin a local upstream checkout. Weight download must be explicit.

DINOv3 checkpoint access follows Meta's license/access process. No fallback to
random initialization or unauthorized mirrors. DINOv2 weights use official URLs.
"""
import argparse
from pathlib import Path
import json
import subprocess
import urllib.request


def prepare(family, revision, download, variant="vitl14"):
    repo = Path("external") / family
    repo.parent.mkdir(exist_ok=True)
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/facebookresearch/{family}.git", str(repo)], check=True)
    if revision:
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    entry = {"family": family, "commit": commit, "repo": str(repo.resolve())}
    if download:
        if family != "dinov2":
            raise ValueError("Supply locally licensed DINOv3 weights; no automatic access requests")
        target = Path(f"weights/dinov2_{variant}_reg4_pretrain.pth")
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{variant}/dinov2_{variant}_reg4_pretrain.pth", tmp)
            tmp.replace(target)
        entry["weights"] = str(target.resolve())
    lock = Path("external") / f"{family}.lock.json"
    lock.write_text(json.dumps(entry, indent=2)); print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--family", choices=["dinov2", "dinov3"], required=True)
    p.add_argument("--revision", default=""); p.add_argument("--download-dinov2", action="store_true")
    p.add_argument("--variant", choices=["vits14", "vitb14", "vitl14", "vitg14"], default="vitl14")
    a = p.parse_args(); prepare(a.family, a.revision, a.download_dinov2, a.variant)
