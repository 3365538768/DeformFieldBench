"""Record independent project sources/configs without third-party weights/data."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
files=[]
for folder in ("src","scripts","tests","configs"):
    files.extend(p for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".py",".json",".sh"))
files.extend(root/name for name in ("README.md","pyproject.toml","requirements-runtime.txt"))
hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root/"docs/source_hashes.json").write_text(json.dumps(hashes,indent=2))
path=root/"docs/BUILD_STATUS.json";status=json.loads(path.read_text())
status["source_file_count"]=len(hashes)
status["active_jobs"].append("one-off 16/64-frame giant profile waiting for verified checkpoint")
status["active_jobs"]=list(dict.fromkeys(status["active_jobs"]))
path.write_text(json.dumps(status,indent=2))
print(json.dumps({"source_file_count":len(hashes)}))
