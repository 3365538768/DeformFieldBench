"""Provisional engineering pilot from completed download chunks; no extra packs downloaded.

Uses official remote ZIP central directory, reads only fully received local byte
ranges, and checks each selected member's CRC. Full archive SHA verification is
still required for the final dataset. Published test samples are excluded.
"""
import argparse
import json
from pathlib import Path
import urllib.request
import zipfile
from fetch_sample_subset import HTTPRange, URL, SIZE


def run(archive, chunks, output, split, proxy, count):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    original = json.loads(Path(split).read_text())
    train_ids = set(original["train_ids"])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy, "http": proxy}))
    urllib.request.install_opener(opener)
    stream = HTTPRange(URL, SIZE)
    with zipfile.ZipFile(stream) as remote:
        members = remote.infolist()
    # Central directory was fetched independently; no reading unreceived sparse holes.
    complete = {int(p.stem) for p in Path(chunks).glob("*.json")}
    width = 64*1024**2
    def available(info):
        end = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + info.compress_size + 65536
        return all(i in complete for i in range(info.header_offset//width, end//width+1))
    by_name = {x.filename: x for x in members}
    selected = []
    for pack in members:
        if not pack.filename.endswith("/sample_pack.npz"):
            continue
        parent = pack.filename.rsplit("/", 1)[0]
        if parent.rsplit("/", 1)[-1] not in train_ids:
            continue
        group = [by_name[parent + suffix] for suffix in ("/sample_pack.npz", "/gt.json", "/meta/run_parameters.json", "/meta/boundary_conditions.json")]
        if all(available(x) for x in group):
            selected.append(group)
        if len(selected) == count:
            break
    # Reuse ZipFile's validated metadata with the local sparse stream.
    with zipfile.ZipFile(HTTPRange(URL, SIZE)) as z:
        z.fp.close()
        z.fp = open(archive, "rb")
        for n, group in enumerate(selected, 1):
            for info in group:
                target = (output / info.filename).resolve()
                if not target.is_relative_to(output):
                    raise ValueError("Unsafe path")
                target.parent.mkdir(parents=True, exist_ok=True)
                data = z.read(info)
                target.write_bytes(data)
            if n % 25 == 0:
                print(json.dumps({"extracted": n, "total": len(selected)}), flush=True)
    (output / "receipt.json").write_text(json.dumps({"samples": len(selected), "published_test_excluded": True,
        "member_crc_checked": True, "full_archive_sha_verified": False,
        "purpose": "Nonrandom engineering pilot only; not final quality evidence"}, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for key in ("archive", "chunks", "output", "split", "proxy"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--count", type=int, default=240)
    a=p.parse_args(); run(a.archive,a.chunks,a.output,a.split,a.proxy,a.count)
