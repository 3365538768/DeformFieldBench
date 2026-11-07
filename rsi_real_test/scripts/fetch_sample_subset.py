"""Read ZIP central directory and selected members via verified HTTP byte ranges.

Avoids downloading the 41.4GB archive for schema/smoke checks. ZIP CRC is checked
by zipfile. Does NOT establish whole-dataset integrity or random sample quality.
"""
import argparse
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile

URL = "https://huggingface.co/datasets/HandsomeHusky/DeformFieldBench/resolve/main/dataset.zip"
SIZE = 41425509436


class HTTPRange(io.RawIOBase):
    def __init__(self, url, size, cache_bust=True):
        self.url, self.size, self.position = url, size, 0
        self.cache_bust = cache_bust
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid seek")
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        if n == 0:
            return b""
        if n > 256 * 1024**2:
            raise ValueError("Refusing unexpectedly large range")
        start, end = self.position, self.position + n - 1
        # Distinct cache key prevents intermediaries reusing a different byte range.
        separator = "&" if "?" in self.url else "?"
        url = self.url + separator + f"download=true&fieldbridge_range={start}-{end}" if self.cache_bust else self.url
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "FieldBridge/0.1", "Cache-Control": "no-cache"})
        last = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as r:
                    expected = f"bytes {start}-{end}/{self.size}"
                    if r.status != 206 or r.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor exact range")
                    data = r.read(n + 1)
                if len(data) != n:
                    raise IOError("Truncated range")
                self.position += n
                self.downloaded += n
                return data
            except Exception as error:
                last = error
        raise last


def run(output, count, list_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stream = HTTPRange(URL, SIZE)
    with zipfile.ZipFile(stream) as archive:
        members = archive.infolist()
        (output / "archive_index.json").write_text(json.dumps(
            [{"name": x.filename, "size": x.file_size, "compressed": x.compress_size} for x in members], indent=2))
        packs = [x for x in members if x.filename.endswith("/sample_pack.npz")]
        gt_names = {x.filename for x in members if x.filename.endswith("/gt.json")}
        print(json.dumps({"members": len(members), "packs": len(packs), "gt": len(gt_names),
                          "first_names": [x.filename for x in members[:20]]}), flush=True)
        if list_only:
            return
        by_name = {x.filename: x for x in members}
        selected, objects = [], set()
        # Read small labels first; select distinct physical objects rather than adjacent variants.
        stride = max(1, len(packs) // max(count, 1))
        candidates = packs[::stride] + packs
        for info in candidates:
            label_name = info.filename.rsplit("/", 1)[0] + "/gt.json"
            if label_name not in gt_names:
                continue
            label_bytes = archive.read(label_name)
            label = json.loads(label_bytes)
            obj = label.get("object")
            if not obj or obj in objects:
                continue
            objects.add(obj)
            selected.append((info, label_name, label_bytes, obj))
            if len(selected) >= count:
                break
        if len(selected) < 3:
            raise ValueError("Insufficient distinct objects for a safe smoke split")
        for info, label_name, label_bytes, obj in selected:
            for name in [label_name, info.filename]:
                target = (output / name).resolve()
                if not target.is_relative_to(output):
                    raise ValueError("Unsafe ZIP path")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    data = label_bytes if name == label_name else archive.read(info)
                    target.write_bytes(data)
            print(json.dumps({"extracted_object": obj, "pack_bytes": info.file_size}), flush=True)
        receipt = {"url": URL, "archive_bytes": SIZE, "range_downloaded_bytes": stream.downloaded,
                   "selected_objects": sorted(objects), "zip_crc_checked": True,
                   "purpose": "Nonrandom real-data software smoke only; not research results"}
        (output / "subset_receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--objects", type=int, default=6)
    parser.add_argument("--list-only", action="store_true")
    a = parser.parse_args(); run(a.output, a.objects, a.list_only)
