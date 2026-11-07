"""Explicit runtime relocation; immutable manifests and cache hashes stay intact."""
import json
import os
from pathlib import Path


def path_map():
    filename = os.environ.get('FIELDBRIDGE_PATH_MAP')
    mappings = json.loads(Path(filename).read_text()) if filename else {}
    if not isinstance(mappings, dict):
        raise ValueError('Path map must be an object of absolute directory pairs')
    for source, target in mappings.items():
        if not isinstance(target, str) or not Path(source).is_absolute() or not Path(target).is_absolute():
            raise ValueError('Path map requires absolute directory pairs')
    return mappings


def relocate_path(value, mappings):
    if not value or not Path(value).is_absolute():
        return value
    path = Path(value).resolve()
    for source in sorted(mappings, key=lambda key: len(Path(key).parts), reverse=True):
        origin = Path(source).resolve()
        if path.is_relative_to(origin):
            return str(Path(mappings[source]).resolve() / path.relative_to(origin))
    return value
