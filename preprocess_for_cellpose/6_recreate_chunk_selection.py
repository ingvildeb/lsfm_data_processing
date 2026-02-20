"""
Recreate an existing chunk selection from newly preprocessed source images.

The script reads chunk filenames from an existing chunk folder, infers source-image
stem + chunk coordinates using underscore-index logic from config. It then finds matching source images 
in subject-specific directories, re-extracts only those chunks, and writes them to a new output folder 
without touching originals.

Option to also copy npy files for the re-created chunks.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import sys

import tifffile

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_helpers import load_script_config, normalize_user_path, require_dir


def list_tif_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])


def find_source_candidates(subject_files: list[Path], source_stem: str) -> list[Path]:
    return [p for p in subject_files if source_stem in p.stem]


def parse_chunk_filename(
    chunk_path: Path,
    underscores_to_sample_id: int,
    underscores_to_mip: int,
    underscores_to_chunk: int,
) -> dict[str, object]:
    parts = chunk_path.stem.split("_")

    required_index = max(
        underscores_to_sample_id,
        underscores_to_mip,
        underscores_to_chunk + 2,
    )
    if len(parts) <= required_index:
        raise RuntimeError(
            f"Chunk filename has fewer underscore parts than configured indices:\n"
            f"{chunk_path.name}\nparts={parts}"
        )

    if parts[underscores_to_mip] != "MIP":
        raise RuntimeError(
            f"Expected 'MIP' token at index {underscores_to_mip} in:\n{chunk_path.name}\nparts={parts}"
        )

    if parts[underscores_to_chunk] != "chunk":
        raise RuntimeError(
            f"Expected 'chunk' token at index {underscores_to_chunk} in:\n{chunk_path.name}\nparts={parts}"
        )

    y = int(parts[underscores_to_chunk + 1])
    x = int(parts[underscores_to_chunk + 2])
    sample_id = parts[underscores_to_sample_id]

    source_stem = "_".join(parts[underscores_to_mip:underscores_to_chunk])
    if source_stem == "":
        raise RuntimeError(f"Could not infer source stem from:\n{chunk_path.name}")

    old_chunk = tifffile.TiffFile(chunk_path).asarray()
    return {
        "chunk_path": chunk_path,
        "y": y,
        "x": x,
        "old_shape": old_chunk.shape,
        "sample_id": sample_id,
        "source_stem": source_stem,
    }


# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "6_recreate_chunk_selection", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------
existing_chunk_dir = require_dir(
    normalize_user_path(cfg["existing_chunk_dir"]),
    "Existing chunk directory",
)

subject_to_new_image_dir = {
    str(k): require_dir(normalize_user_path(v), f"New image directory for subject {k}")
    for k, v in cfg["subject_to_new_image_dir"].items()
}

output_chunk_dir = normalize_user_path(cfg["output_chunk_dir"])
copy_seg_files = cfg["copy_seg_files"]
strict_bounds = cfg["strict_bounds"]
underscores_to_sample_id = cfg["underscores_to_sample_id"]
underscores_to_mip = cfg["underscores_to_mip"]
underscores_to_chunk = cfg["underscores_to_chunk"]

# -------------------------
# OUTPUT GUARDS
# -------------------------
if output_chunk_dir.resolve() == existing_chunk_dir.resolve():
    raise RuntimeError("output_chunk_dir must be different from existing_chunk_dir.")

output_chunk_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# INDEX NEW IMAGES
# -------------------------
subject_to_files: dict[str, list[Path]] = {}
for subject_id, search_dir in subject_to_new_image_dir.items():
    files = list_tif_files(search_dir)
    if len(files) == 0:
        raise RuntimeError(f"No TIFF images found in subject directory:\n{search_dir}")
    subject_to_files[subject_id] = files

# -------------------------
# PARSE EXISTING CHUNK SET
# -------------------------
existing_chunks = list_tif_files(existing_chunk_dir)
if not existing_chunks:
    raise RuntimeError(f"No TIFF chunks found in:\n{existing_chunk_dir}")

# Warn about orphan annotation files that cannot be copied because the matching
# chunk TIFF file does not exist.
existing_seg_files = sorted(existing_chunk_dir.glob("*_seg.npy"))
existing_chunk_stems = {p.stem for p in existing_chunks}
orphan_seg_files = [p for p in existing_seg_files if p.stem[:-4] not in existing_chunk_stems]
if orphan_seg_files:
    print(
        f"Warning: found {len(orphan_seg_files)} orphan *_seg.npy file(s) "
        "without matching chunk TIFF in existing_chunk_dir."
    )
    for p in orphan_seg_files[:10]:
        print(f"  Orphan annotation: {p.name}")
    if len(orphan_seg_files) > 10:
        print(f"  ... and {len(orphan_seg_files) - 10} more")

jobs_by_source: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

for chunk_path in existing_chunks:
    parsed = parse_chunk_filename(
        chunk_path=chunk_path,
        underscores_to_sample_id=underscores_to_sample_id,
        underscores_to_mip=underscores_to_mip,
        underscores_to_chunk=underscores_to_chunk,
    )
    sample_id = parsed["sample_id"]  # type: ignore[assignment]
    source_stem = parsed["source_stem"]  # type: ignore[assignment]
    jobs_by_source[(sample_id, source_stem)].append(parsed)

# upfront completeness check: require all source images to exist uniquely before writing anything
missing_subjects: list[str] = []
missing_stems: list[str] = []
ambiguous_stems: list[str] = []
resolved_source_path: dict[tuple[str, str], Path] = {}

for sample_id, source_stem in sorted(jobs_by_source.keys()):
    subject_files = subject_to_files.get(sample_id)
    if subject_files is None:
        missing_subjects.append(sample_id)
        continue

    candidates = find_source_candidates(subject_files, source_stem)
    if len(candidates) == 0:
        missing_stems.append(f"{sample_id}:{source_stem}")
    elif len(candidates) > 1:
        ambiguous_stems.append(f"{sample_id}:{source_stem}")
    else:
        resolved_source_path[(sample_id, source_stem)] = candidates[0]

if missing_subjects or missing_stems or ambiguous_stems:
    msg_lines = ["Failed preflight source-image match check. No output files were written."]
    if missing_subjects:
        msg_lines.append("Missing subject IDs in subject_to_new_image_dir:")
        msg_lines.extend(sorted(set(missing_subjects))[:50])
    if missing_stems:
        msg_lines.append("Missing source stems (sample_id:source_stem):")
        msg_lines.extend(missing_stems[:50])
    if ambiguous_stems:
        msg_lines.append("Ambiguous source stems (sample_id:source_stem):")
        msg_lines.extend(ambiguous_stems[:50])
    raise RuntimeError("\n".join(msg_lines))

# -------------------------
# RECREATE CHUNKS
# -------------------------
written = 0
skipped = 0
copied_seg = 0

for (sample_id, source_stem), jobs in jobs_by_source.items():
    source_image_path = resolved_source_path[(sample_id, source_stem)]
    source_img = tifffile.TiffFile(source_image_path).asarray()

    for job in jobs:
        chunk_path: Path = job["chunk_path"]  # type: ignore[assignment]
        y = int(job["y"])  # type: ignore[arg-type]
        x = int(job["x"])  # type: ignore[arg-type]
        old_shape = tuple(job["old_shape"])  # type: ignore[arg-type]

        if len(old_shape) == 2:
            if source_img.ndim != 2:
                msg = f"Dim mismatch for {chunk_path.name}: old chunk is 2D but source is {source_img.ndim}D"
                if strict_bounds:
                    raise RuntimeError(msg)
                print(f"Skipping: {msg}")
                skipped += 1
                continue

            h, w = old_shape
            if y + h > source_img.shape[0] or x + w > source_img.shape[1]:
                msg = f"Out-of-bounds chunk {chunk_path.name} for source {source_image_path.name}"
                if strict_bounds:
                    raise RuntimeError(msg)
                print(f"Skipping: {msg}")
                skipped += 1
                continue

            new_chunk = source_img[y : y + h, x : x + w]

        elif len(old_shape) == 3:
            if source_img.ndim != 3:
                msg = f"Dim mismatch for {chunk_path.name}: old chunk is 3D but source is {source_img.ndim}D"
                if strict_bounds:
                    raise RuntimeError(msg)
                print(f"Skipping: {msg}")
                skipped += 1
                continue

            z, h, w = old_shape
            if source_img.shape[0] != z:
                msg = (
                    f"Z-size mismatch for {chunk_path.name}: old={z}, source={source_img.shape[0]} "
                    f"({source_image_path.name})"
                )
                if strict_bounds:
                    raise RuntimeError(msg)
                print(f"Skipping: {msg}")
                skipped += 1
                continue

            if y + h > source_img.shape[1] or x + w > source_img.shape[2]:
                msg = f"Out-of-bounds chunk {chunk_path.name} for source {source_image_path.name}"
                if strict_bounds:
                    raise RuntimeError(msg)
                print(f"Skipping: {msg}")
                skipped += 1
                continue

            new_chunk = source_img[:, y : y + h, x : x + w]

        else:
            msg = f"Unsupported chunk dimensionality ({len(old_shape)}) for {chunk_path.name}"
            if strict_bounds:
                raise RuntimeError(msg)
            print(f"Skipping: {msg}")
            skipped += 1
            continue

        out_chunk_path = output_chunk_dir / chunk_path.name
        if out_chunk_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing file:\n{out_chunk_path}")

        tifffile.imwrite(out_chunk_path, new_chunk)
        written += 1

        if copy_seg_files:
            src_seg = chunk_path.with_name(f"{chunk_path.stem}_seg.npy")
            if src_seg.exists():
                dst_seg = output_chunk_dir / src_seg.name
                if dst_seg.exists():
                    raise RuntimeError(f"Refusing to overwrite existing file:\n{dst_seg}")
                shutil.copy2(src_seg, dst_seg)
                copied_seg += 1

print("Recreate chunk selection complete.")
print(f"Input chunks scanned: {len(existing_chunks)}")
print(f"Chunks written: {written}")
print(f"Chunks skipped: {skipped}")
if copy_seg_files:
    print(f"Annotation files copied: {copied_seg}")
