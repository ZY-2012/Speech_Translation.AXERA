#!/usr/bin/env python3
"""Generate calibration archives for four-way split fm_decoder_part3."""

from __future__ import annotations

import argparse
import io
import shutil
import tarfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calibration data for fm_decoder_part3_0..3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-calib-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_tar_arrays(tar_path: Path) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        members.sort(key=lambda member: member.name)
        for member in members:
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            arrays.append(np.load(io.BytesIO(fileobj.read()), allow_pickle=False))
    return arrays


def _write_tar(arrays: list[np.ndarray], output_dir: Path, tensor_name: str) -> None:
    tensor_dir = output_dir / tensor_name
    tensor_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / f"{tensor_name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for index, array in enumerate(arrays):
            value = np.asarray(array)
            if value.shape == ():
                value = value.reshape(1)
            npy_path = tensor_dir / f"{index:05d}.npy"
            np.save(npy_path, value)
            tar.add(npy_path, arcname=f"{tensor_name}/{npy_path.name}")


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    source_calib_dir = Path(args.source_calib_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    time_hidden = _load_tar_arrays(source_calib_dir / "time_hidden.tar.gz")
    padding_mask = _load_tar_arrays(source_calib_dir / "padding_mask.tar.gz")
    hidden = _load_tar_arrays(source_calib_dir / "decoder_hidden_p2.tar.gz")
    count = len(hidden)
    if len(time_hidden) != count or len(padding_mask) != count:
        raise ValueError(
            "mismatched calibration counts: "
            f"decoder_hidden_p2={len(hidden)}, time_hidden={len(time_hidden)}, "
            f"padding_mask={len(padding_mask)}"
        )

    input_names = [
        "decoder_hidden_p2",
        "decoder_hidden_p30",
        "decoder_hidden_p31",
        "decoder_hidden_p32",
    ]
    output_names = [
        "decoder_hidden_p30",
        "decoder_hidden_p31",
        "decoder_hidden_p32",
        "v",
    ]

    current = hidden
    for part_index in range(4):
        part_dir = output_dir / f"fm_decoder_part3_{part_index}"
        part_dir.mkdir(parents=True, exist_ok=True)
        in_name = input_names[part_index]
        _write_tar(current, part_dir, in_name)
        _write_tar(time_hidden, part_dir, "time_hidden")
        _write_tar(padding_mask, part_dir, "padding_mask")

        if part_index == 3:
            break

        sess = _session(model_dir / f"fm_decoder_part3_{part_index}.onnx")
        out_name = output_names[part_index]
        next_hidden: list[np.ndarray] = []
        for sample_index in range(count):
            feed = {
                in_name: np.asarray(current[sample_index], dtype=np.float32),
                "time_hidden": np.asarray(time_hidden[sample_index], dtype=np.float32),
                "padding_mask": np.asarray(padding_mask[sample_index]).astype(np.bool_),
            }
            out = sess.run([out_name], feed)[0].astype(np.float32)
            next_hidden.append(out)
        current = next_hidden

    print(f"Generated {count} samples under {output_dir}")


if __name__ == "__main__":
    main()
