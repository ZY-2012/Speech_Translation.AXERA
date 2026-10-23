#!/usr/bin/env python3
"""Prepare fixed calibration cases and ONNX reference outputs for board checks."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


PARTS = {
    "encoder_core": ["cat_tokens"],
    "fm_decoder_part0": ["t", "x", "text_condition", "speech_condition", "guidance_scale", "padding_mask"],
    "fm_decoder_part1": ["decoder_hidden_p0", "time_hidden", "padding_mask"],
    "fm_decoder_part2": ["decoder_hidden_p1", "time_hidden", "padding_mask"],
    "fm_decoder_part3": ["decoder_hidden_p2", "time_hidden", "padding_mask"],
}


def load_sample(archive: Path, index: int) -> np.ndarray:
    with tarfile.open(archive, "r:gz") as tar:
        wanted = f"{index:05d}.npy"
        member = next((item for item in tar.getmembers() if Path(item.name).name == wanted), None)
        if member is None:
            raise KeyError(f"filename {wanted!r} not found in {archive}")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Cannot read {member.name} from {archive}")
        from io import BytesIO

        return np.load(BytesIO(extracted.read()), allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/decoder4_static_distill_1024_ax630c")
    parser.add_argument("--calib-dir", default="model_convert/calib_data_decoder4_rich_distill_1024")
    parser.add_argument("--output-dir", default="model_convert/axmodel_compare_distill_rich_ax630c")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    calib_dir = Path(args.calib_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"model_dir": str(model_dir), "calib_dir": str(calib_dir), "samples": args.samples, "parts": {}}
    for part, inputs in PARTS.items():
        session = ort.InferenceSession(str(model_dir / f"{part}.onnx"), providers=["CPUExecutionProvider"])
        part_dir = output_dir / part
        part_dir.mkdir(exist_ok=True)
        input_archives = {name: calib_dir / part / f"{name}.tar.gz" for name in inputs}
        for name, archive in input_archives.items():
            if not archive.exists():
                raise FileNotFoundError(archive)

        output_names = [item.name for item in session.get_outputs()]
        manifest["parts"][part] = {"inputs": inputs, "outputs": output_names}
        for index in range(args.samples):
            feed = {name: load_sample(archive, index) for name, archive in input_archives.items()}
            # ONNX uses int64 token ids; AXEngine accepts the copied int32 input.
            if part == "encoder_core":
                feed["cat_tokens"] = feed["cat_tokens"].astype(np.int64)
            outputs = session.run(None, feed)
            case_dir = part_dir / f"{index:05d}"
            case_dir.mkdir(exist_ok=True)
            for name, value in feed.items():
                np.save(case_dir / f"input_{name}.npy", value)
            for name, value in zip(output_names, outputs):
                np.save(case_dir / f"ref_{name}.npy", np.asarray(value))

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"prepared {args.samples} cases for {len(PARTS)} models under {output_dir}")


if __name__ == "__main__":
    main()
