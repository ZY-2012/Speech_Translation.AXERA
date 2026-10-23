#!/usr/bin/env python3
"""ZipVoice runtime for four-part fm_decoder ONNX models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort

from onnx_zipvoice_runtime import StaticOnnxZipVoiceRuntime


class Decoder4StaticOnnxZipVoiceRuntime(StaticOnnxZipVoiceRuntime):
    """Runs encoder_core plus fm_decoder_part0..fm_decoder_part3."""

    def _init_sessions(self, num_threads: int):
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = num_threads
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session_opts = sess_opts

        encoder_path = self.encoder_model_path or (self.model_dir / "encoder_core.onnx")
        part_paths = [self.model_dir / f"fm_decoder_part{index}.onnx" for index in range(4)]
        for path in [encoder_path, *part_paths]:
            if not path.exists():
                raise FileNotFoundError(path)

        self.encoder_path = encoder_path
        self.encoder_session = ort.InferenceSession(
            str(encoder_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.part_sessions = [
            ort.InferenceSession(
                str(path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            for path in part_paths
        ]

        part0_inputs = {value.name: value for value in self.part_sessions[0].get_inputs()}
        if "x" not in part0_inputs:
            raise RuntimeError(f"fm_decoder_part0.onnx must have input named 'x': {part_paths[0]}")
        decoder_x_shape = part0_inputs["x"].shape
        self.decoder_seq_len = (
            decoder_x_shape[1] if isinstance(decoder_x_shape[1], int) else None
        )
        self.decoder_has_padding_mask = "padding_mask" in part0_inputs
        logging.info("Decoder4 ONNX sessions initialized successfully")

    @property
    def sessions(self) -> Dict[str, ort.InferenceSession]:
        sessions = {"encoder_core": self.encoder_session}
        for index, session in enumerate(self.part_sessions):
            sessions[f"fm_decoder_part{index}"] = session
        return sessions

    @staticmethod
    def _outputs_by_name(session: ort.InferenceSession, outputs):
        return {meta.name: value for meta, value in zip(session.get_outputs(), outputs)}

    @staticmethod
    def _feed_required(session: ort.InferenceSession, values: Dict[str, np.ndarray]):
        feed = {}
        model_name = Path(session._model_path).name
        for input_meta in session.get_inputs():
            if input_meta.name not in values:
                raise KeyError(f"missing input '{input_meta.name}' for {model_name}")
            value = values[input_meta.name]
            if input_meta.shape == [] and np.asarray(value).size == 1:
                value = np.asarray(value, dtype=np.asarray(value).dtype).reshape(())
            feed[input_meta.name] = value
        return feed

    def run_decoder(
        self,
        t: np.ndarray,
        x: np.ndarray,
        text_condition: np.ndarray,
        speech_condition: np.ndarray,
        guidance_scale: np.ndarray,
        padding_mask: np.ndarray = None,
    ) -> np.ndarray:
        seq_len = x.shape[1]
        values = {
            "t": t,
            "x": x,
            "text_condition": text_condition,
            "speech_condition": speech_condition,
            "guidance_scale": guidance_scale,
            "padding_mask": padding_mask
            if padding_mask is not None
            else np.zeros((1, seq_len), dtype=np.bool_),
        }

        for session in self.part_sessions:
            feed = self._feed_required(session, values)
            outputs = self._outputs_by_name(session, session.run(None, feed))
            values.update(outputs)

        output_name = self.part_sessions[-1].get_outputs()[0].name
        return values[output_name].astype(np.float32)
