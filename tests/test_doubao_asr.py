# -*- coding: utf-8 -*-
"""
doubao-asr 单元测试与端到端实测套件（验证零第三方依赖、Protobuf/WebSocket 编解码、Opus 帧编码、流式接口、LLM 智能整理与多线程并发长音频性能）
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from doubao_asr import (
    DoubaoASR,
    OpusFrameEncoder,
    compute_rms,
    encode_pcm_to_opus_frames,
    organize_text,
    split_pcm_by_silence,
    transcribe,
    transcribe_stream,
)
from doubao_asr.audio import _ffmpeg_encode_opus_frames
from doubao_asr.codec import build_request, compute_delta, fld, pack_ws_frame, parse_response


class TestDoubaoASRUnit(unittest.TestCase):
    def test_protobuf_and_ws_frame(self):
        req = build_request(
            token="RTIHIRzbwS",
            service_name="ASR",
            method_name="TaskRequest",
            payload='{"extra":{},"timestamp_ms":123}',
            audio=b"\x01\x02\x03\x04",
            request_id="req-123",
            frame_state=3,
        )
        self.assertGreater(len(req), 20)

        # 构造模拟 AsrResponse 并验证 parse_response
        fake_resp = (
            fld(1, "req-123")
            + fld(2, "task-456")
            + fld(3, "ASR")
            + fld(4, "SessionFinished")
            + fld(5, 20000000, wire=0)
            + fld(6, "OK")
            + fld(7, '{"results":[{"text":"测试文本","is_interim":false,"index":0}]}')
        )
        parsed = parse_response(fake_resp)
        self.assertEqual(parsed["request_id"], "req-123")
        self.assertEqual(parsed["task_id"], "task-456")
        self.assertEqual(parsed["message_type"], "SessionFinished")
        self.assertEqual(parsed["status_code"], 20000000)
        self.assertIn("测试文本", parsed["result_json"])

        ws_bin = pack_ws_frame(b"hello", opcode=2)
        self.assertEqual(ws_bin[:2], bytes([0x82, 0x85]))
        self.assertEqual(ws_bin[-5:], b"hello")

    def test_opus_encoder_and_vad(self):
        pcm = b"\x10\x01" * 16000  # 1s 16kHz 16-bit mono
        frames_10ms = encode_pcm_to_opus_frames(pcm, sr=16000, frame_ms=10)
        self.assertEqual(len(frames_10ms), 100)
        self.assertTrue(all(len(f) > 2 for f in frames_10ms))

        frames_20ms = encode_pcm_to_opus_frames(pcm, sr=16000, frame_ms=20)
        self.assertEqual(len(frames_20ms), 50)

        # 验证 ffmpeg Ogg 解复用兜底同样产出 ~100 帧 (含 Ogg 尾帧为 100~101 帧)
        ff_frames = _ffmpeg_encode_opus_frames(pcm, sr=16000, frame_ms=10)
        self.assertIn(len(ff_frames), (100, 101))
        self.assertGreater(compute_rms(pcm), 0)

    def test_compute_delta(self):
        self.assertEqual(compute_delta("欢迎。家来体验", "欢迎大家来体验达摩院"), "达摩院")

    def test_device_and_organize(self):
        with DoubaoASR() as asr:
            dev = asr.get_device()
            self.assertTrue(dev.get("device_id"))
            org = asr.organize_text("今天天气真不错我们下午三点半去开个会讨论一下deepseek和claude code的集成方案吧")
            self.assertIn("今天天气真不错", org)
            self.assertIn("下午", org)

    def test_live_asr_and_streaming(self):
        sample_wav = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example_zh.wav")
        )
        if not os.path.isfile(sample_wav):
            self.skipTest("Sample wav not found")

        # 1. 测试流式识别
        events = list(transcribe_stream(sample_wav, chunk_sec=0.8))
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(events[-1].is_final)
        self.assertIn("达摩院", events[-1].text)

        # 2. 测试并发长音频切分转写性能（将两段示例音频拼接为 ~16s 多句长音频测试并发 4 线程 + LLM 智能整理）
        sample_wav2 = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "examples", "asr_example.wav")
        )
        with DoubaoASR() as asr:
            pcm1 = asr.load_pcm16k(sample_wav)
            pcm2 = asr.load_pcm16k(sample_wav2)
            silence_1s = b"\x00\x00" * 16000
            long_pcm = pcm1 + silence_1s + pcm2 + silence_1s + pcm1

            t0 = time.perf_counter()
            res = asr.transcribe(
                long_pcm,
                workers=4,
                min_segment_sec=4.0,
                max_segment_sec=7.0,
                organize=True,
            )
            elapsed = time.perf_counter() - t0
            print(
                f"\n[Benchmark] {res.duration:.1f}s 音频 ({len(res.segments)} 分片) 4 线程并发转写+智能整理总耗时: {elapsed:.2f}s "
                f"({res.duration / elapsed:.1f}x 实时)"
            )
            print(f"[Benchmark] ASR 原文: {res.raw_text}")
            print(f"[Benchmark] 智能整理: {res.organized_text}")
            self.assertIn("达摩院", res.raw_text)
            self.assertTrue(len(res.to_srt()) > 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
