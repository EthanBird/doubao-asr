# -*- coding: utf-8 -*-
"""
doubao_asr — 零依赖高性能豆包输入法语音识别 + 官方 LLM 智能整理 SDK
"""

from .audio import (
    OpusFrameEncoder,
    compute_rms,
    encode_pcm_to_opus_frames,
    iter_pcm16k_chunks,
    load_pcm16k,
    split_pcm_by_silence,
)
from .client import DoubaoASR, organize_text, transcribe, transcribe_stream
from .models import Segment, StreamEvent, TranscriptResult
from .stream import DoubaoStreamSession

__version__ = "1.0.0"
__all__ = [
    "DoubaoASR",
    "DoubaoStreamSession",
    "transcribe",
    "transcribe_stream",
    "organize_text",
    "Segment",
    "StreamEvent",
    "TranscriptResult",
    "OpusFrameEncoder",
    "encode_pcm_to_opus_frames",
    "compute_rms",
    "split_pcm_by_silence",
    "iter_pcm16k_chunks",
    "load_pcm16k",
]
