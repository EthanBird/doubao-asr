# -*- coding: utf-8 -*-
"""
协议级实时推流识别会话 (`DoubaoStreamSession`)：
支持不断向 `feed(pcm_chunk)` 推入任意长度的 16kHz/16bit/单声道 PCM 字节流，
内部通过 `WSSConnection` 实时发送 Opus 帧并读取服务端增量响应，实时产出 `partial` 与 `final` 事件。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

from .audio import OpusFrameEncoder, compute_rms
from .codec import build_request, compute_delta, pack_ws_frame, parse_response
from .models import Segment, StreamEvent, TranscriptResult
from .transport import WSSConnection, build_extra

if TYPE_CHECKING:
    from .client import DoubaoASR


class DoubaoStreamSession:
    """
    协议级实时流式语音识别会话（Push Stream，复用单条 WebSocket 长连接）
    """

    def __init__(
        self,
        client: "DoubaoASR",
        *,
        chunk_sec: float = 0.8,
        min_utterance_sec: float = 5.0,
        max_utterance_sec: float = 24.0,
        vad_silence_rms: float = 220.0,
        vad_silence_sec: float = 0.5,
        frame_ms: int = 10,
        opus_bitrate: Optional[int] = None,
        hotwords: Optional[List[str]] = None,
        context: str = "",
        app_name: str = "com.android.chrome",
        organize: bool = False,
        source_name: str = "<stream>",
    ):
        self.client = client
        self.chunk_sec = max(0.1, float(chunk_sec))
        self.min_utterance_sec = max(1.5, float(min_utterance_sec))
        self.max_utterance_sec = max(self.min_utterance_sec + 1.0, float(max_utterance_sec))
        self.vad_silence_rms = float(vad_silence_rms)
        self.vad_silence_sec = float(vad_silence_sec)
        self.frame_ms = int(frame_ms)
        self.opus_bitrate = opus_bitrate
        self.hotwords = hotwords
        self.context = context
        self.app_name = app_name
        self.organize = organize
        self.source_name = source_name

        frame_bytes = (16000 * self.frame_ms // 1000) * 2
        raw_chunk = int(round(self.chunk_sec * 16000)) * 2
        self._chunk_bytes = max(frame_bytes, (raw_chunk // frame_bytes) * frame_bytes)

        self._pcm_buf = bytearray()
        self._total_pcm_bytes = 0
        self._t0 = time.perf_counter()

        self.segments: List[Segment] = []
        self._seg_index = 1
        self._utt_start_sec = 0.0
        self._utt_pcm_bytes = 0
        self._utt_pcm_data = bytearray()
        self._utt_silence_sec = 0.0
        self._utt_rms_sum = 0.0
        self._utt_rms_cnt = 0
        self._utt_failed = False

        self._conn: Optional[WSSConnection] = None
        self._rid: Optional[str] = None
        self._encoder: Optional[OpusFrameEncoder] = None
        self._frame_seq = 0
        self._seq = 1
        self._t0_ms = 0
        self._finals: Dict[int, str] = {}
        self._interims: Dict[int, str] = {}
        self._confidences: List[float] = []
        self._last_partial_text = ""

    def __enter__(self) -> "DoubaoStreamSession":
        self.client.get_device()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            for _ in self.flush():
                pass
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _reset_utterance(self, next_start_sec: float):
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
        self._rid = None
        self._frame_seq = 0
        self._seq = 1
        self._utt_start_sec = next_start_sec
        self._utt_pcm_bytes = 0
        self._utt_pcm_data.clear()
        self._utt_silence_sec = 0.0
        self._utt_rms_sum = 0.0
        self._utt_rms_cnt = 0
        self._utt_failed = False
        self._finals.clear()
        self._interims.clear()
        self._confidences.clear()
        self._last_partial_text = ""

    def _ensure_utterance_started(self):
        if self._rid is not None:
            return
        dev = self.client.get_device()
        if self._conn is None or self._conn.closed:
            self._conn = self.client.transport.ws_pool.acquire(dev["device_id"], dev["token"])
        self._rid = str(uuid.uuid4())
        self._encoder = OpusFrameEncoder(16000, frame_ms=self.frame_ms, bitrate=self.opus_bitrate)
        self._frame_seq = 0
        self._seq = 1
        self._t0_ms = int(time.time() * 1000)
        self._finals.clear()
        self._interims.clear()
        self._confidences.clear()
        self._last_partial_text = ""

        extra = build_extra(
            dev,
            app_name=self.app_name,
            context=self.context,
            hotwords=self.hotwords,
        )
        session_cfg = {
            "audio_info": {"channel": 1, "format": "speech_opus", "sample_rate": 16000},
            "enable_punctuation": True,
            "enable_speech_rejection": False,
            "extra": extra,
        }
        self._conn.send_binary(
            build_request(
                token=dev["token"],
                method_name="StartSession",
                request_id=self._rid,
                payload=json.dumps(session_cfg, ensure_ascii=False),
            )
        )
        r = parse_response(self._conn.recv_message(timeout=10.0))
        if r["message_type"] in ("SessionFailed", "TaskFailed"):
            self._conn.close()
            self._conn = None
            raise RuntimeError(f"StartSession 失败: {r['status_code']} {r['status_message']}")

    @property
    def _history_text(self) -> str:
        return "".join(s.text for s in self.segments if s.text.strip())

    def _current_utterance_text(self) -> str:
        merged = dict(self._interims)
        merged.update(self._finals)
        return "".join(merged[k] for k in sorted(merged))

    def _consume_pb(self, raw_pb: bytes) -> bool:
        pb = parse_response(raw_pb)
        if pb["result_json"]:
            try:
                j = json.loads(pb["result_json"])
            except json.JSONDecodeError:
                j = {}
            for res in j.get("results") or []:
                txt = (res.get("text") or "").strip()
                if not txt:
                    continue
                idx = int(res.get("index") or 0)
                if res.get("is_interim") is False or res.get("is_vad_finished"):
                    self._finals[idx] = txt
                    self._interims.pop(idx, None)
                    if res.get("confidence"):
                        self._confidences.append(float(res["confidence"]))
                else:
                    self._interims[idx] = txt
        if pb["message_type"] == "SessionFinished":
            return True
        if pb["message_type"] in ("TaskFailed", "SessionFailed"):
            self._utt_failed = True
            return True
        return False

    def _send_chunk(self, pcm_chunk: bytes) -> Optional[StreamEvent]:
        rms = compute_rms(pcm_chunk)
        chunk_dur = len(pcm_chunk) / 32000.0
        self._utt_pcm_bytes += len(pcm_chunk)
        self._utt_pcm_data.extend(pcm_chunk)
        self._utt_rms_sum += rms
        self._utt_rms_cnt += 1

        if rms < self.vad_silence_rms:
            self._utt_silence_sec += chunk_dur
        else:
            self._utt_silence_sec = 0.0

        if not self._utt_failed:
            self._ensure_utterance_started()
            assert self._encoder is not None and self._conn is not None and self._rid is not None
            frames = self._encoder.encode_frames(pcm_chunk, pad_final=True)
            for fr in frames:
                state = 1 if self._frame_seq == 0 else 3
                req_pb = build_request(
                    method_name="TaskRequest",
                    request_id=self._rid,
                    audio=fr,
                    frame_state=state,
                    payload=json.dumps(
                        {"extra": {}, "timestamp_ms": self._t0_ms + self._frame_seq * self.frame_ms}
                    ),
                )
                self._frame_seq += 1
                self._conn.send_binary(req_pb)

            # 通过 select 等待至多 50ms 并读取已到达的所有服务端增量响应（不触发 SSLSocket 读超时）
            for raw_msg in self._conn.drain_messages(initial_wait_sec=0.05):
                self._consume_pb(raw_msg)

        cur_seq = self._seq
        self._seq += 1

        t = self._current_utterance_text()
        if t and t != self._last_partial_text:
            delta = compute_delta(self._last_partial_text, t)
            self._last_partial_text = t
            cur_end = self._utt_start_sec + (self._utt_pcm_bytes / 32000.0)
            return StreamEvent(
                type="partial",
                segment_index=self._seg_index,
                seq=cur_seq,
                start=round(self._utt_start_sec, 2),
                end=round(cur_end, 2),
                text=t,
                delta=delta,
                full_text=self._history_text + t,
                is_final=False,
            )
        return None

    def _finalize_utterance(self) -> Optional[StreamEvent]:
        if self._rid is None or self._utt_pcm_bytes == 0 or self._conn is None or self._encoder is None:
            return None

        dev = self.client.get_device()
        if not self._utt_failed:
            # 发送尾帧标记 (state=9, force_asr_twopass=True, finish_audio=True) + FinishSession
            silent_tail = self._encoder.encode_frames(b"\x00" * self._encoder.frame_bytes, pad_final=True)
            last_fr = silent_tail[0] if silent_tail else b""
            req_last = build_request(
                method_name="TaskRequest",
                request_id=self._rid,
                audio=last_fr,
                frame_state=9,
                payload=json.dumps(
                    {
                        "extra": {"force_asr_twopass": True, "finish_audio": True},
                        "timestamp_ms": self._t0_ms + self._frame_seq * self.frame_ms,
                    }
                ),
            )
            try:
                self._conn.send_binary(req_last)
                time.sleep(0.02)
                self._conn.send_binary(
                    build_request(token=dev["token"], method_name="FinishSession", request_id=self._rid)
                )

                deadline = time.time() + 12.0
                while time.time() < deadline:
                    raw_msg = self._conn.recv_message(timeout=max(0.1, deadline - time.time()))
                    if self._consume_pb(raw_msg):
                        break
            except Exception:
                self._utt_failed = True

        best = "".join(self._finals[k] for k in sorted(self._finals)) or self._current_utterance_text()
        avg_rms = self._utt_rms_sum / max(1, self._utt_rms_cnt)
        if not best and self._utt_failed and avg_rms >= 15.0 and self._utt_pcm_data:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            best, conf_fallback = self.client.transcribe_pcm_segment(
                bytes(self._utt_pcm_data),
                frame_ms=self.frame_ms,
                bitrate=self.opus_bitrate,
                hotwords=self.hotwords,
                context=self.context,
                app_name=self.app_name,
            )
            if conf_fallback:
                self._confidences.append(conf_fallback)
        avg_conf = sum(self._confidences) / len(self._confidences) if self._confidences else 0.0
        cur_end = self._utt_start_sec + (self._utt_pcm_bytes / 32000.0)

        delta = compute_delta(self._last_partial_text, best)
        seg = Segment(
            index=self._seg_index,
            start=round(self._utt_start_sec, 2),
            end=round(cur_end, 2),
            text=best,
            rms=round(avg_rms, 1),
            confidence=round(avg_conf, 4),
        )
        self.segments.append(seg)
        ev = StreamEvent(
            type="final",
            segment_index=self._seg_index,
            seq=self._seq,
            start=seg.start,
            end=seg.end,
            text=best,
            delta=delta,
            full_text=self._history_text,
            is_final=True,
        )
        self._seg_index += 1
        self._reset_utterance(cur_end)
        return ev

    def feed(self, pcm_bytes: bytes) -> Iterator[StreamEvent]:
        """向流式会话推入任意长度的 16kHz/16bit/mono PCM 字节块"""
        if not pcm_bytes:
            return
        self._pcm_buf.extend(pcm_bytes)
        self._total_pcm_bytes += len(pcm_bytes)

        while len(self._pcm_buf) >= self._chunk_bytes:
            chunk = bytes(self._pcm_buf[: self._chunk_bytes])
            del self._pcm_buf[: self._chunk_bytes]

            ev = self._send_chunk(chunk)
            if ev is not None:
                yield ev

            utt_dur = self._utt_pcm_bytes / 32000.0
            should_cut = (
                utt_dur >= self.min_utterance_sec and self._utt_silence_sec >= self.vad_silence_sec
            ) or (utt_dur >= self.max_utterance_sec)
            if should_cut:
                fin_ev = self._finalize_utterance()
                if fin_ev is not None:
                    yield fin_ev

    def flush(self) -> Iterator[StreamEvent]:
        """冲刷缓冲区剩余不足一个 chunk 的尾帧音频并完成最终定稿"""
        if self._pcm_buf:
            rem = bytes(self._pcm_buf)
            self._pcm_buf.clear()
            ev = self._send_chunk(rem)
            if ev is not None:
                yield ev

        if self._utt_pcm_bytes > 0:
            fin_ev = self._finalize_utterance()
            if fin_ev is not None:
                yield fin_ev

    @property
    def result(self) -> TranscriptResult:
        raw_full = "".join(s.text for s in self.segments if s.text.strip())
        org = self.client.organize_text(raw_full) if (self.organize and raw_full) else ""
        return TranscriptResult(
            segments=list(self.segments),
            duration=round(self._total_pcm_bytes / 32000.0, 2),
            source=self.source_name,
            elapsed_sec=round(time.perf_counter() - self._t0, 2),
            organized_text=org,
        )
