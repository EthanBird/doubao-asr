# -*- coding: utf-8 -*-
"""
数据结构定义：Segment（定稿分段）、StreamEvent（流式事件）、TranscriptResult（完整转写结果）
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


def fmt_srt_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    s, ms = divmod(total_ms, 1000)
    hrs, rem = divmod(s, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def fmt_vtt_time(seconds: float) -> str:
    return fmt_srt_time(seconds).replace(",", ".")


def fmt_short_time(seconds: float) -> str:
    mins, secs = divmod(max(0.0, seconds), 60)
    return f"{int(mins):02d}:{secs:04.1f}"


@dataclass
class Segment:
    """单个语音切分片段的定稿识别结果"""
    index: int
    start: float
    end: float
    text: str
    rms: float = 0.0
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    @property
    def timestamp(self) -> str:
        return f"{fmt_short_time(self.start)} - {fmt_short_time(self.end)}"


@dataclass
class StreamEvent:
    """
    流式识别事件对象
    - type: 'partial' (增量中间结果) | 'final' (句末标点定稿结果)
    - segment_index: 当前语句序号 (从 1 开始)
    - seq: 当前语句内的帧/包序号
    - start: 当前语句起始秒数
    - end: 当前已送入音频结束秒数
    - text: 当前语句完整识别文本
    - delta: 相较于上一帧新增的尾部文本
    - full_text: 历史已定稿语句 + 当前语句的完整文本拼接
    - is_final: 是否为当前语句最终定稿
    """
    type: str
    segment_index: int
    seq: int
    start: float
    end: float
    text: str
    delta: str
    full_text: str
    is_final: bool

    @property
    def timestamp(self) -> str:
        return f"{fmt_short_time(self.start)} - {fmt_short_time(self.end)}"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TranscriptResult:
    """完整转写结果对象，支持原文与官方 LLM 智能整理成稿，一键导出 Text / SRT / VTT / JSON / Markdown"""
    segments: List[Segment] = field(default_factory=list)
    duration: float = 0.0
    source: str = ""
    elapsed_sec: float = 0.0
    organized_text: str = ""

    @property
    def raw_text(self) -> str:
        return "".join(s.text for s in self.segments if s.text.strip())

    @property
    def text(self) -> str:
        """若启用了官方 LLM 智能整理则返回整理后成稿，否则返回 ASR 原文"""
        return self.organized_text if self.organized_text.strip() else self.raw_text

    @property
    def lines(self) -> List[str]:
        return [s.text for s in self.segments if s.text.strip()]

    def to_srt(self) -> str:
        blocks = []
        idx = 1
        for s in self.segments:
            t = s.text.strip()
            if not t:
                continue
            blocks.append(
                f"{idx}\n{fmt_srt_time(s.start)} --> {fmt_srt_time(s.end)}\n{t}\n"
            )
            idx += 1
        return "\n".join(blocks)

    def to_vtt(self) -> str:
        blocks = ["WEBVTT\n"]
        for s in self.segments:
            t = s.text.strip()
            if not t:
                continue
            blocks.append(
                f"{fmt_vtt_time(s.start)} --> {fmt_vtt_time(s.end)}\n{t}\n"
            )
        return "\n".join(blocks)

    def to_markdown(self, title: str = "豆包语音转文字稿") -> str:
        md = [
            f"# {title}\n",
            f"- **音频来源**: `{self.source or 'PCM Stream'}`",
            f"- **音频时长**: `{self.duration:.1f}s`",
            f"- **转写耗时**: `{self.elapsed_sec:.2f}s`",
            f"- **实时倍率**: `{self.duration / max(0.001, self.elapsed_sec):.1f}x`",
            f"- **有效片段**: `{len(self.lines)} / {len(self.segments)}` 段\n",
        ]
        if self.organized_text.strip():
            md.extend([
                "## 一、 智能整理成稿\n",
                self.organized_text.strip(),
                "\n## 二、 ASR 原始识别文本\n",
                self.raw_text or "*(无识别内容)*",
                "\n## 三、 分段时间轴明细\n",
            ])
        else:
            md.extend([
                "## 一、 完整文字稿\n",
                self.raw_text or "*(无识别内容)*",
                "\n## 二、 分段时间轴明细\n",
            ])
        md.extend([
            "| 序号 | 时间轴 | 识别文本 |",
            "| :---: | :---: | :--- |",
        ])
        for s in self.segments:
            txt = s.text.strip() if s.text.strip() else "*(静音/背景音)*"
            md.append(f"| {s.index:02d} | `{s.timestamp}` | {txt} |")
        return "\n".join(md) + "\n"

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "duration": round(self.duration, 2),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "text": self.text,
            "raw_text": self.raw_text,
            "organized_text": self.organized_text,
            "segments": [asdict(s) for s in self.segments],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str, fmt: Optional[str] = None) -> str:
        if not fmt:
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            fmt = ext if ext in ("txt", "srt", "vtt", "json", "md") else "txt"
        fmt = fmt.lower()
        if fmt == "srt":
            content = self.to_srt()
        elif fmt == "vtt":
            content = self.to_vtt()
        elif fmt == "json":
            content = self.to_json()
        elif fmt in ("md", "markdown"):
            content = self.to_markdown()
        else:
            content = self.text + "\n"
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
