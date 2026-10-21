# -*- coding: utf-8 -*-
"""
命令行工具入口 (`python -m doubao_asr` / `doubao-asr`)
"""

from __future__ import annotations

import argparse
import json
import sys

from .client import DoubaoASR
from .models import Segment


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="doubao-asr",
        description="Doubao ASR — 零依赖高性能豆包输入法长语音/音视频转文字 + LLM 智能整理工具",
    )
    parser.add_argument("input", nargs="*", help="输入音频或视频文件路径 (wav/mp3/m4a/mp4/m4s/flac...)")
    parser.add_argument("-o", "--output", help="输出文件路径 (可选，如 out.srt / out.md / out.txt)")
    parser.add_argument(
        "-f",
        "--format",
        choices=["txt", "srt", "vtt", "md", "json"],
        default=None,
        help="输出格式: txt | srt | vtt | md | json",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="并发识别线程数 (默认 8，双 AppKey 负载均衡，仅在非 --stream 批量模式下生效)",
    )
    parser.add_argument("--stream", action="store_true", help="启用协议级真流式模式（实时显示增量识别）")
    parser.add_argument("--text", default="", help="纯后处理模式：仅将该段文字交给官方 LLM「智能整理」")
    parser.add_argument("--organize", action="store_true", help="在 ASR 转写后执行官方 LLM「智能整理」")
    parser.add_argument("--raw", action="store_true", help="只输出 ASR 原文，不执行 LLM「智能整理」")
    parser.add_argument("--hotwords", default="", help="逗号分隔热词（写入 asr_params.brand_words）")
    parser.add_argument("--context", default="", help="上下文文本（辅助专有名词识别）")
    parser.add_argument("--app", default="com.android.chrome", help="app_name（作用到的应用包名）")
    parser.add_argument("--frame-ms", type=int, default=10, choices=[10, 20, 40, 60], help="Opus 帧长 ms (默认 10)")
    parser.add_argument("--no-split", action="store_true", help="长音频不按静音分段")
    parser.add_argument("--fresh", action="store_true", help="强制重新注册新匿名设备")
    parser.add_argument("--quiet", action="store_true", help="静默模式，不打印中间进度")
    args = parser.parse_args()

    with DoubaoASR() as asr:
        if args.text:
            out = asr.organize_text(args.text)
            if args.format == "json":
                print(json.dumps({"organize": out}, ensure_ascii=False, indent=2))
            else:
                print(out)
            return

        if not args.input:
            parser.error("请提供音频/视频文件，或使用 --text 走纯智能整理模式")

        hot = [w.strip() for w in args.hotwords.split(",") if w.strip()] if args.hotwords else None
        do_organize = bool(args.organize) and not args.raw

        for src in args.input:
            if args.stream:
                asr.get_device(force=args.fresh)
                with asr.create_stream(
                    chunk_sec=0.8,
                    frame_ms=args.frame_ms,
                    hotwords=hot,
                    context=args.context,
                    app_name=args.app,
                    organize=do_organize,
                    source_name=src,
                ) as stream:
                    for pcm_chunk in asr.iter_pcm16k_chunks(src, chunk_bytes=int(0.8 * 32000)):
                        for ev in stream.feed(pcm_chunk):
                            if args.quiet:
                                continue
                            if ev.is_final:
                                print(
                                    f"\r[定稿 #{ev.segment_index:02d}] [{ev.timestamp}] {ev.text}"
                                    + " " * 15,
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\r[流式 #{ev.segment_index:02d}] [{ev.timestamp}] {ev.text} ...",
                                    end="",
                                    flush=True,
                                )
                    for ev in stream.flush():
                        if not args.quiet and ev.is_final:
                            print(
                                f"\r[定稿 #{ev.segment_index:02d}] [{ev.timestamp}] {ev.text}"
                                + " " * 15,
                                flush=True,
                            )
                res = stream.result
            else:
                def _printer(seg: Segment):
                    if not args.quiet:
                        txt = seg.text if seg.text else "(静音/BGM)"
                        print(f"[{seg.index:02d}] [{seg.timestamp}] {txt}", flush=True)

                res = asr.transcribe(
                    src,
                    fresh_device=args.fresh,
                    split=not args.no_split,
                    workers=args.workers,
                    frame_ms=args.frame_ms,
                    hotwords=hot,
                    context=args.context,
                    app_name=args.app,
                    organize=do_organize,
                    on_segment=_printer,
                )

            if args.output and len(args.input) == 1:
                saved = res.save(args.output, fmt=args.format)
                if not args.quiet:
                    print(
                        f"\n[✓] 已保存至: {saved} (音频 {res.duration:.1f}s, 耗时 {res.elapsed_sec:.2f}s, "
                        f"加速比 {res.duration / max(0.001, res.elapsed_sec):.1f}x)"
                    )
            elif not args.stream or args.format:
                fmt = args.format or "txt"
                if not args.quiet:
                    print("=" * 60)
                if fmt == "srt":
                    print(res.to_srt())
                elif fmt == "vtt":
                    print(res.to_vtt())
                elif fmt == "json":
                    print(res.to_json())
                elif fmt == "md":
                    print(res.to_markdown())
                else:
                    print(res.text)


if __name__ == "__main__":
    main()
