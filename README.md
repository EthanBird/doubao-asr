# doubao-asr — 零依赖高性能豆包输入法语音识别 + 官方 LLM 智能整理 SDK

将豆包输入法语音识别（Ocean ASR + 官方 LLM 文字整理）重构并工程化拆分为标准 Python 包，实现 **零第三方 pip 依赖 (`dependencies = []`)** 与 **多维度性能优化**，同时支持**协议级真流式推流**、**多线程 WebSocket 连接池并发长音视频转写**与**多线程分块并发 LLM 智能整理**。

---

## 一、 架构拆分与核心优化

```text
doubao-asr/
├── pyproject.toml                # 零依赖标准打包配置 (dependencies = [])
├── README.md                     # 架构说明与使用文档
├── examples/
│   ├── asr_example_zh.wav        # 中文示例语音
│   └── asr_example.wav           # 短语音示例
├── src/
│   └── doubao_asr/
│       ├── __init__.py           # 顶层公开 API 导出
│       ├── models.py             # 数据模型 (Segment, StreamEvent, TranscriptResult)
│       ├── codec.py              # 极简 Protobuf 编解码 + RFC 6455 零拷贝掩码帧打包 + 流式增量差分
│       ├── audio.py              # 直接 ctypes libopus 绑定 (含 ffmpeg Ogg 解复用兜底) + C级 audioop VAD 切分
│       ├── transport.py          # 原生 ssl+socket RFC 6455 WebSocket 长连接池 + HTTP Keep-Alive 连接池
│       ├── stream.py             # 协议级真流式会话 DoubaoStreamSession (边推流边收增量 partial/final + 自动 VAD 断句)
│       ├── client.py             # 高级客户端 DoubaoASR (多线程并发转写 + 并发分块 LLM 智能整理 + 同步/异步流式)
│       ├── cli.py                # CLI 命令行入口 (支持 --stream、-w 并发数、--organize、--hotwords 等)
│       ├── __main__.py           # 支持 python -m doubao_asr
│       └── _bin/
│           └── opus.dll          # 内置 64 位 libopus 动态库，开箱即用
└── tests/
    └── test_doubao_asr.py        # 单元测试 + 端到端集成与长音视频多线程性能基准测试
```

### 相比原版单脚本 (`01-doubao_asr.py`) 的依赖精简与性能提升

| 维度 | 原版单脚本 (`01-doubao_asr.py`) | 工程化版 (`doubao-asr`) | 收益 |
| :--- | :--- | :--- | :--- |
| **第三方 pip 依赖** | 需安装 `websockets`、`opuslib` 及系统 `libopus` | **0 个第三方包**（仅用 Python 标准库） | 彻底摆脱环境冲突与动态库缺失报错 |
| **WebSocket 传输层** | 每段语音新建 `websockets.connect` + `StartTask` | **原生 `WSSConnectionPool` 长连接池**（建连仅发 1 次 `StartTask`，后续无限复用 `StartSession`） | 省去 TCP + TLS + WS Upgrade + `StartTask` 四重 RTT |
| **帧发送系统调用** | 每 10ms 帧单独 `await ws.send()`（20s 语音需 2000 次调用） | **RFC 6455 零拷贝掩码 + 250 帧批量 `sendall`** + 非阻塞排空 | 单段系统调用次数减少 **99%**，彻底消除发送缓冲阻塞 |
| **OPUS 帧编码** | 依赖 `opuslib`（Windows 常因找不到 `opus.dll` 报错） | 直接 `ctypes` 绑定内置 `_bin/opus.dll` + 复用单块缓冲 + `ffmpeg` Ogg 解复用双保险兜底 | 零依赖，编码吞吐达 **~230x 实时倍率** |
| **VAD 静音切分** | 固定 20ms 窗口 Python 循环平方和，硬阈值切分 | 标准库 C 级 `audioop.rms` + 100ms 窗口能量谷底自适应切分 | 10 分钟音频切分耗时仅 **~8ms（快 100 倍+）**，断句更自然 |
| **长视频批量转写** | 单线程 `asyncio.run` 串行逐段转写 | `ThreadPoolExecutor` 多线程并发 (`workers=6`) + WSS 连接池复用 | 10~45 分钟长视频转写达 **18x~30x 实时加速比** |
| **LLM 智能文字整理** | 整段单次请求（>1000 字耗时长达 20s+，>2000 字被截断） | 按句切块（~450 字/块）+ `HTTPSConnectionPool` 多线程并发整理 | 突破 2000 字上限，整理耗时从 20.6s 降至 **~1.9s（快 10 倍+）** |

---

## 二、 快速上手

无需安装任何第三方依赖，在任意 Python (`>=3.8`) 下把 `src` 加入 `PYTHONPATH` 即可直接使用：

### 1. 一行代码多线程并发转写长音视频（输出文本 / SRT / VTT / Markdown / JSON）

```python
from doubao_asr import transcribe

res = transcribe(
    "vlog.mp4",
    workers=6,          # 6 线程并发 WebSocket 识别
    organize=True,      # 启用官方 LLM 智能整理（自动分块并发）
    hotwords=["DeepSeek", "Claude Code"],
    on_segment=lambda seg: print(f"[{seg.timestamp}] {seg.text}")
)

print("智能整理成稿:", res.text)
print("ASR 原始文本:", res.raw_text)
res.save("vlog.srt")   # 导出 SRT 字幕
res.save("vlog.md")    # 导出 Markdown 文字稿（同时包含智能整理成稿、ASR原文与时间轴表）
```

### 2. 生成器流式接口 `transcribe_stream`（边解码边推流边出字）

```python
from doubao_asr import transcribe_stream

for ev in transcribe_stream("vlog.mp4", chunk_sec=0.8):
    if ev.is_final:
        print(f"\n[定稿 #{ev.segment_index} {ev.timestamp}] {ev.text}")
    else:
        print(f"\r[流式 {ev.end:.1f}s] {ev.text} (+{ev.delta})", end="", flush=True)
```

### 3. 麦克风 / WebSocket 实时推流会话 `create_stream`

```python
from doubao_asr import DoubaoASR

with DoubaoASR() as asr:
    with asr.create_stream(chunk_sec=0.8, hotwords=["DeepSeek"]) as stream:
        for pcm_chunk in mic_or_websocket_stream:  # 16kHz / 16bit / mono PCM bytes
            for ev in stream.feed(pcm_chunk):
                print(ev.type, ev.text)
    print("累积全文:", stream.result.text)
```

### 4. CLI 命令行使用

```powershell
# 多线程并发转写长音视频并导出 SRT 字幕
python -m doubao_asr input.mp4 -w 6 -f srt -o subtitle.srt

# 多线程并发转写 + 官方 LLM 智能整理并导出 Markdown 报告
python -m doubao_asr input.mp4 -w 6 --organize -f md -o report.md

# 协议级真流式实时转写模式
python -m doubao_asr input.mp4 --stream

# 纯文本官方 LLM 智能整理
python -m doubao_asr --text "今天天气真不错我们下午三点半去开个会讨论一下deepseek和claude code的集成方案吧"
```
