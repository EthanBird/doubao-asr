# -*- coding: utf-8 -*-
"""
零依赖编解码模块：
1. 豆包输入法 Ocean ASR Protobuf Wire 编码 (`build_request`) 与解码 (`parse_response`)
2. RFC 6455 WebSocket 零开销客户端掩码帧打包 (`pack_ws_frame`)
3. 流式增量文本差分计算器 (`compute_delta`)
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Tuple, Union


def varint(n: int) -> bytes:
    if n < 0:
        n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def fld(num: int, data: Union[int, str, bytes], wire: int = 2) -> bytes:
    if wire == 0:
        return varint(num << 3) + varint(int(data))
    if isinstance(data, str):
        data = data.encode("utf-8")
    return varint((num << 3) | 2) + varint(len(data)) + data


def build_request(
    *,
    token: str = "",
    service_name: str = "ASR",
    method_name: str,
    payload: str = "",
    audio: bytes = b"",
    request_id: str = "",
    frame_state: int = 0,
) -> bytes:
    """
    构造豆包输入法 Ocean ASR 请求 Protobuf 帧：
      field 2: token (app_key)
      field 3: service_name ("ASR")
      field 5: method_name ("StartTask" | "StartSession" | "TaskRequest" | "FinishSession")
      field 6: payload (JSON 字符串)
      field 7: audio (单帧 Opus 裸字节)
      field 8: request_id (UUID)
      field 9: frame_state (1=FIRST, 3=MIDDLE, 9=LAST)
    """
    out = bytearray()
    if token:
        out.extend(fld(2, token))
    if service_name:
        out.extend(fld(3, service_name))
    out.extend(fld(5, method_name))
    if payload:
        out.extend(fld(6, payload))
    if audio:
        out.extend(fld(7, audio))
    if request_id:
        out.extend(fld(8, request_id))
    if frame_state:
        out.extend(fld(9, frame_state, 0))
    return bytes(out)


def read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    shift = 0
    val = 0
    n = len(buf)
    while i < n:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7
    return val, i


def parse_response(buf: bytes) -> Dict[str, Any]:
    """
    解析豆包输入法 AsrResponse Protobuf 帧：
      field 1: request_id (string)
      field 2: task_id (string)
      field 3: service_name (string)
      field 4: message_type (string: TaskStarted / SessionStarted / SessionFinished / TaskFailed / SessionFailed)
      field 5: status_code (varint: 20000000 = OK)
      field 6: status_message (string)
      field 7: result_json (string)
    """
    d: Dict[int, Any] = {}
    i = 0
    n = len(buf)
    while i < n:
        try:
            key, i = read_varint(buf, i)
            num, wire = key >> 3, key & 7
            if wire == 0:
                v, i = read_varint(buf, i)
                d[num] = v
            elif wire == 2:
                ln, i = read_varint(buf, i)
                d[num] = buf[i : i + ln]
                i += ln
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            else:
                break
        except Exception:
            break

    def _str(k: int) -> str:
        val = d.get(k)
        if isinstance(val, bytes):
            return val.decode("utf-8", "ignore")
        return str(val) if val is not None else ""

    return {
        "request_id": _str(1),
        "task_id": _str(2),
        "service_name": _str(3),
        "message_type": _str(4),
        "status_code": int(d.get(5, 0)),
        "status_message": _str(6),
        "result_json": _str(7),
    }


def pack_ws_frame(payload: bytes, opcode: int = 2) -> bytes:
    """
    打包 RFC 6455 客户端到服务端 WebSocket 帧（FIN=1, MASK=1）。
    采用 4 字节全零 Masking-Key (b'\\x00\\x00\\x00\\x00')，免除逐字节 XOR 运算开销。
    """
    n = len(payload)
    if n < 126:
        return struct.pack("!BB4s", 0x80 | opcode, 0x80 | n, b"\x00\x00\x00\x00") + payload
    if n < 65536:
        return struct.pack("!BBH4s", 0x80 | opcode, 0x80 | 126, n, b"\x00\x00\x00\x00") + payload
    return struct.pack("!BBQ4s", 0x80 | opcode, 0x80 | 127, n, b"\x00\x00\x00\x00") + payload


def compute_delta(prev_text: str, curr_text: str) -> str:
    """
    计算流式识别相邻两帧文本之间的新增尾部内容 (delta)。
    即使服务端修正了前文中间的标点或同音字，也能准确提取尾部新增文字。
    """
    if not curr_text:
        return ""
    if not prev_text:
        return curr_text
    if curr_text.startswith(prev_text):
        return curr_text[len(prev_text) :]

    # 寻找最长公共前缀（忽略标点修正后的对齐）
    common = 0
    limit = min(len(prev_text), len(curr_text))
    while common < limit and prev_text[common] == curr_text[common]:
        common += 1

    # 若仅末尾标点被改写，从最后一个共同非标点字符之后截取
    punc = set("，。！？、；：,.!?;: ")
    prev_clean = [c for c in prev_text if c not in punc]
    curr_clean = [c for c in curr_text if c not in punc]
    if len(curr_clean) >= len(prev_clean) and prev_clean:
        tail_char = prev_clean[-1]
        idx = curr_text.rfind(tail_char)
        if idx != -1 and idx + 1 <= len(curr_text):
            return curr_text[idx + 1 :]

    return curr_text[common:]
