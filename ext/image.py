"""ASM-Lang extension: image loading (PNG, JPEG, BMP) using stdlib only.

The implementation prefers Windows GDI+ via ``ctypes`` when available, which
gives broad codec coverage without third-party modules. On non-Windows
platforms, a small pure-Python decoder handles non-interlaced 8-bit PNG (RGB
or RGBA) and uncompressed 24/32-bit BMP. JPEG decoding is only available when
GDI+ is present.
"""

from __future__ import annotations

import atexit
import os
import struct
import sys
import zlib
from typing import Any, List, Tuple

import numpy as np

from extensions import ASMExtensionError, ExtensionAPI


ASM_LANG_EXTENSION_NAME = "image"
ASM_LANG_EXTENSION_API_VERSION = 1
ASM_LANG_EXTENSION_ASMODULE = True


def _expect_str(v: Any, rule: str, location: Any) -> str:
    from interpreter import ASMRuntimeError, TYPE_STR

    if getattr(v, "type", None) != TYPE_STR:
        raise ASMRuntimeError(f"{rule} expects STR", location=location, rewrite_rule=rule)
    return str(v.value)


def _check_path(path: str, rule: str, location: Any) -> None:
    from interpreter import ASMRuntimeError

    if not path:
        raise ASMRuntimeError(f"{rule}: path must be non-empty", location=location, rewrite_rule=rule)
    if not os.path.exists(path):
        raise ASMRuntimeError(f"{rule}: file not found", location=location, rewrite_rule=rule)


def _guard_image_size(width: int, height: int, rule: str, location: Any) -> None:
    from interpreter import ASMRuntimeError

    if width <= 0 or height <= 0:
        raise ASMRuntimeError(f"{rule}: invalid image dimensions", location=location, rewrite_rule=rule)
    # Simple safety limit to avoid exhausting memory on crafted inputs.
    if width * height > 100_000_000:
        raise ASMRuntimeError(f"{rule}: image too large", location=location, rewrite_rule=rule)


def _make_tensor_from_pixels(width: int, height: int, pixels: List[int], rule: str, location: Any):
    from interpreter import ASMRuntimeError, TYPE_INT, TYPE_TNS, Tensor, Value

    expected = width * height * 4
    if len(pixels) != expected:
        raise ASMRuntimeError(f"{rule}: pixel buffer has unexpected length", location=location, rewrite_rule=rule)
    data = np.array([Value(TYPE_INT, int(ch)) for ch in pixels], dtype=object)
    shape = [int(height), int(width), 4]  # [row][column][channel]
    return Value(TYPE_TNS, Tensor(shape=shape, data=data))


# ---- Windows GDI+ fast path ----

if sys.platform == "win32":
    import ctypes

    class _GdiplusStartupInput(ctypes.Structure):
        _fields_ = [
            ("GdiplusVersion", ctypes.c_uint),
            ("DebugEventCallback", ctypes.c_void_p),
            ("SuppressBackgroundThread", ctypes.c_bool),
            ("SuppressExternalCodecs", ctypes.c_bool),
        ]

    class _Rect(ctypes.Structure):
        _fields_ = [("X", ctypes.c_int), ("Y", ctypes.c_int), ("Width", ctypes.c_int), ("Height", ctypes.c_int)]

    class _BitmapData(ctypes.Structure):
        _fields_ = [
            ("Width", ctypes.c_uint),
            ("Height", ctypes.c_uint),
            ("Stride", ctypes.c_int),
            ("PixelFormat", ctypes.c_uint),
            ("Scan0", ctypes.c_void_p),
            ("Reserved", ctypes.c_uint),
        ]

    _gdiplus_token = ctypes.c_ulong()
    _gdiplus_ready = False
    _gdiplus_handle: Any = None
    _ImageLockModeRead = 1
    _PixelFormat32bppARGB = 0x26200A

    def _gdiplus_start() -> Any:
        global _gdiplus_ready, _gdiplus_handle
        if _gdiplus_ready and _gdiplus_handle is not None:
            return _gdiplus_handle
        gdiplus = ctypes.windll.gdiplus
        startup = _GdiplusStartupInput(1, None, False, False)
        status = gdiplus.GdiplusStartup(ctypes.byref(_gdiplus_token), ctypes.byref(startup), None)
        if status != 0:
            raise RuntimeError(f"GdiplusStartup failed ({status})")
        _gdiplus_handle = gdiplus
        _gdiplus_ready = True
        atexit.register(_gdiplus_shutdown)
        return gdiplus

    def _gdiplus_shutdown() -> None:
        global _gdiplus_ready, _gdiplus_handle
        if not _gdiplus_ready or _gdiplus_handle is None:
            return
        try:
            _gdiplus_handle.GdiplusShutdown(_gdiplus_token)
        except Exception:
            pass
        _gdiplus_ready = False
        _gdiplus_handle = None

    def _load_with_gdiplus(path: str) -> Tuple[int, int, List[int]]:
        gdiplus = _gdiplus_start()

        img = ctypes.c_void_p()
        status = gdiplus.GdipLoadImageFromFile(ctypes.c_wchar_p(path), ctypes.byref(img))
        if status != 0:
            raise RuntimeError(f"GdipLoadImageFromFile failed ({status})")

        try:
            width = ctypes.c_uint()
            height = ctypes.c_uint()
            gdiplus.GdipGetImageWidth(img, ctypes.byref(width))
            gdiplus.GdipGetImageHeight(img, ctypes.byref(height))

            rect = _Rect(0, 0, int(width.value), int(height.value))
            data = _BitmapData()
            status = gdiplus.GdipBitmapLockBits(
                img,
                ctypes.byref(rect),
                _ImageLockModeRead,
                _PixelFormat32bppARGB,
                ctypes.byref(data),
            )
            if status != 0:
                raise RuntimeError(f"GdipBitmapLockBits failed ({status})")

            try:
                stride = int(data.Stride)
                abs_stride = abs(stride)
                buf_len = abs_stride * rect.Height
                buf = (ctypes.c_ubyte * buf_len).from_address(int(data.Scan0))
                pixels: List[int] = []
                for y in range(rect.Height):
                    row_index = y if stride >= 0 else (rect.Height - 1 - y)
                    base = row_index * abs_stride
                    for x in range(rect.Width):
                        idx = base + x * 4
                        b = buf[idx]
                        g = buf[idx + 1]
                        r = buf[idx + 2]
                        a = buf[idx + 3]
                        pixels.extend((int(r), int(g), int(b), int(a)))
                return rect.Width, rect.Height, pixels
            finally:
                gdiplus.GdipBitmapUnlockBits(img, ctypes.byref(data))
        finally:
            gdiplus.GdipDisposeImage(img)
else:
    _load_with_gdiplus = None  # type: ignore[assignment]


# ---- Pure-Python decoders ----

def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png(path: str) -> Tuple[int, int, List[int]]:
    with open(path, "rb") as handle:
        data = handle.read()

    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Not a PNG file")

    pos = 8
    width = height = None
    bit_depth = None
    color_type = None
    interlace = None
    idat = bytearray()

    while pos + 8 <= len(data):
        length = struct.unpack("!I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        pos += 8
        chunk = data[pos : pos + length]
        pos += length + 4  # skip CRC

        if ctype == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                "!IIBBBBB", chunk
            )
            if compression != 0 or filter_method != 0:
                raise RuntimeError("Unsupported PNG compression or filter method")
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None or interlace is None:
        raise RuntimeError("Malformed PNG: missing IHDR")
    if interlace != 0:
        raise RuntimeError("Interlaced PNG is not supported")
    if bit_depth != 8:
        raise RuntimeError("Only 8-bit PNG is supported")
    if color_type not in (2, 6):
        raise RuntimeError("Unsupported PNG color type")

    bpp = 4 if color_type == 6 else 3
    stride = width * bpp
    raw = zlib.decompress(bytes(idat))
    expected = (stride + 1) * height
    if len(raw) < expected:
        raise RuntimeError("PNG data truncated")

    pixels: List[int] = []
    prev = bytearray(stride)
    idx = 0
    for _ in range(height):
        ftype = raw[idx]
        idx += 1
        row = bytearray(raw[idx : idx + stride])
        idx += stride

        recon = bytearray(stride)
        for i in range(stride):
            left = recon[i - bpp] if i >= bpp else 0
            up = prev[i] if prev else 0
            up_left = prev[i - bpp] if i >= bpp else 0
            if ftype == 0:
                val = row[i]
            elif ftype == 1:
                val = (row[i] + left) & 0xFF
            elif ftype == 2:
                val = (row[i] + up) & 0xFF
            elif ftype == 3:
                val = (row[i] + ((left + up) >> 1)) & 0xFF
            elif ftype == 4:
                val = (row[i] + _paeth(left, up, up_left)) & 0xFF
            else:
                raise RuntimeError(f"Unsupported PNG filter {ftype}")
            recon[i] = val

        prev = recon
        for x in range(width):
            off = x * bpp
            r = recon[off]
            g = recon[off + 1]
            b = recon[off + 2]
            a = recon[off + 3] if bpp == 4 else 255
            pixels.extend((int(r), int(g), int(b), int(a)))

    return width, height, pixels


def _decode_bmp(path: str) -> Tuple[int, int, List[int]]:
    with open(path, "rb") as handle:
        data = handle.read()

    if len(data) < 54 or data[:2] != b"BM":
        raise RuntimeError("Not a BMP file")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    header_size = struct.unpack_from("<I", data, 14)[0]
    if header_size < 40:
        raise RuntimeError("Unsupported BMP header")

    width_raw = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]

    if planes != 1:
        raise RuntimeError("Unsupported BMP planes")
    if compression not in (0,):
        raise RuntimeError("Compressed BMP not supported")
    if bpp not in (24, 32):
        raise RuntimeError("Only 24-bit and 32-bit BMP are supported")

    top_down = height < 0
    h = abs(height)
    w = abs(width_raw)
    row_stride = ((bpp * w + 31) // 32) * 4
    pixels: List[int] = []
    for row in range(h):
        src_row = row if top_down else (h - 1 - row)
        base = pixel_offset + src_row * row_stride
        for col in range(w):
            off = base + col * (bpp // 8)
            if off + 3 > len(data):
                raise RuntimeError("BMP data truncated")
            b = data[off]
            g = data[off + 1]
            r = data[off + 2]
            a = data[off + 3] if bpp == 32 else 255
            pixels.extend((int(r), int(g), int(b), int(a)))

    return w, h, pixels


# ---- Dispatcher ----

def _load_png_file(path: str) -> Tuple[int, int, List[int]]:
    if _load_with_gdiplus is not None:
        try:
            return _load_with_gdiplus(path)
        except Exception:
            pass
    return _decode_png(path)


def _load_jpeg_file(path: str) -> Tuple[int, int, List[int]]:
    if _load_with_gdiplus is None:
        raise RuntimeError("JPEG decoding requires Windows GDI+")
    return _load_with_gdiplus(path)


def _load_bmp_file(path: str) -> Tuple[int, int, List[int]]:
    if _load_with_gdiplus is not None:
        try:
            return _load_with_gdiplus(path)
        except Exception:
            pass
    return _decode_bmp(path)


# ---- Operators ----

def _op_load_png(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    path = _expect_str(args[0], "LOAD_PNG", location)
    _check_path(path, "LOAD_PNG", location)
    try:
        w, h, pixels = _load_png_file(path)
        _guard_image_size(w, h, "LOAD_PNG", location)
        return _make_tensor_from_pixels(w, h, pixels, "LOAD_PNG", location)
    except ASMRuntimeError:
        raise
    except Exception as exc:
        raise ASMRuntimeError(f"LOAD_PNG failed: {exc}", location=location, rewrite_rule="LOAD_PNG")


def _op_load_jpeg(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    path = _expect_str(args[0], "LOAD_JPEG", location)
    _check_path(path, "LOAD_JPEG", location)
    try:
        w, h, pixels = _load_jpeg_file(path)
        _guard_image_size(w, h, "LOAD_JPEG", location)
        return _make_tensor_from_pixels(w, h, pixels, "LOAD_JPEG", location)
    except ASMRuntimeError:
        raise
    except Exception as exc:
        raise ASMRuntimeError(f"LOAD_JPEG failed: {exc}", location=location, rewrite_rule="LOAD_JPEG")


def _op_load_bmp(interpreter, args, _arg_nodes, _env, location):
    from interpreter import ASMRuntimeError

    path = _expect_str(args[0], "LOAD_BMP", location)
    _check_path(path, "LOAD_BMP", location)
    try:
        w, h, pixels = _load_bmp_file(path)
        _guard_image_size(w, h, "LOAD_BMP", location)
        return _make_tensor_from_pixels(w, h, pixels, "LOAD_BMP", location)
    except ASMRuntimeError:
        raise
    except Exception as exc:
        raise ASMRuntimeError(f"LOAD_BMP failed: {exc}", location=location, rewrite_rule="LOAD_BMP")


# ---- Registration ----

def asm_lang_register(ext: ExtensionAPI) -> None:
    ext.metadata(name="image", version="0.1.0")
    ext.register_operator("LOAD_PNG", 1, 1, _op_load_png, doc="LOAD_PNG(path) -> TNS[height][width][r,g,b,a]")
    ext.register_operator("LOAD_JPEG", 1, 1, _op_load_jpeg, doc="LOAD_JPEG(path) -> TNS[height][width][r,g,b,a]")
    ext.register_operator("LOAD_BMP", 1, 1, _op_load_bmp, doc="LOAD_BMP(path) -> TNS[height][width][r,g,b,a]")
