#!/usr/bin/env python3
"""
compare_fp_dequant.py — dequantize a single row slice of a tensor from
the ORIGINAL DeepSeek-V4 safetensors checkpoint (FP4 experts / FP8
attention with E8M0 block scales) without loading the full 340 GB
model into RAM.

Reads only the requested ~bytes via raw seek + read (no safetensors
library needed), then applies the same LUTs the Swift converter uses
(`Sources/DeepSeekConverter/DTypePacking.swift`) and prints values one
per line, in scientific notation. Designed to be diffed byte-for-byte
against the Swift CLI's `--dump-tensor` output:

    deepseek <int4-dir> "" --dump-tensor NAME:row=R:cols=A..B > swift.txt
    python3 Tools/compare_fp_dequant.py \\
        --model-dir <ORIGINAL-fp4-dir> \\
        --tensor NAME --row R --cols A..B > original.txt
    paste swift.txt original.txt | awk '{print $1, $2, $1-$2}'

Stdlib only (numpy + struct + json). No dependencies on safetensors,
torch, transformers.

Limits: rank-2 tensors only (weight matrices), F8_E4M3 / F4_E2M1
weight dtypes, F8_E8M0 scale dtype. Other formats raise.
"""
import argparse
import glob
import json
import os
import struct
import sys
from typing import Optional, Tuple


# ---------------------------------------------------------------- LUTs

def _make_e4m3_lut():
    """E4M3 FP8 (1 sign + 4 exp + 3 mant, bias 7). Mirrors Swift's
    `deqE4M3` in DTypePacking.swift."""
    lut = [0.0] * 256
    for b in range(256):
        sign = (b >> 7) & 1
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 0 and mant == 0:
            lut[b] = -0.0 if sign else 0.0
        elif exp == 0xF and mant == 0x7:
            lut[b] = float('nan')
        elif exp == 0:
            # Subnormal
            v = mant * (2.0 ** -9)
            lut[b] = -v if sign else v
        else:
            # value = (-1)^sign * 2^(exp - 7) * (1 + mant/8)
            sign_factor = -1.0 if sign else 1.0
            lut[b] = sign_factor * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
    return lut


def _make_e2m1_lut():
    """FP4 E2M1 (sign + 3-bit magnitude). 8 magnitudes: 0, 0.5, 1, 1.5,
    2, 3, 4, 6. Mirrors Swift's `deqE2M1`."""
    mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    return [(-1.0 if (n & 8) else 1.0) * mag[n & 7] for n in range(16)]


def _make_e8m0_lut():
    """E8M0 (exponent-only). value = 2^(byte - 127), produced by
    writing `byte << 23` into the exponent bits of a Float32. Mirrors
    Swift's `deqE8M0`."""
    lut = [0.0] * 256
    for b in range(256):
        if b == 0xFF:
            lut[b] = float('nan')
        else:
            bits = b << 23
            (lut[b],) = struct.unpack('<f', struct.pack('<I', bits))
    return lut


E4M3_LUT = _make_e4m3_lut()
E2M1_LUT = _make_e2m1_lut()
E8M0_LUT = _make_e8m0_lut()


# -------------------------------------------------- safetensors helpers

def _read_header(path: str) -> Tuple[dict, int]:
    """Return (header_json, data_start_byte_offset)."""
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
    return hdr, 8 + hdr_len


def _find_tensor(model_dir: str, name: str) -> Optional[Tuple[str, dict, int]]:
    """Locate the safetensors shard containing `name`. Returns
    (file_path, header_dict, data_start). Skips the `__metadata__`
    pseudo-entry. None if not found."""
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        hdr, data_start = _read_header(path)
        if name in hdr and isinstance(hdr.get(name), dict):
            return path, hdr, data_start
    return None


def _read_bytes(path: str, offset: int, n: int) -> bytes:
    """Raw fread of `n` bytes at absolute byte offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(n)


# -------------------------------------------------------------- dequant

def dequant_row_slice(
    model_dir: str,
    tensor_name: str,
    row: int,
    col_lo: int,
    col_hi: int,
) -> Tuple[list, dict, dict, str]:
    """Return (values, weight_info, scale_info, scale_name)."""
    loc = _find_tensor(model_dir, tensor_name)
    if loc is None:
        raise SystemExit(f"tensor not found in checkpoint: {tensor_name}")
    w_path, w_hdr, w_data_start = loc
    w_entry = w_hdr[tensor_name]
    w_dtype = w_entry["dtype"].upper()
    w_shape = w_entry["shape"]
    if len(w_shape) != 2:
        raise SystemExit(
            f"only rank-2 weights are supported (got shape={w_shape})")
    out_dim, in_dim = w_shape

    if not (0 <= row < out_dim):
        raise SystemExit(f"row {row} out of range 0..{out_dim}")
    if not (0 <= col_lo < col_hi <= in_dim):
        raise SystemExit(
            f"cols {col_lo}..{col_hi} out of range 0..{in_dim}")

    # Locate scale companion. Original DeepSeek FP4/FP8 checkpoints
    # use the `weight_scale_inv` suffix; the in-tree converter renames
    # that to `scale` for the INT4 path.
    base = tensor_name[:-len(".weight")] \
        if tensor_name.endswith(".weight") else tensor_name
    scale_candidates = [
        f"{base}.weight_scale_inv",
        f"{base}.scale_inv",
        f"{base}.scale",
    ]
    s_path, s_hdr, s_data_start, s_entry, scale_name = (
        None, None, None, None, None)
    # Prefer same-shard hit (cheaper, common case)
    for c in scale_candidates:
        if c in w_hdr and isinstance(w_hdr.get(c), dict):
            scale_name = c
            s_path, s_hdr, s_data_start = w_path, w_hdr, w_data_start
            s_entry = w_hdr[c]
            break
    if scale_name is None:
        for c in scale_candidates:
            hit = _find_tensor(model_dir, c)
            if hit is not None:
                scale_name = c
                s_path, s_hdr, s_data_start = hit
                s_entry = s_hdr[c]
                break
    if scale_name is None:
        raise SystemExit(
            f"no scale companion found for {tensor_name} "
            f"(tried {scale_candidates})")

    s_dtype = s_entry["dtype"].upper()
    s_shape = s_entry["shape"]
    if s_dtype not in ("F8_E8M0", "F8E8M0", "FLOAT8_E8M0FNU"):
        raise SystemExit(
            f"scale dtype {s_dtype} not supported (expected F8_E8M0)")

    # ---- weight reader (returns dequant'd float for an absolute col)
    if w_dtype in ("F8_E4M3", "F8E4M3", "FLOAT8_E4M3FN"):
        # 1 byte per value, row major.
        row_bytes = in_dim
        weight_lut = E4M3_LUT

        def fetch_weight_row():
            offset = (w_data_start
                      + w_entry["data_offsets"][0]
                      + row * row_bytes + col_lo)
            n = col_hi - col_lo
            raw = _read_bytes(w_path, offset, n)
            return [weight_lut[b] for b in raw]
    elif w_dtype in ("F4_E2M1", "F4E2M1", "FLOAT4_E2M1FN_X2"):
        # 4 bits per value, packed 2-per-byte: byte[k] = high<<4 | low,
        # where low = col 2k, high = col 2k+1 (mirrors our Int4Quant
        # nibble convention; verify against the original by checking
        # if values look like clean multiples of the block scale).
        row_bytes = in_dim // 2
        weight_lut = E2M1_LUT

        def fetch_weight_row():
            # Read full bytes covering [col_lo, col_hi)
            first_byte = col_lo // 2
            last_byte = (col_hi + 1) // 2  # exclusive
            offset = (w_data_start
                      + w_entry["data_offsets"][0]
                      + row * row_bytes + first_byte)
            raw = _read_bytes(w_path, offset, last_byte - first_byte)
            out = []
            for c in range(col_lo, col_hi):
                byte = raw[c // 2 - first_byte]
                nib = (byte & 0xF) if (c % 2 == 0) else (byte >> 4)
                out.append(weight_lut[nib])
            return out
    else:
        raise SystemExit(
            f"weight dtype {w_dtype} not supported "
            f"(expected F8_E4M3 or F4_E2M1)")

    # ---- scale reader (auto-detect 1D-per-row vs 2D-blocks)
    if len(s_shape) != 2:
        raise SystemExit(f"unexpected scale rank {len(s_shape)}")
    if s_shape[0] == out_dim:
        # Per-row × per-G blocks. G = in_dim / s_shape[1].
        group_in = in_dim // s_shape[1]
        s_row_bytes = s_shape[1]

        def fetch_scale_for(col):
            blk = col // group_in
            offset = (s_data_start
                      + s_entry["data_offsets"][0]
                      + row * s_row_bytes + blk)
            (b,) = _read_bytes(s_path, offset, 1)
            return E8M0_LUT[b]
        scale_layout = f"1D per-row × per-{group_in}"
    else:
        # 2D blocks. Bo = out_dim / s_shape[0], Bi = in_dim / s_shape[1].
        Bo = out_dim // s_shape[0]
        Bi = in_dim // s_shape[1]
        s_row_bytes = s_shape[1]

        def fetch_scale_for(col):
            block_r = row // Bo
            block_c = col // Bi
            offset = (s_data_start
                      + s_entry["data_offsets"][0]
                      + block_r * s_row_bytes + block_c)
            (b,) = _read_bytes(s_path, offset, 1)
            return E8M0_LUT[b]
        scale_layout = f"2D blocks {Bo}×{Bi}"

    # ---- dequant + assemble result
    weight_vals = fetch_weight_row()
    values = []
    for k, c in enumerate(range(col_lo, col_hi)):
        s = fetch_scale_for(c)
        values.append(weight_vals[k] * s)

    info = {
        "weight": tensor_name,
        "weight_path": os.path.basename(w_path),
        "weight_dtype": w_dtype,
        "weight_shape": w_shape,
        "scale": scale_name,
        "scale_path": os.path.basename(s_path),
        "scale_dtype": s_dtype,
        "scale_shape": s_shape,
        "scale_layout": scale_layout,
        "row": row,
        "cols": (col_lo, col_hi),
    }
    return values, w_entry, s_entry, info


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Dequantize a row slice of an original DeepSeek-V4 FP4/FP8 "
            "tensor; compare against the Swift INT4 dump."))
    ap.add_argument("--model-dir", required=True,
                    help="Directory of the ORIGINAL (FP4/FP8) safetensors "
                         "checkpoint")
    ap.add_argument("--tensor", required=True,
                    help="e.g. layers.0.attn.wq_a.weight")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--cols", default="0..32",
                    help="A..B (default 0..32). Half-open; B exclusive.")
    args = ap.parse_args()

    try:
        c_lo, c_hi = map(int, args.cols.split(".."))
    except ValueError:
        sys.exit("--cols must be A..B")

    values, _, _, info = dequant_row_slice(
        args.model_dir, args.tensor, args.row, c_lo, c_hi)

    sys.stderr.write(
        f"# weight {info['weight']} {info['weight_dtype']} "
        f"{info['weight_shape']} in {info['weight_path']}\n"
        f"# scale  {info['scale']} {info['scale_dtype']} "
        f"{info['scale_shape']} ({info['scale_layout']})\n"
        f"# row={info['row']} cols={info['cols'][0]}..{info['cols'][1]}"
        f" ({len(values)} values)\n")

    for v in values:
        print(f"{v:.8e}")


if __name__ == "__main__":
    main()
