#!/usr/bin/env python3
# ruff: noqa: E501
"""
Diagnose strange decoded substrings in the prefill trace.

Use case: you saw something like "321133203232042" appear between
two known sections of the prompt trace (e.g. between `</think>` and
`<｜User｜>`) and want to understand whether:

  (a) it's a single token whose `content` is literally that digit
      string in the tokenizer's `added_tokens`,
  (b) it's a sequence of multiple consecutive tokens being
      decoded back as digits each, or
  (c) something else (silent token drop, byte-level mangling).

This script encodes a piece of text with the same tokenizer the
Swift app uses, then decodes it back and diffs the two — exactly
the round-trip path `Tools/inspect-prefill-trace.py` follows in
`BPETokenizer.decode` (Sources/DeepSeekKit/BPETokenizer.swift:233).

Usage:
  python3 Tools/inspect-prefill-trace.py \\
      /path/to/model_dir/tokenizer.json \\
      --substring "321133203232042" \\
      [--encode-roundtrip]
      [--list-digit-special-tokens]

If you don't have HuggingFace `tokenizers` installed, the script
falls back to a "best-effort" inspector that only scans
`added_tokens` for suspect content fields — useful even without the
encode/decode capability.

Why the round-trip matters: the Swift `BPETokenizer.decode`
silently skips tokens whose id is in neither `invAddedTokens` nor
`invVocab` (line 241: `guard let token = invVocab[id] else { continue }`).
If a chunk of the prompt encodes into tokens that the decoder can't
resolve, that chunk vanishes from the trace, leaving only the
tokens it *can* resolve — which is exactly the symptom of "long
section replaced by short digit string".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_tokenizer_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def added_tokens(tokjson: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the `added_tokens` array (or an empty list)."""
    arr = tokjson.get("added_tokens", [])
    if not isinstance(arr, list):
        return []
    return [e for e in arr if isinstance(e, dict)]


def vocab(tokjson: dict[str, Any]) -> dict[str, int]:
    """Return the BPE vocab dict (token-string -> id)."""
    model = tokjson.get("model", {})
    v = model.get("vocab", {})
    return v if isinstance(v, dict) else {}


def inv_added(tokjson: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for e in added_tokens(tokjson):
        tid = e.get("id")
        content = e.get("content")
        if isinstance(tid, int) and isinstance(content, str):
            out[tid] = content
    return out


def inv_vocab(tokjson: dict[str, Any]) -> dict[int, str]:
    return {tid: tok for tok, tid in vocab(tokjson).items()}


def report_digit_specials(tokjson: dict[str, Any]) -> None:
    """List every `added_tokens` entry whose content is a pure
    digit string — the smoking gun for "token id N decoded as the
    string of digits N"."""
    suspects = []
    for e in added_tokens(tokjson):
        tid = e.get("id")
        c = e.get("content")
        if isinstance(c, str) and c.strip() and c.strip().isdigit():
            suspects.append((tid, c))
    if not suspects:
        print("[ok] No added_tokens entry has digit-only `content`.")
        return
    print(f"[!] Found {len(suspects)} added_tokens with digit-only content:")
    for tid, c in suspects[:50]:
        match = "  ← content == str(id)" if str(tid) == c.strip() else ""
        print(f"  id={tid:>6}  content={c!r}{match}")
    if len(suspects) > 50:
        print(f"  … {len(suspects) - 50} more")


def report_suspicious_substring(tokjson: dict[str, Any], needle: str) -> None:
    """Look up `needle` in both added_tokens content and the BPE
    vocab — if anything matches, that's the source."""
    iadd = inv_added(tokjson)
    ivoc = inv_vocab(tokjson)

    hits_add = [(tid, c) for tid, c in iadd.items() if c == needle]
    hits_voc = [(tid, c) for tid, c in ivoc.items() if c == needle]
    print(f"\n--- Exact matches for needle {needle!r} ---")
    if hits_add:
        print(f"  added_tokens: {hits_add}")
    if hits_voc:
        print(f"  vocab:        {hits_voc}")
    if not hits_add and not hits_voc:
        print("  none — the needle is not a single token; it must be a "
              "sequence of multiple tokens.")

    # If not a single token, try to parse as concatenated short
    # token-id strings. Brute force across plausible widths (3..6
    # digits per id, typical DeepSeek special-token id range).
    print(f"\n--- Brute-force parse of {needle!r} as concatenated ids ---")
    parses = list(parse_as_id_concat(needle, iadd, min_w=3, max_w=6))
    if not parses:
        print("  No clean parse where every chunk maps to an "
              "added_tokens id with digit-only content.")
    else:
        for ids, contents in parses[:20]:
            arrow = " | ".join(f"{i}→{c!r}" for i, c in zip(ids, contents))
            print(f"  {ids} → {arrow}")


def parse_as_id_concat(
    s: str,
    iadd: dict[int, str],
    min_w: int = 3,
    max_w: int = 6,
):
    """Yield every way to split `s` into chunks where each chunk
    is the str() of a key in `iadd` whose value equals that chunk
    (i.e. an added token whose `content` IS its id-as-string)."""

    def go(i: int, acc_ids: list[int], acc_contents: list[str]):
        if i == len(s):
            yield acc_ids.copy(), acc_contents.copy()
            return
        for w in range(min_w, max_w + 1):
            chunk = s[i:i + w]
            if len(chunk) != w:
                continue
            try:
                tid = int(chunk)
            except ValueError:
                continue
            content = iadd.get(tid)
            if content == chunk:
                acc_ids.append(tid)
                acc_contents.append(content)
                yield from go(i + w, acc_ids, acc_contents)
                acc_ids.pop()
                acc_contents.pop()

    yield from go(0, [], [])


# ---------------------------------------------------------------
# Optional HF-tokenizers round-trip
# ---------------------------------------------------------------

def encode_decode_diff(tokenizer_json_path: Path, text: str) -> None:
    """If HuggingFace `tokenizers` is installed, encode + decode
    `text` through it and report any character drift."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("\n[skip] `pip install tokenizers` to enable the "
              "encode→decode round-trip check.")
        return

    tok = Tokenizer.from_file(str(tokenizer_json_path))
    enc = tok.encode(text, add_special_tokens=False)
    ids = enc.ids
    dec = tok.decode(ids, skip_special_tokens=False)
    print(f"\n--- Encode→decode round-trip ({len(text)} chars in → "
          f"{len(ids)} tokens → {len(dec)} chars out) ---")
    if dec == text:
        print("  [ok] round-trip is identity.")
    else:
        print("  [!] round-trip is LOSSY — character drift below.")
        print(f"  in : {text!r}")
        print(f"  out: {dec!r}")

    # Surface the highest 20 ids — if any of them are special
    # tokens with digit-only content, they're likely the source
    # of the digit substring you spotted.
    counts = Counter(ids)
    top = counts.most_common(20)
    iadd = inv_added(load_tokenizer_json(tokenizer_json_path))
    print(f"\n  Top 20 token ids by count (out of {len(ids)} total):")
    for tid, n in top:
        content = iadd.get(tid)
        marker = ""
        if isinstance(content, str) and content.isdigit():
            marker = "  ← digit-only added_token!"
        print(f"    id={tid:>6}  n={n}  added.content={content!r}{marker}")


# ---------------------------------------------------------------
# The toolsBlock template (mirrors Sources/DeepSeekKit/Encoding/
# EncodingDSV4.swift:58, with placeholder substitutions)
# ---------------------------------------------------------------

def build_sample_tools_block(tools_json: str = '[]') -> str:
    """Recreate the exact text the Swift app puts in the system
    message when tool schemas are present. Use it as the input to
    `--encode-roundtrip` to reproduce the same encode path the app
    takes at prefill time."""
    dt = "<｜DSML｜"
    think_open = "<think>"
    think_close = "</think>"
    return (
        "## Tools\n\n"
        f'You have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "{dt}tool_calls>" block like the following:\n\n'
        f"{dt}tool_calls>\n"
        f'{dt}invoke name="$TOOL_NAME">\n'
        f'{dt}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dt}parameter>\n'
        "...\n"
        f"</{dt}invoke>\n"
        f"</{dt}tool_calls>\n\n"
        'String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.\n\n'
        f"If thinking_mode is enabled (triggered by {think_open}), you MUST output your complete reasoning inside {think_open}...{think_close} BEFORE any tool calls or final response.\n\n"
        f"Otherwise, output directly after {think_close} with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n"
        f"{tools_json}\n\n"
        "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n\n"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tokenizer_json", type=Path,
                   help="Path to tokenizer.json from your model directory.")
    p.add_argument("--substring", type=str, default=None,
                   help="Suspicious substring spotted in the prefill trace "
                        "(e.g. '321133203232042').")
    p.add_argument("--list-digit-special-tokens", action="store_true",
                   help="Dump every added_tokens entry whose `content` is "
                        "a digit-only string.")
    p.add_argument("--encode-roundtrip", action="store_true",
                   help="Encode the toolsBlock template through HuggingFace "
                        "tokenizers and decode back to detect lossy tokens.")
    args = p.parse_args()

    if not args.tokenizer_json.exists():
        print(f"error: {args.tokenizer_json} not found", file=sys.stderr)
        return 2

    tokjson = load_tokenizer_json(args.tokenizer_json)
    print(f"Loaded {args.tokenizer_json}: "
          f"{len(added_tokens(tokjson))} added_tokens, "
          f"{len(vocab(tokjson))} vocab entries.")

    if args.list_digit_special_tokens or args.substring:
        report_digit_specials(tokjson)

    if args.substring:
        report_suspicious_substring(tokjson, args.substring)

    if args.encode_roundtrip:
        sample = build_sample_tools_block(tools_json='[]')
        encode_decode_diff(args.tokenizer_json, sample)

    return 0


if __name__ == "__main__":
    sys.exit(main())
