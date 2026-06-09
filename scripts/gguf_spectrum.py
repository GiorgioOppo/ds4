#!/usr/bin/env python3
"""gguf-spectrum — analizza la "comprimibilità" di un modello GGUF.

Per i tensori principali (proiezioni di attenzione, shared FFN, router, esperti,
testa di output) calcola lo spettro dei valori singolari e riporta:

  • il rank effettivo al 90/95/99% dell'energia (somma dei quadrati dei sing. val.);
  • il guadagno se la matrice fosse fattorizzata low-rank  W ≈ U·V  (r·(out+in) vs out·in);
  • per gli esperti routed: la RIDONDANZA fra i 256 esperti (PCA fra esperti, via
    proiezione casuale) → quante "basi" condivise catturano il 90/95/99% della
    varianza, cioè quanto un dizionario condiviso + delta per-esperto comprimerebbe.

Serve a decidere SUI NUMERI se conviene una rappresentazione fattorizzata/"a grafo"
(pesi sui cammini) prima di scrivere kernel o fare fine-tuning.

Dipendenze:  pip install -U gguf numpy
La dequantizzazione (Q8_0, Q4_K, Q2_K, IQ2_XXS, F16, …) usa gguf.quants di
llama.cpp, quindi è corretta per tutti i formati che l'engine usa.

Esempi:
  python3 gguf_spectrum.py model.gguf                  # layer 2, attn+shared+router
  python3 gguf_spectrum.py model.gguf --layers 0,2,20 --experts
  python3 gguf_spectrum.py model.gguf --full           # include output/embeddings (lento)
  python3 gguf_spectrum.py model.gguf --layers 2 --experts --proj 8192 --json out.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("Manca numpy:  pip install numpy")

try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType
except Exception as e:  # noqa: BLE001
    sys.exit(f"Manca/obsoleto il pacchetto gguf ({e}).  pip install -U gguf")


# ---------------------------------------------------------------- dequant utils

def dequant_flat(raw_bytes: np.ndarray, qtype: GGMLQuantizationType, n_elements: int) -> np.ndarray:
    """Dequantizza un blocco di byte grezzi in float32 (lunghezza n_elements)."""
    out = dequantize(raw_bytes, qtype).astype(np.float32).reshape(-1)
    if out.size != n_elements:
        out = out[:n_elements]
    return out


# ---------------------------------------------------------------- spettro SVD

def singular_values(mat: np.ndarray) -> np.ndarray:
    """Valori singolari (decrescenti) di una matrice 2D, in float32."""
    m = mat.astype(np.float32, copy=False)
    # gesdd è veloce e stabile; compute_uv=False evita U/V (solo i sing. val.).
    return np.linalg.svd(m, compute_uv=False)


def energy_ranks(s: np.ndarray, thresholds=(0.90, 0.95, 0.99)) -> dict[float, int]:
    if s.size == 0:
        return {t: 0 for t in thresholds}
    energy = np.cumsum(s.astype(np.float64) ** 2)
    energy /= energy[-1]
    return {t: int(np.searchsorted(energy, t) + 1) for t in thresholds}


def lowrank_ratio(out_f: int, in_f: int, r: int) -> float:
    """Frazione di parametri tenuti se W≈U·V con rank r (r·(out+in) / (out·in))."""
    full = out_f * in_f
    return (r * (out_f + in_f)) / full if full else 1.0


def analyze_matrix(name: str, mat: np.ndarray) -> dict:
    out_f, in_f = mat.shape
    t0 = time.time()
    s = singular_values(mat)
    er = energy_ranks(s)
    r95 = er[0.95]
    return {
        "name": name,
        "shape": [out_f, in_f],
        "full_rank": int(min(out_f, in_f)),
        "rank90": er[0.90], "rank95": er[0.95], "rank99": er[0.99],
        "lowrank_keep95": round(lowrank_ratio(out_f, in_f, r95), 3),
        "lowrank_x95": round(1.0 / lowrank_ratio(out_f, in_f, r95), 2) if r95 else None,
        "secs": round(time.time() - t0, 1),
    }


def fmt_matrix(r: dict) -> str:
    o, i = r["shape"]
    keep = r["lowrank_keep95"]
    win = f"{r['lowrank_x95']}×" if keep < 1.0 else "nessun guadagno"
    return (f"  {r['name']:<34} {o:>6}×{i:<6} fullrank={r['full_rank']:>5} | "
            f"rank @90/95/99 = {r['rank90']:>4}/{r['rank95']:>4}/{r['rank99']:>4} | "
            f"low-rank@95: tieni {keep*100:5.1f}%  ({win})")


# ------------------------------------------------------------- ridondanza esperti

def analyze_experts(name: str, raw: np.ndarray, qtype, n_expert: int, out_f: int,
                    in_f: int, proj: int, seed: int = 0) -> dict:
    """PCA fra esperti: proietta ogni esperto (out·in) su `proj` dimensioni casuali
    fisse, poi misura quante componenti catturano la varianza fra i 256 esperti."""
    per = out_f * in_f
    raw = np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    bytes_per_expert = raw.size // n_expert
    if bytes_per_expert * n_expert != raw.size:
        raise ValueError(f"{name}: byte non divisibili per {n_expert} esperti")

    rng = np.random.default_rng(seed)
    idx = rng.choice(per, size=min(proj, per), replace=False)
    idx.sort()

    X = np.empty((n_expert, idx.size), dtype=np.float32)
    rank95_sum = 0
    sample_ranks = []
    for e in range(n_expert):
        sl = raw[e * bytes_per_expert:(e + 1) * bytes_per_expert]
        vals = dequant_flat(sl, qtype, per)
        X[e] = vals[idx]
        if e < 8:  # rank effettivo medio su un campione di esperti (matrice piena)
            s = singular_values(vals.reshape(out_f, in_f))
            sample_ranks.append(energy_ranks(s)[0.95])

    mean = X.mean(axis=0)
    Xc = X - mean
    # energia nella media (componente condivisa) vs nei residui
    mean_energy = float((mean ** 2).sum() * n_expert)
    total_energy = float((X ** 2).sum())
    s = np.linalg.svd(Xc, compute_uv=False)
    comp = energy_ranks(s)
    avg_rank = int(round(sum(sample_ranks) / len(sample_ranks))) if sample_ranks else 0
    return {
        "name": name, "n_expert": n_expert, "shape": [out_f, in_f], "proj": int(idx.size),
        "per_expert_rank95_avg": avg_rank,
        "shared_basis_90": comp[0.90], "shared_basis_95": comp[0.95], "shared_basis_99": comp[0.99],
        "mean_energy_frac": round(mean_energy / total_energy, 3) if total_energy else 0.0,
        "naive_share_x95": round(n_expert / (comp[0.95] + 1), 1) if comp[0.95] else None,
    }


def fmt_experts(r: dict) -> str:
    return (f"  {r['name']:<24} {r['n_expert']}×({r['shape'][0]}×{r['shape'][1]}) | "
            f"rank/esperto≈{r['per_expert_rank95_avg']} | "
            f"basi condivise @90/95/99 = {r['shared_basis_90']}/{r['shared_basis_95']}/{r['shared_basis_99']} | "
            f"energia nella media {r['mean_energy_frac']*100:.0f}% | "
            f"sharing naive ~{r['naive_share_x95']}×")


# ---------------------------------------------------------------- selezione tensori

DENSE_PER_LAYER = [
    "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
    "attn_output_a.weight", "attn_output_b.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    "ffn_gate_inp.weight",
]
EXPERTS_PER_LAYER = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]
GLOBAL_FULL = ["output.weight", "token_embd.weight"]


def ne_dims(t) -> list[int]:
    """Dimensioni ggml (ne0, ne1, …) del tensore."""
    return [int(x) for x in t.shape]


def main() -> int:
    ap = argparse.ArgumentParser(description="Analisi spettrale / ridondanza di un GGUF.")
    ap.add_argument("gguf")
    ap.add_argument("--layers", default="2", help="lista layer, es. 0,2,20 (default: 2)")
    ap.add_argument("--experts", action="store_true", help="analizza la ridondanza fra esperti")
    ap.add_argument("--full", action="store_true", help="includi output.weight e token_embd.weight (lento)")
    ap.add_argument("--proj", type=int, default=4096, help="dim. proiezione PCA esperti (default 4096)")
    ap.add_argument("--max-dim", type=int, default=20000,
                    help="salta la SVD di matrici con lato > questo (default 20000)")
    ap.add_argument("--json", help="scrivi i risultati anche in questo file JSON")
    args = ap.parse_args()

    try:
        layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    except ValueError:
        return ap.error("--layers vuole interi separati da virgola")

    print(f"Apro {args.gguf} …")
    reader = GGUFReader(args.gguf)
    index = {t.name: t for t in reader.tensors}
    arch_n = len({m.group(1) for n in index for m in [re.match(r"blk\.(\d+)\.", n)] if m})
    print(f"  {len(index)} tensori, ~{arch_n} layer rilevati\n")

    results = {"dense": [], "experts": [], "global": []}

    def get_matrix(name):
        t = index.get(name)
        if t is None:
            return None
        ne = ne_dims(t)                       # [ne0=in, ne1=out, …]
        if len(ne) < 2:
            return None
        in_f, out_f = ne[0], ne[1]
        if max(out_f, in_f) > args.max_dim:
            print(f"  (salto {name}: {out_f}×{in_f} > --max-dim)")
            return None
        raw = np.ascontiguousarray(t.data).view(np.uint8).reshape(-1)
        flat = dequant_flat(raw, t.tensor_type, out_f * in_f)
        return flat.reshape(out_f, in_f)

    # --- dense per-layer
    for L in layers:
        print(f"=== Layer {L} — dense (attn / shared / router) ===")
        for suffix in DENSE_PER_LAYER:
            name = f"blk.{L}.{suffix}"
            try:
                mat = get_matrix(name)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {name}: {e}")
                continue
            if mat is None:
                continue
            r = analyze_matrix(name, mat)
            results["dense"].append(r)
            print(fmt_matrix(r))
        print()

    # --- esperti
    if args.experts:
        for L in layers:
            print(f"=== Layer {L} — esperti (ridondanza fra esperti) ===")
            for suffix in EXPERTS_PER_LAYER:
                name = f"blk.{L}.{suffix}"
                t = index.get(name)
                if t is None:
                    continue
                ne = ne_dims(t)               # [in, out, n_expert]
                if len(ne) != 3:
                    print(f"  ! {name}: atteso 3D, trovato {ne}")
                    continue
                in_f, out_f, n_expert = ne[0], ne[1], ne[2]
                try:
                    raw = np.ascontiguousarray(t.data)
                    r = analyze_experts(name, raw, t.tensor_type, n_expert, out_f, in_f, args.proj)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {name}: {e}")
                    continue
                results["experts"].append(r)
                print(fmt_experts(r))
            print()

    # --- globali
    if args.full:
        print("=== Globali (output / embeddings) ===")
        for name in GLOBAL_FULL:
            try:
                mat = get_matrix(name)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {name}: {e}")
                continue
            if mat is None:
                continue
            r = analyze_matrix(name, mat)
            results["global"].append(r)
            print(fmt_matrix(r))
        print()

    print("Legenda: 'rank @95' = quante componenti catturano il 95% dell'energia; "
          "'low-rank@95 tieni X%' = parametri residui se fattorizzi a quel rank "
          "(<100% = comprimibile). Per gli esperti, 'basi condivise' = dimensione di un "
          "dizionario comune; 'sharing naive ~K×' è il limite superiore di compressione "
          "del blocco MoE PRIMA dei residui e SENZA fine-tuning.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nScritto {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
