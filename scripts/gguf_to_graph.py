#!/usr/bin/env python3
"""gguf-to-graph — trasforma i pesi di un GGUF in un GRAFO con i pesi sugli ARCHI.

Idea (dalla discussione): invece di tenere ogni matrice di pesi in un "nodo"
(tensore denso `W[out×in]`), la si rappresenta come un cammino attraverso un nodo
collo-di-bottiglia di dimensione r:

        in ──A[out×r?]──►  (bottleneck r)  ──B[r×in]──► out         (sbagliato dir.)
    in (dim in) ──B[r×in]──► [bottleneck r] ──A[out×r]──► out (dim out)

ovvero  W ≈ A · B  (SVD troncata):  i PARAMETRI vivono sui due archi A e B, i
nodi sono solo spazi di attivazione (senza parametri). Per gli esperti routed si
fa la stessa cosa per esperto. Il guadagno c'è solo se la matrice è davvero a
basso rank: usa prima `gguf_spectrum.py` per vedere dove conviene.

⚠️ È una trasformazione LOSSY (SVD troncata): lo script riporta l'errore relativo
di ricostruzione per ogni matrice. Per recuperare la qualità servirebbe un
fine-tuning; questo strumento serve a costruire/visualizzare il grafo fattorizzato
e a misurare la compressione, non a produrre un modello pronto all'uso.

Output:
  • un grafo (nodi = spazi di attivazione + bottleneck; archi = matrici fattore)
    in JSON e, opzionale, in Graphviz DOT;
  • opzionale, le matrici fattore A,B salvate in un .npz;
  • un riepilogo: parametri originali vs fattorizzati e ratio, errore medio.

Dipendenze:  pip install -U gguf numpy
Esempi:
  python3 gguf_to_graph.py model.gguf --layers 2 --energy 0.95 --dot graph.dot
  python3 gguf_to_graph.py model.gguf --layers 0,2 --experts --rank 64 --npz factors.npz
"""
from __future__ import annotations
import argparse
import json
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("Manca numpy:  pip install numpy")

try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
except Exception as e:  # noqa: BLE001
    sys.exit(f"Manca/obsoleto il pacchetto gguf ({e}).  pip install -U gguf")


# ---------------------------------------------------------------- dequant utils

def dequant_flat(raw_bytes: np.ndarray, qtype, n_elements: int) -> np.ndarray:
    out = dequantize(raw_bytes, qtype).astype(np.float32).reshape(-1)
    return out[:n_elements] if out.size != n_elements else out


def ne_dims(t) -> list[int]:
    return [int(x) for x in t.shape]


# ---------------------------------------------------------------- fattorizzazione

def truncated_factor(W: np.ndarray, energy: float, fixed_rank: int | None):
    """W[out×in] ≈ A[out×r] · B[r×in] via SVD troncata.

    Ritorna (A, B, r, rel_error). rel_error = ||W-AB||_F / ||W||_F (esatto dai
    valori singolari scartati). I fattori assorbono sqrt(S) ciascuno."""
    out_f, in_f = W.shape
    U, S, Vt = np.linalg.svd(W.astype(np.float32), full_matrices=False)
    total = float((S ** 2).sum())
    if fixed_rank is not None:
        r = max(1, min(fixed_rank, S.size))
    else:
        cum = np.cumsum(S.astype(np.float64) ** 2) / (total if total else 1.0)
        r = int(np.searchsorted(cum, energy) + 1)
        r = max(1, min(r, S.size))
    dropped = float((S[r:] ** 2).sum())
    rel_err = (dropped / total) ** 0.5 if total else 0.0
    sq = np.sqrt(S[:r])
    A = (U[:, :r] * sq).astype(np.float32)          # [out×r]
    B = (sq[:, None] * Vt[:r, :]).astype(np.float32)  # [r×in]
    return A, B, r, rel_err


# ---------------------------------------------------------------- costruzione grafo

class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.factors: dict[str, np.ndarray] = {}
        self.orig_params = 0
        self.factored_params = 0
        self.errors: list[float] = []

    def node(self, nid: str, dim: int, role: str):
        self.nodes.setdefault(nid, {"id": nid, "dim": int(dim), "role": role})
        return nid

    def add_linear(self, name: str, src: str, dst: str, A: np.ndarray, B: np.ndarray,
                   r: int, err: float, out_f: int, in_f: int, save: bool):
        """src --B[r×in]--> bottleneck --A[out×r]--> dst. I pesi stanno sugli archi."""
        bn = self.node(f"{name}::bottleneck", r, "bottleneck")
        self.edges.append({"name": f"{name}.B", "src": src, "dst": bn,
                           "shape": [r, in_f], "params": r * in_f, "kind": "factor"})
        self.edges.append({"name": f"{name}.A", "src": bn, "dst": dst,
                           "shape": [out_f, r], "params": out_f * r, "kind": "factor"})
        self.orig_params += out_f * in_f
        self.factored_params += r * (out_f + in_f)
        self.errors.append(err)
        if save:
            self.factors[f"{name}.A"] = A
            self.factors[f"{name}.B"] = B

    def summary(self) -> dict:
        ratio = self.factored_params / self.orig_params if self.orig_params else 1.0
        return {
            "orig_params": self.orig_params,
            "factored_params": self.factored_params,
            "keep_fraction": round(ratio, 4),
            "compression_x": round(1.0 / ratio, 2) if ratio else None,
            "avg_rel_error": round(sum(self.errors) / len(self.errors), 4) if self.errors else 0.0,
            "max_rel_error": round(max(self.errors), 4) if self.errors else 0.0,
            "n_linears": len(self.errors),
            "n_nodes": len(self.nodes), "n_edges": len(self.edges),
        }

    def to_dot(self) -> str:
        lines = ["digraph gguf {", '  rankdir=LR;', '  node [shape=circle,fontsize=9];']
        for n in self.nodes.values():
            shape = "doublecircle" if n["role"] == "bottleneck" else "circle"
            lines.append(f'  "{n["id"]}" [label="{n["role"]}\\n{n["dim"]}",shape={shape}];')
        for e in self.edges:
            lbl = f'{e["name"].split(".")[-1]} {e["shape"][0]}x{e["shape"][1]}\\n{e["params"]:,}p'
            lines.append(f'  "{e["src"]}" -> "{e["dst"]}" [label="{lbl}",fontsize=8];')
        lines.append("}")
        return "\n".join(lines)


DENSE_PER_LAYER = [
    ("attn_q_a.weight", "x", "q_lat"), ("attn_q_b.weight", "q_lat", "q"),
    ("attn_kv.weight", "x", "kv_lat"),
    ("attn_output_a.weight", "heads", "attn_lat"), ("attn_output_b.weight", "attn_lat", "x"),
    ("ffn_gate_shexp.weight", "x", "ffn_h"), ("ffn_up_shexp.weight", "x", "ffn_h"),
    ("ffn_down_shexp.weight", "ffn_h", "x"),
    ("ffn_gate_inp.weight", "x", "router"),
]
EXPERTS_PER_LAYER = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]


def main() -> int:
    ap = argparse.ArgumentParser(description="GGUF → grafo fattorizzato (pesi sugli archi).")
    ap.add_argument("gguf")
    ap.add_argument("--layers", default="2", help="lista layer, es. 0,2 (default 2)")
    ap.add_argument("--experts", action="store_true", help="fattorizza anche gli esperti (per esperto)")
    ap.add_argument("--energy", type=float, default=0.95, help="energia da conservare (default 0.95)")
    ap.add_argument("--rank", type=int, default=None, help="rank fisso (ignora --energy)")
    ap.add_argument("--max-dim", type=int, default=20000, help="salta SVD con lato > questo")
    ap.add_argument("--max-experts", type=int, default=32, help="quanti esperti fattorizzare per tensore")
    ap.add_argument("--dot", help="scrivi il grafo in Graphviz DOT")
    ap.add_argument("--json", help="scrivi il grafo (nodi/archi/summary) in JSON")
    ap.add_argument("--npz", help="salva le matrici fattore A,B in un .npz")
    args = ap.parse_args()

    try:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    except ValueError:
        return ap.error("--layers vuole interi separati da virgola")
    save = args.npz is not None

    print(f"Apro {args.gguf} …")
    reader = GGUFReader(args.gguf)
    index = {t.name: t for t in reader.tensors}
    g = Graph()

    def factor_matrix(name, src, dst):
        t = index.get(name)
        if t is None:
            return
        ne = ne_dims(t)
        if len(ne) < 2:
            return
        in_f, out_f = ne[0], ne[1]
        if max(out_f, in_f) > args.max_dim:
            print(f"  (salto {name}: {out_f}×{in_f} > --max-dim)")
            return
        raw = np.ascontiguousarray(t.data).view(np.uint8).reshape(-1)
        W = dequant_flat(raw, t.tensor_type, out_f * in_f).reshape(out_f, in_f)
        t0 = time.time()
        A, B, r, err = truncated_factor(W, args.energy, args.rank)
        g.node(src, in_f, "act"); g.node(dst, out_f, "act")
        g.add_linear(name, src, dst, A, B, r, err, out_f, in_f, save)
        keep = r * (out_f + in_f) / (out_f * in_f)
        print(f"  {name:<30} {out_f}×{in_f}  r={r:<4} err={err*100:5.2f}%  "
              f"tieni {keep*100:5.1f}%  ({time.time()-t0:.1f}s)")

    for L in layers:
        print(f"=== Layer {L} — dense ===")
        for suffix, src, dst in DENSE_PER_LAYER:
            factor_matrix(f"blk.{L}.{suffix}", f"L{L}.{src}", f"L{L}.{dst}")
        if args.experts:
            print(f"=== Layer {L} — esperti (primi {args.max_experts}) ===")
            for suffix in EXPERTS_PER_LAYER:
                name = f"blk.{L}.{suffix}"
                t = index.get(name)
                if t is None:
                    continue
                ne = ne_dims(t)
                if len(ne) != 3:
                    print(f"  ! {name}: atteso 3D, trovato {ne}")
                    continue
                in_f, out_f, n_expert = ne[0], ne[1], ne[2]
                raw = np.ascontiguousarray(t.data).view(np.uint8).reshape(-1)
                bpe = raw.size // n_expert
                ne_count = min(args.max_experts, n_expert)
                errs = []
                t0 = time.time()
                for e in range(ne_count):
                    W = dequant_flat(raw[e * bpe:(e + 1) * bpe], t.tensor_type, out_f * in_f).reshape(out_f, in_f)
                    A, B, r, err = truncated_factor(W, args.energy, args.rank)
                    g.node(f"L{L}.expert_in", in_f, "act"); g.node(f"L{L}.expert_out", out_f, "act")
                    g.add_linear(f"{name}.e{e}", f"L{L}.expert_in", f"L{L}.expert_out",
                                 A, B, r, err, out_f, in_f, save)
                    errs.append(err)
                avg = sum(errs) / len(errs) if errs else 0.0
                print(f"  {name:<26} {ne_count}×{out_f}×{in_f}  err~{avg*100:5.2f}%  ({time.time()-t0:.1f}s)")
        print()

    s = g.summary()
    print("=== Riepilogo grafo fattorizzato ===")
    print(f"  linears fattorizzate : {s['n_linears']}  (nodi {s['n_nodes']}, archi {s['n_edges']})")
    print(f"  parametri originali  : {s['orig_params']:,}")
    print(f"  parametri sugli archi: {s['factored_params']:,}")
    print(f"  tieni {s['keep_fraction']*100:.1f}%  →  compressione ~{s['compression_x']}×")
    print(f"  errore ricostruzione : medio {s['avg_rel_error']*100:.2f}%  max {s['max_rel_error']*100:.2f}%")
    print("  (LOSSY: per recuperare la qualità servirebbe fine-tuning.)")

    if args.dot:
        with open(args.dot, "w") as f:
            f.write(g.to_dot())
        print(f"\nGrafo DOT → {args.dot}   (render: dot -Tsvg {args.dot} -o graph.svg)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"nodes": list(g.nodes.values()), "edges": g.edges, "summary": s}, f, indent=2)
        print(f"Grafo JSON → {args.json}")
    if args.npz:
        np.savez_compressed(args.npz, **g.factors)
        print(f"Fattori → {args.npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
