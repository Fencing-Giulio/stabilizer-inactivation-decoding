
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import Counter

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm_sparse import ERASURE, SparseMatrix
from decoder_sparse import SparseDecoder


def classify_from_decoder_output(
    out,
    Hz_dense: np.ndarray,
    Lz: np.ndarray | None = None,
) -> str:
    """
    EXACTLY like your old script.
    """
    corr = out.correction
    orig = out.answer

    if np.any(corr == ERASURE):
        return "nonconv"

    residual = (corr % 2 + orig % 2) % 2

    if not np.all((Hz_dense @ residual) % 2 == 0):
        return "nonconv"

    if residual.sum() == 0:
        return "same"

    if Lz is not None and Lz.size:
        if np.all((Lz @ residual) % 2 == 0):
            return "degen"
        else:
            return "logic"

    return "logic"


def run_once(
    decoder: SparseDecoder,
    Hz_dense: np.ndarray,
    p: float,
    seed: int,
    mode: int,
    Lz: Optional[np.ndarray] = None,
) -> Tuple[str, int, int]:
    """
    same as your old run_once_dense, just decoder-agnostic
    """
    rng = np.random.default_rng(seed)
    n = Hz_dense.shape[1]

    msg = np.zeros(n, dtype=np.int8)
    erased_mask = rng.random(n) < p
    msg[erased_mask] = ERASURE

    out = decoder.decode(msg, mode)

    cls = classify_from_decoder_output(out, Hz_dense, Lz=Lz)

    inact_guesses = out.inactivation_guesses or 0
    stab_guesses = out.stabilizer_guesses or 0

    return cls, inact_guesses, stab_guesses


def sim(
    decoder: SparseDecoder,
    Hz_dense: np.ndarray,
    ps: list[float],
    runs: int,
    mode: int,
    Lz: Optional[np.ndarray] = None,
    seed_offset: int = 0,
):
    """
    Returns:
      results: [(p, counts_dict), ...]
      hist_counts: {p: {"inact_success": {g: c}, "inact_failure": {g: c}, "stab": {g: c}}}

    Key change:
      We DO NOT store per-run lists anymore.
      We store additive histogram COUNTS so we can merge across array tasks.
    """
    results = []
    hist_counts: dict[float, dict[str, dict[int, int]]] = {}

    success_labels = {"same", "degen"}

    for p in ps:
        counts = dict(
            same=0,
            degen=0,
            logic=0,
            nonconv=0,
            inact_guesses_total=0,
            stab_guesses_total=0,
            total_time=0.0,
        )

        inact_success = Counter()
        inact_failure = Counter()
        stab_hist = Counter()

        t0 = time.perf_counter()
        for i in range(runs):
            t1 = time.perf_counter()
            cls, inact_g, stab_g = run_once(
                decoder, Hz_dense, p, seed=i + seed_offset, mode=mode, Lz=Lz
            )
            counts[cls] += 1
            counts["inact_guesses_total"] += inact_g
            counts["stab_guesses_total"] += stab_g
            counts["total_time"] += time.perf_counter() - t1

            if cls in success_labels:
                inact_success[inact_g] += 1
            else:
                inact_failure[inact_g] += 1

            stab_hist[stab_g] += 1

        t_end = time.perf_counter()

        counts["runs"] = runs
        counts["avg_time_per_run"] = counts["total_time"] / runs
        counts["total_time_all_runs"] = t_end - t0

        results.append((p, counts))
        hist_counts[p] = {
            "inact_success": {int(k): int(v) for k, v in inact_success.items()},
            "inact_failure": {int(k): int(v) for k, v in inact_failure.items()},
            "stab": {int(k): int(v) for k, v in stab_hist.items()},
        }

    return results, hist_counts


def plot_inactivation_histograms_from_counts(
    hist_by_mode: dict[int, dict[float, dict[str, dict[int, int]]]],
    ps: list[float],
    out_prefix: str = "inact_hist_stacked",
):
    have2 = 2 in hist_by_mode
    have4 = 4 in hist_by_mode
    if not (have2 or have4):
        return

    for p in ps:
        modes_for_p = []
        if have2 and p in hist_by_mode[2]:
            modes_for_p.append(2)
        if have4 and p in hist_by_mode[4]:
            modes_for_p.append(4)
        if not modes_for_p:
            continue

        n_sub = len(modes_for_p)
        fig, axes = plt.subplots(1, n_sub, figsize=(5 * n_sub, 4), sharey=True)
        if n_sub == 1:
            axes = [axes]

        for ax, mode in zip(axes, modes_for_p):
            data = hist_by_mode[mode][p]
            succ = data.get("inact_success", {})
            fail = data.get("inact_failure", {})

            max_g = 0
            if succ:
                max_g = max(max_g, max(succ.keys()))
            if fail:
                max_g = max(max_g, max(fail.keys()))

            x = np.arange(max_g + 1)
            succ_arr = np.array([succ.get(int(i), 0) for i in x])
            fail_arr = np.array([fail.get(int(i), 0) for i in x])

            ax.bar(x, succ_arr, label="success")
            ax.bar(x, fail_arr, bottom=succ_arr, label="failure")
            ax.set_title(f"Mode {mode} – inactivation guesses (p={p:.2f})")
            ax.set_xlabel("# inactivation guesses")
            ax.set_ylabel("run count")
            ax.legend()

        fig.suptitle(f"Inactivation guesses: success vs failure (p={p:.2f})")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_p{p:.2f}.png", dpi=150)
        plt.close(fig)


def plot_mode4_stabilizer_histograms_from_counts(
    hist_by_mode: dict[int, dict[float, dict[str, dict[int, int]]]],
    ps: list[float],
    out_prefix: str = "stab_hist",
):
    if 4 not in hist_by_mode:
        return

    for p in ps:
        if p not in hist_by_mode[4]:
            continue

        stab = hist_by_mode[4][p].get("stab", {})
        if not stab:
            continue

        max_guess = max(stab.keys())
        x = np.arange(max_guess + 1)
        y = np.array([stab.get(int(i), 0) for i in x])

        plt.figure(figsize=(5, 4))
        plt.bar(x, y, align="center")
        plt.title(f"Mode 4 – stabilizer moves (p={p:.2f})")
        plt.xlabel("# stabilizer moves")
        plt.ylabel("count")
        plt.tight_layout()

        plt.savefig(f"{out_prefix}_p{p:.2f}.png", dpi=150)
        plt.close()


def main(p, runs):
    ap = argparse.ArgumentParser(description="Sparse decoder simulation (DCC harness)")
    ap.add_argument("--npz", default="codes/hgp_code_2025_81_10_peg.npz",
                help="Path to npz with Hx, Hz, (opt) Lz")
    ap.add_argument("--ps", type=float, nargs="+", default=p,
                    help="Erasure probabilities")
    ap.add_argument("--runs", type=int, default=runs, help="Runs per p")
    ap.add_argument("--modes", type=int, nargs="+", default=[2, 3],
                    help="Decoder modes to run")
    ap.add_argument("--guess-cap", type=int, default=None,
                    help="Guess cap for inactivation (None = unlimited)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Global seed offset (use SLURM array offset)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: results/auto.json)")
    ap.add_argument("--results-dir", default="results",
                    help="Directory to store JSON if --out not given")
    ap.add_argument("--tag", default=None,
                    help="Optional tag to include in filename")

    ap.add_argument("--make-local-hists", action="store_true",
                    help="Generate PNG histograms from THIS run only (debug)")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    Hz = (npz["Hz"] % 2).astype(np.uint8)
    Hx = (npz["Hx"] % 2).astype(np.uint8)
    Lz = (
        (npz["Lz"] % 2).astype(np.uint8)
        if "Lz" in npz
        else np.empty((0, Hz.shape[1]), dtype=np.uint8)
    )

    sparse_dec = SparseDecoder(Hz, Hx, guess_cap=args.guess_cap)

    sparse_summary: dict[int, list[tuple[float, dict]]] = {}
    sparse_hist_counts: dict[int, dict[float, dict[str, dict[int, int]]]] = {}

    for mode in args.modes:
        res, hist = sim(
            sparse_dec,
            Hz,
            args.ps,
            args.runs,
            mode,
            Lz=Lz,
            seed_offset=args.seed_offset,
        )
        sparse_summary[mode] = res
        sparse_hist_counts[mode] = hist

    print("\n================ SPARSE ================")
    for mode in args.modes:
        print(f"\n=== Sparse mode: {mode} ===")
        for p_val, c in sparse_summary[mode]:
            total_fail = c["logic"] + c["nonconv"]
            success = c["same"] + c["degen"]
            err_rate = total_fail / c["runs"]
            avg_inact = c["inact_guesses_total"] / c["runs"]
            avg_stab = c["stab_guesses_total"] / c["runs"]
            avg_time = c.get("avg_time_per_run", 0.0)
            print(
                f"p={p_val:.2f}  runs={c['runs']:4d}  "
                f"same={c['same']:4d}  degen={c['degen']:4d}  "
                f"logic={c['logic']:4d}  nonconv={c['nonconv']:4d}  "
                f"success={success:4d}  "
                f"error_rate={err_rate:.4f}  "
                f"avg_inact_guesses={avg_inact:.2f}  "
                f"avg_stabilizer_moves={avg_stab:.2f}  "
                f"avg_time={avg_time*1000:.2f} ms"
            )

    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    default_name = (
        f"sparse_modes{''.join(map(str, args.modes))}"
        f"_runs{args.runs}_seedoff{args.seed_offset}{tag}_{ts}.json"
    )
    out_path = args.out or str(Path(args.results_dir) / default_name)
    os.makedirs(Path(out_path).parent, exist_ok=True)

    serializable = {
        "npz": args.npz,
        "ps": args.ps,
        "runs": args.runs,
        "modes": args.modes,
        "guess_cap": args.guess_cap,
        "seed_offset": args.seed_offset,
        "results": {
            int(mode): [
                {"p": float(p_val), **counts} for (p_val, counts) in sparse_summary[mode]
            ]
            for mode in args.modes
        },
        "hists": {
            int(mode): {
                float(p_val): sparse_hist_counts[mode][p_val]
                for p_val in sparse_hist_counts[mode].keys()
            }
            for mode in args.modes
        },
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved JSON -> {out_path}")

    if args.make_local_hists:
        out_path_p = Path(out_path)
        hist_dir = out_path_p.parent / "histograms_local"
        os.makedirs(hist_dir, exist_ok=True)
        base_name = out_path_p.stem
        inact_prefix = str(hist_dir / f"{base_name}_inact")
        stab_prefix = str(hist_dir / f"{base_name}_stab")

        plot_inactivation_histograms_from_counts(
            sparse_hist_counts,
            args.ps,
            out_prefix=inact_prefix,
        )
        plot_mode4_stabilizer_histograms_from_counts(
            sparse_hist_counts,
            args.ps,
            out_prefix=stab_prefix,
        )
        print(f"Saved LOCAL histograms -> {hist_dir}")


if __name__ == "__main__":
    p = [0.38, 0.42, 0.46, 0.48]
    runs = 1000
    main(p, runs)
