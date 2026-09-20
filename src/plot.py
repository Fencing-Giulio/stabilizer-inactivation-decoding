from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_modes(modes) -> list[int] | None:
    if modes is None:
        return None
    return sorted(int(x) for x in modes)


def same_ps(ps_a, ps_b, *, tol=1e-12) -> bool:
    if ps_a is None or ps_b is None:
        return False
    if len(ps_a) != len(ps_b):
        return False
    a = sorted(float(x) for x in ps_a)
    b = sorted(float(x) for x in ps_b)
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def merge_results(
    glob_pattern: str,
    *,
    npz: str | None = None,
    ps: list[float] | None = None,
    runs: int | None = None,
    modes: list[int] | None = None,
    guess_cap=None,
):
    """
    Merges the 'results' field across matching JSON files.
    Returns:
      merged[mode][p] = {same, degen, logic, nonconv, runs, ...} (summed counts)
    """
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise SystemExit(f"No files matched: {glob_pattern}")

    modes_norm = normalize_modes(modes)

    merged: dict[int, dict[float, dict]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    used_files = []
    rejected = 0
    meta_example = None

    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            rejected += 1
            continue

        if npz is not None and data.get("npz") != npz:
            continue
        if ps is not None and not same_ps(data.get("ps"), ps):
            continue
        if runs is not None and int(data.get("runs", -1)) != runs:
            continue
        if modes_norm is not None and normalize_modes(data.get("modes")) != modes_norm:
            continue
        if guess_cap is not None:
            if data.get("guess_cap") is None or int(data["guess_cap"]) != int(guess_cap):
                continue

        results = data.get("results")
        if not isinstance(results, dict):
            rejected += 1
            continue

        used_files.append(fp)
        if meta_example is None:
            meta_example = {k: data.get(k) for k in ("npz", "ps", "runs", "modes", "guess_cap")}

        for mode_str, entries in results.items():
            mode_i = int(mode_str)
            for entry in entries:
                p_val = float(entry["p"])
                for key in ("same", "degen", "logic", "nonconv", "runs"):
                    merged[mode_i][p_val][key] += int(entry.get(key, 0))

    if not used_files:
        raise SystemExit(
            f"No matching JSONs found.\n"
            f"Pattern : {glob_pattern}\n"
            f"Filters : npz={npz}, ps={ps}, runs={runs}, modes={modes}, guess_cap={guess_cap}"
        )

    print(f"Matched {len(used_files)} file(s), ignored {rejected} unreadable.")
    return merged, meta_example, used_files


MODE_LABELS = {
    1: "Mode 1 – peel only",
    2: "Mode 2 – inactivation only",
    3: "Mode 3 – structured + peel",
    4: "Mode 4 – structured + inactivation",
    5: "Mode 5 – iterative + inactivation",
    6: "Mode 6 – peel + hard guess",
    7: "Mode 7 – structured + peel + hard guess",
}

MODE_STYLES = {
    1: dict(color="#378ADD", marker="o", linestyle="-"),
    2: dict(color="#1D9E75", marker="s", linestyle="-"),
    3: dict(color="#D85A30", marker="^", linestyle="-"),
    4: dict(color="#D4537E", marker="D", linestyle="-"),
    5: dict(color="#7F77DD", marker="p", linestyle="-"),
    6: dict(color="#BA7517", marker="X", linestyle="-"),
    7: dict(color="#639922", marker="*", linestyle="-"),
}


def plot_threshold(
    merged: dict,
    out_path: str,
    *,
    title: str | None = None,
    show_wilson: bool = True,
):
    """
    One line per mode: x = p, y = logical error rate = (logic + nonconv) / runs.
    Optional Wilson 95% CI error bars.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for mode in sorted(merged.keys()):
        per_p = merged[mode]
        pairs = sorted(per_p.items())
        ps_vals = [p for p, _ in pairs]
        err_rates, lo_errs, hi_errs = [], [], []

        for p, c in pairs:
            n = c["runs"]
            if n == 0:
                err_rates.append(float("nan"))
                lo_errs.append(0)
                hi_errs.append(0)
                continue

            fail = c["logic"] + c["nonconv"]
            rate = fail / n
            err_rates.append(rate)

            if show_wilson:
                z = 1.96
                denom = 1 + z**2 / n
                centre = (rate + z**2 / (2 * n)) / denom
                half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denom
                lo_errs.append(max(0, rate - max(0, centre - half)))
                hi_errs.append(max(0, min(1, centre + half) - rate))

        style = MODE_STYLES.get(mode, dict(color="gray", marker="o", linestyle="--"))
        label = MODE_LABELS.get(mode, f"Mode {mode}")

        if show_wilson:
            ax.errorbar(
                ps_vals, err_rates,
                yerr=[lo_errs, hi_errs],
                label=label,
                capsize=3,
                **style,
            )
        else:
            ax.plot(ps_vals, err_rates, label=label, **style)

    ax.set_xlabel("Physical erasure rate  p", fontsize=12)
    ax.set_ylabel("Logical error rate", fontsize=12)
    ax.set_title(title or "Threshold curve", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def print_summary(merged: dict):
    for mode in sorted(merged.keys()):
        print(f"\n=== Mode {mode} ===")
        print(f"{'p':>6}  {'runs':>6}  {'logic':>6}  {'nonconv':>8}  {'err_rate':>10}  {'total_success':>14}")
        for p, c in sorted(merged[mode].items()):
            n = c["runs"]
            fail = c["logic"] + c["nonconv"]
            success = c["same"] + c["degen"]
            rate = fail / n if n else float("nan")
            print(f"{p:6.3f}  {n:6d}  {c['logic']:6d}  {c['nonconv']:8d}  {rate:10.6f}  {success:14d}")


def main():
    ap = argparse.ArgumentParser(description="Plot logical error rate vs p from result JSONs")
    ap.add_argument("--glob", default="results/*.json",
                    help="Glob pattern for input JSONs (default: results/*.json)")
    ap.add_argument("--out", default="results/threshold_plot.png",
                    help="Output PNG path (default: results/threshold_plot.png)")
    ap.add_argument("--title", default=None, help="Plot title")
    ap.add_argument("--no-errorbars", action="store_true",
                    help="Disable Wilson CI error bars")

    ap.add_argument("--npz", default=None)
    ap.add_argument("--ps", type=float, nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--modes", type=int, nargs="+", default=None)
    ap.add_argument("--guess-cap", type=int, default=None)

    args = ap.parse_args()

    merged, meta, used = merge_results(
        args.glob,
        npz=args.npz,
        ps=args.ps,
        runs=args.runs,
        modes=args.modes,
        guess_cap=args.guess_cap,
    )

    print_summary(merged)

    title = args.title
    if title is None and meta and meta.get("npz"):
        title = f"Threshold – {Path(meta['npz']).stem}"

    plot_threshold(
        merged,
        args.out,
        title=title,
        show_wilson=not args.no_errorbars,
    )


if __name__ == "__main__":
    main()