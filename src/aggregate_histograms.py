from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def counter_from_dict(d: dict) -> Counter:
    c = Counter()
    for k, v in d.items():
        c[int(k)] += int(v)
    return c


def same_ps(ps_a, ps_b, *, tol=1e-12) -> bool:
    """Order-insensitive float list equality with tolerance."""
    if ps_a is None or ps_b is None:
        return False
    if len(ps_a) != len(ps_b):
        return False
    a = sorted(float(x) for x in ps_a)
    b = sorted(float(x) for x in ps_b)
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def normalize_modes(modes) -> list[int] | None:
    if modes is None:
        return None
    return sorted(int(x) for x in modes)


def merge_jsons_filtered(
    glob_pattern: str,
    *,
    npz: str | None = None,
    ps: list[float] | None = None,
    runs: int | None = None,
    modes: list[int] | None = None,
    guess_cap: int | None = None,
):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise SystemExit(f"No files matched: {glob_pattern}")

    modes_norm = normalize_modes(modes)

    merged = defaultdict(lambda: defaultdict(lambda: {
        "inact_success": Counter(),
        "inact_failure": Counter(),
        "stab": Counter(),
    }))

    used_files = []
    rejected = 0

    meta_example = None

    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception:
            rejected += 1
            continue

        if npz is not None and data.get("npz") != npz:
            continue

        if ps is not None and not same_ps(data.get("ps"), ps):
            continue

        if runs is not None and int(data.get("runs", -1)) != int(runs):
            continue

        if guess_cap is not None:
            if data.get("guess_cap", None) is None:
                continue
            if int(data.get("guess_cap")) != int(guess_cap):
                continue

        if modes_norm is not None:
            if normalize_modes(data.get("modes")) != modes_norm:
                continue

        hists = data.get("hists", None)
        if not isinstance(hists, dict) or not hists:
            continue

        used_files.append(fp)
        if meta_example is None:
            meta_example = {
                "npz": data.get("npz"),
                "ps": data.get("ps"),
                "runs": data.get("runs"),
                "modes": data.get("modes"),
                "guess_cap": data.get("guess_cap"),
            }

        for mode_str, per_p in hists.items():
            mode_i = int(mode_str)
            for p_str, hist in per_p.items():
                p_val = float(p_str)
                merged[mode_i][p_val]["inact_success"] += counter_from_dict(hist.get("inact_success", {}))
                merged[mode_i][p_val]["inact_failure"] += counter_from_dict(hist.get("inact_failure", {}))
                merged[mode_i][p_val]["stab"] += counter_from_dict(hist.get("stab", {}))

    if not used_files:
        raise SystemExit(
            "No matching JSONs found.\n"
            f"Pattern: {glob_pattern}\n"
            f"Filters: npz={npz}, ps={ps}, runs={runs}, modes={modes}, guess_cap={guess_cap}"
        )

    return used_files, meta_example, merged, rejected


def plot_inact(merged, ps, out_prefix: str):
    for p in ps:
        modes = [m for m in (2, 4) if (m in merged and p in merged[m])]
        if not modes:
            continue

        fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), sharey=True)
        if len(modes) == 1:
            axes = [axes]

        for ax, mode in zip(axes, modes):
            succ = merged[mode][p]["inact_success"]
            fail = merged[mode][p]["inact_failure"]

            max_g = 0
            if succ:
                max_g = max(max_g, max(succ))
            if fail:
                max_g = max(max_g, max(fail))

            x = np.arange(max_g + 1)
            succ_arr = np.array([succ.get(i, 0) for i in x])
            fail_arr = np.array([fail.get(i, 0) for i in x])

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


def plot_stab_mode4(merged, ps, out_prefix: str):
    if 4 not in merged:
        return

    for p in ps:
        if p not in merged[4]:
            continue

        stab = merged[4][p]["stab"]
        if not stab:
            continue

        max_g = max(stab)
        x = np.arange(max_g + 1)
        y = np.array([stab.get(i, 0) for i in x])

        plt.figure(figsize=(5, 4))
        plt.bar(x, y, align="center")
        plt.title(f"Mode 4 – stabilizer moves (p={p:.2f})")
        plt.xlabel("# stabilizer moves")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_p{p:.2f}.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Merge histogram counts across many result JSONs and plot once."
    )
    ap.add_argument("--glob", default="results/*.json", help="Where to look for JSONs (default: results/*.json)")
    ap.add_argument("--outdir", default="results/merged_histograms", help="Where to write PNGs")
    ap.add_argument("--label", default="merged", help="Prefix label for output files")

    ap.add_argument("--npz", default=None, help="Only include JSONs whose 'npz' matches exactly")
    ap.add_argument("--ps", type=float, nargs="+", default=None, help="Only include JSONs with exactly these ps (order-insensitive)")
    ap.add_argument("--runs", type=int, default=None, help="Only include JSONs with this runs value")
    ap.add_argument("--modes", type=int, nargs="+", default=None, help="Only include JSONs with exactly these modes (order-insensitive)")
    ap.add_argument("--guess-cap", type=int, default=None, help="Only include JSONs with this guess_cap")

    args = ap.parse_args()

    used_files, meta, merged, rejected = merge_jsons_filtered(
        args.glob,
        npz=args.npz,
        ps=args.ps,
        runs=args.runs,
        modes=args.modes,
        guess_cap=args.guess_cap,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.ps is not None:
        ps_list = sorted(float(x) for x in args.ps)
    else:
        ps_list = sorted(float(x) for x in (meta.get("ps") if meta else []))

    inact_prefix = str(outdir / f"{args.label}_inact")
    stab_prefix = str(outdir / f"{args.label}_stab")

    plot_inact(merged, ps_list, inact_prefix)
    plot_stab_mode4(merged, ps_list, stab_prefix)

    merged_out = outdir / f"{args.label}_merged_counts.json"
    serial = {}
    for mode, per_p in merged.items():
        serial[str(mode)] = {}
        for p, h in per_p.items():
            serial[str(mode)][str(p)] = {
                "inact_success": dict(h["inact_success"]),
                "inact_failure": dict(h["inact_failure"]),
                "stab": dict(h["stab"]),
            }

    with open(merged_out, "w") as f:
        json.dump(
            {"meta_example": meta, "used_files": used_files, "rejected_unreadable": rejected, "hists": serial},
            f,
            indent=2
        )

    print(f"Matched {len(used_files)} JSONs (ignored unreadable={rejected}).")
    print(f"Wrote PNGs to: {outdir}")
    print(f"Wrote merged counts to: {merged_out}")


if __name__ == "__main__":
    main()


