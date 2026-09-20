import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MODE_LABELS = {
    "1": "Peeling",
    "2": "Inactivation",
    "3": "Peeling+Stab",
    "4": "Stab+Inact",
    "6": "Hard guessing",
    "7": "Stab+Hard guessing",
}

PLOT_MODES = sorted(MODE_LABELS.keys(), key=int)


def iter_json_files(folder: Path):
    yield from sorted(folder.glob("*.json"))


def load_and_aggregate(folder: Path):
    """
    Returns:
      agg[mode][p] = dict(logic=..., nonconv=..., runs=...)
    Aggregation rule:
      sums logic/nonconv/runs across files (assumes files cover disjoint seeds/runs per p).
    """
    agg = defaultdict(lambda: defaultdict(lambda: {"logic": 0, "nonconv": 0, "runs": 0}))
    files = list(iter_json_files(folder))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {folder}")

    for fp in files:
        with fp.open("r") as f:
            data = json.load(f)

        results = data.get("results", {})
        for mode, entries in results.items():
            if mode not in MODE_LABELS:
                continue

            for e in entries:
                p = round(float(e["p"]), 6)
                agg[mode][p]["logic"] += int(e.get("logic", 0))
                agg[mode][p]["nonconv"] += int(e.get("nonconv", 0))
                agg[mode][p]["runs"] += int(e.get("runs", 0))

    return agg


def mode_curve(agg_for_mode: dict):
    """
    Input:
      agg_for_mode: dict[p] -> {logic, nonconv, runs}
    Output:
      ps_sorted, err_sorted
    """
    ps = sorted(agg_for_mode.keys())
    errs = []
    for p in ps:
        logic = agg_for_mode[p]["logic"]
        nonconv = agg_for_mode[p]["nonconv"]
        runs = agg_for_mode[p]["runs"]
        errs.append(float("nan") if runs <= 0 else (logic + nonconv) / runs)
    return ps, errs


def plot_datasets(datasets, title, out_path, logy=True, show=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    linestyles = {
        "1": (0, ()),
        "2": (0, (6, 2)),
        "3": (0, (2, 2)),
        "4": (0, (1, 2)),
        "6": (0, (6, 2, 1, 2)),
        "7": (0, (2, 2, 1, 2)),
    }

    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for i, (label, folder) in enumerate(datasets):
        agg = load_and_aggregate(folder)
        marker = markers[i % len(markers)]

        for mode in PLOT_MODES:
            if mode not in agg:
                continue
            ps, errs = mode_curve(agg[mode])
            if not ps:
                continue

            ax.plot(
                ps,
                errs,
                marker=marker,
                linestyle=linestyles.get(mode, (0, ())),
                label=f"{label} {MODE_LABELS[mode]}",
            )

    ax.set_xlabel("Channel parameter p")
    ax.set_ylabel("Logical error rate ( (logic + nonconv) / runs )")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.subplots_adjust(right=0.72)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize="small",
        frameon=True,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def discover_folders(base: Path):
    """All Sum_total_* result folders under base, sorted by name."""
    return sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("Sum_total_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=".",
        help="Path to the folder containing Sum_total_B1 / Sum_total_HGP2025 / etc.",
    )
    ap.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Which Sum_total_* folders to include (relative to --base). "
             "Default: every Sum_total_* folder found under --base.",
    )
    ap.add_argument("--out", default="decoder_comparison.png",
                    help="Output PNG path (default: decoder_comparison.png)")
    ap.add_argument("--show", action="store_true",
                    help="Also open an interactive window (needs a display)")
    ap.add_argument("--title", default="Decoder Comparison", help="Plot title")
    ap.add_argument("--no-logy", action="store_true", help="Disable log-scale y axis")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    names = args.folders if args.folders else discover_folders(base)
    if not names:
        raise FileNotFoundError(f"No Sum_total_* folders found under: {base}")

    datasets = []
    for name in names:
        folder = base / name
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        label = name.replace("Sum_total_", "").removesuffix(".npz")
        datasets.append((label, folder))

    plot_datasets(
        datasets,
        title=args.title,
        out_path=args.out,
        logy=(not args.no_logy),
        show=args.show,
    )


if __name__ == "__main__":
    main()
