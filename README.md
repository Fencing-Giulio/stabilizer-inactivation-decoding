# Stabilizer-Assisted Inactivation Decoding

Code and simulation data for the paper

> **Stabilizer-Assisted Inactivation Decoding of Quantum Error-Correcting Codes with Erasures**
> Giulio Pech, Mert Gökduman, Hanwen Yao, Henry D. Pfister (Duke University)
> *IEEE International Symposium on Information Theory (ISIT) 2026*; poster at QIP 2026
> [arXiv:2601.14236](https://arxiv.org/abs/2601.14236) · [IEEE Xplore](https://ieeexplore.ieee.org/document/11653654)

## Overview

Inactivation decoding achieves maximum-likelihood (ML) erasure decoding for
CSS / QLDPC codes by combining peeling with symbolic guesses, but its cost
grows with the number of inactivated variables. This decoder adds a
**dual peeling** step on the stabilizer matrix, guided by the known
(non-erased) qubits: known columns of weight 2 are eliminated by merging their
two rows, and rows with a single known qubit are used as pivots to peel that
qubit out of the other rows. Rows left with no support on the known qubits are
stabilizers supported entirely on the erased set. Each one lets us fix an
erased bit for free, since it only changes the correction by a stabilizer.
That removes a symbolic guess and shrinks the linear system solved at the end,
while the decoder stays ML.

Main results:
- dual peeling followed by inactivation matches ML performance with fewer
  symbolic guesses (more than 20% fewer for the B1 lifted-product code at
  high erasure rates);
- dual peeling plus standard peeling alone, with no inactivation, is already
  ML for surface codes.

## Repository layout

~~~
src/
  algorithm_sparse.py      sparse GF(2) matrix, peeling, dual peeling (structured_decode), inactivation
  decoder_sparse.py        SparseDecoder: all decoder variants ("modes")
  decoder.py               Monte Carlo driver for the erasure channel; writes JSON results
  plot.py                  logical failure rate vs erasure rate from result JSONs
  aggregate_histograms.py  merges guess-count histograms across many runs / array tasks
codes/                     parity-check matrices (Hx, Hz, logicals) as .npz
data/processed/            aggregated simulation results, one folder per code: B1, BB [[108]],
                           [[144]], [[288]], HGP2025, surface d=11, 13
scripts/
  plot_from_folders.py     plots logical failure rate for the codes in data/processed
  sbatch_stabilizer.sh     SLURM array job used for the large simulations
~~~

## Decoder modes

| Mode | Decoder | Label in plots |
|------|---------|----------------|
| 1 | Primal peeling only | Peeling |
| 2 | Inactivation decoding (ML baseline) | Inactivation |
| 3 | Dual peeling, then primal peeling (ML for surface codes; see the note below) | Peeling+Stab |
| 4 | Dual peeling, then inactivation (**stabilizer-assisted, ML**) | Stab+Inact |
| 5 | Alternate dual and primal peeling, then inactivation (no gain over mode 4; see Lemma 1 in the paper) | |
| 6 | Peeling with hard (random) guessing when stuck | Hard guessing |
| 7 | Dual peeling, then peeling with hard guessing (not in the paper) | Stab+Hard guessing |

## Code and paper terminology

The code predates the write-up, so some names differ from the paper:

| Paper | Code |
|-------|------|
| dual peeling (on `H_X`, guided by the known set) | `algorithm_sparse.structured_decode` |
| known-degree-1 row elimination (rule 2) | `algorithm_sparse.peel_left_side` |
| known-column weight-2 elimination (rule 1) | `algorithm_sparse.squeeze_multiweight_left_columns` |
| fixing a bit covered by a fully erased stabilizer | `algorithm_sparse.sweep_one_pivot` |
| primal peeling (on `H_Z`) | `algorithm_sparse.hz_peel_only` |
| inactivation decoding / symbolic guessing | `algorithm_sparse.inactivation_decoding` |
| Gaussian elimination core | `algorithm_sparse.solve_gf2_random` |
| number of stabilizer moves | `DecodeResult.stabilizer_guesses` |
| number of symbolic guesses (inactivations) | `DecodeResult.inactivation_guesses` |

`decoder_sparse.SparseDecoder.decode` assembles these into the modes above;
`decoder.py` is the Monte Carlo harness around it.

## A note on ML and tie-breaking

Modes 1 and 3 do not guess: they stop when peeling can no longer determine a
bit uniquely, and the harness records such a run as `nonconv`, counted as a
failure. Modes 2 and 4 use inactivation, which assigns the remaining free bits
arbitrarily and therefore always returns a correction.

The two conventions differ only on erasure patterns whose erased set supports a
logical operator. There the syndrome does not determine the logical class, so a
non-guessing decoder reports failure while a guessing decoder effectively flips
a coin and is right half the time. This is the only source of non-convergence
we observe for mode 3: on the d=13 surface code at p=0.40, mode 3 left erasures
in 15 runs out of 300, and in all 15 the nullity of `H_Z` restricted to the
erased set exceeded the number of independent fully erased X-stabilizers by
exactly one.

For a code with one logical qubit this costs a factor of two in the plotted
failure rate, with no change in threshold or slope. Across the surface-code data
in `data/processed/`, the ratio of mode 3 to mode 2 is 2.00 at every erasure
rate from p=0.22 to p=0.48. To make mode 3 agree with the inactivation-based
decoders, fix the remaining erased bits at the end to any syndrome-consistent
value instead of reporting non-convergence.

## Codes included

Surface codes (d = 11, 13, 17, 21, 25), bivariate bicycle codes
(n = 72, 90, 108, 144, 288, 360, 756), hypergraph-product codes
(n = 1600, 2025), and the B1 lifted-product code.

## Data

`data/processed/` holds the aggregated simulation results used for the plots
in the paper (one folder per code, listed above). It covers a subset of the
simulated codes; see the paper for the complete data.

## Usage

Requires Python 3.9+.

~~~bash
git clone https://github.com/Fencing-Giulio/stabilizer-inactivation-decoding.git
cd stabilizer-inactivation-decoding
python3 -m venv .venv && source .venv/bin/activate && pip install -q -r requirements.txt

# quick test (a few seconds): [[72,12]] BB code, ML inactivation (mode 2)
# vs stabilizer-assisted inactivation (mode 4); writes a JSON to results/
python src/decoder.py --npz codes/BB_n72_k12_l6_m6_Ax3_y1_y2_By3_x1_x2.npz \
    --ps 0.30 0.35 0.40 --runs 200 --modes 2 4 --results-dir results

# plot everything in results/
python src/plot.py --glob "results/*.json" --out results/threshold_plot.png

# logical failure rate vs erasure rate for the aggregated data in data/processed
# (defaults to every code found there; pass --folders for a readable legend)
python scripts/plot_from_folders.py --base data/processed \
    --folders Sum_total_Surface11 Sum_total_Surface13 \
    --title "Surface codes" --out surface_codes.png
~~~

On a SLURM cluster, `sbatch scripts/sbatch_stabilizer.sh` (from the repo root)
runs 200 array tasks with independent seeds. Merge their outputs with
`python src/aggregate_histograms.py --glob "results/*.json"`.

## Citation

~~~bibtex
@inproceedings{pech2026stabilizer,
  title     = {Stabilizer-Assisted Inactivation Decoding of Quantum Error-Correcting Codes with Erasures},
  author    = {Pech, Giulio and G{\"o}kduman, Mert and Yao, Hanwen and Pfister, Henry D.},
  booktitle = {IEEE International Symposium on Information Theory (ISIT)},
  year      = {2026},
  note      = {arXiv:2601.14236}
}
~~~

## Status

This is research code released as the artifact for the paper, not a maintained
library. It has not been fully cleaned up: there are unused parameters, some
duplication between the plotting scripts, and no test suite. The decoder modes
and the data in `data/processed/` are the parts that were used for the paper.

Simulations are seeded per erasure pattern, but the true error and the choice
among equally likely solutions use the global NumPy and `random` generators,
which are not seeded. Individual runs are therefore not bit-for-bit
reproducible; the published numbers come from millions of runs per point, where
this does not matter.
