#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# analyze.py — turns the raw Kaggle outputs in data/ into EVERY number, table
# and figure used in the paper.  Nothing in the paper is typed by hand.
#   inputs : data/v3_T4_kaggle/ (primitives + CNN workloads)
#            data/v4_T4_kaggle/ (transformer workloads)
#            data/v2_T4_kaggle/ (earlier independent session, reproducibility)
#            data/colab_v3_T4_partial_log.txt (second software stack, partial)
#   outputs: out/numbers.json, out/tab_*.tex, out/fig_*.pdf
# ─────────────────────────────────────────────────────────────────────────────
import os, re, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

D3, D4, D2 = "data/v3_T4_kaggle", "data/v4_T4_kaggle", "data/v2_T4_kaggle"
OUT = "out"
os.makedirs(OUT, exist_ok=True)
N = {}                                     # every number quoted in the paper


def wilson_hi(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c + h


def sci(x, d=1):
    """LaTeX scientific notation, e.g. 3.1\\times10^{-4}"""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{d}f}\\times10^{{{e}}}"


plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42})

# ── environment ──────────────────────────────────────────────────────────────
env3, env4 = json.load(open(f"{D3}/env.json")), json.load(open(f"{D4}/env.json"))
N["env"] = {k: env3[k] for k in ("gpu", "capability", "torch", "cuda_runtime", "cudnn", "python")}
N["env"]["driver"] = env3["driver_and_max_sm_clock"].split("\n")[0].split(",")[0]
N["env"]["max_sm_clock"] = env3["driver_and_max_sm_clock"].split("\n")[0].split(",")[1].strip()
assert all(env4[k] == env3[k] for k in ("gpu", "torch", "cudnn", "cuda_runtime"))
N["settings"] = env3["settings"]
N["runtime_min"] = {"v3": float(re.findall(r"total minutes: ([\d.]+)", open(f"{D3}/summary.txt").read())[0]),
                    "v4": float(re.findall(r"total minutes: ([\d.]+)", open(f"{D4}/summary.txt").read())[0])}

# ── STAGE 1: primitives ──────────────────────────────────────────────────────
s1 = pd.read_csv(f"{D3}/stage1_primitives.csv")
ordd = s1[s1["mode"] == "ordered"]
N["s1_control_min"] = float(s1.control_rerun_B1.min())
N["s1_shuffle_min"] = float(ordd.rows_bitident_shuffled_vs_ordered_sameB.min())
N["s1_n_batch_sizes"] = int(ordd.B.nunique())
dig = pd.read_csv(f"{D3}/digests_stage1.csv")
CUDNN_PRIMS = ["fir_conv1d", "conv1d_16to32", "conv2d_8to16"]
g = dig[dig.primitive.isin(CUDNN_PRIMS)].groupby(["primitive", "dtype", "B"])
N["s1_cudnn_digest_groups"] = int(g.ngroups)
N["s1_cudnn_groups_with_3cfgs"] = int((g.cfg.nunique() == 3).sum())
N["s1_cudnn_groups_cfg_dependent"] = int((g.sha256.nunique() > 1).sum())

PRIM_LABEL = {"fft1d_L256": "rFFT, $L{=}256$", "fft1d_L1024": "rFFT, $L{=}1024$",
              "fft1d_L4096": "rFFT, $L{=}4096$", "fft2d_64": "2-D rFFT, $64^2$",
              "fft2d_128": "2-D rFFT, $128^2$", "stft_logpow": "STFT log-power",
              "fir_conv1d": "FIR, 101 taps", "conv1d_16to32": "Conv1d $16{\\to}32$",
              "conv2d_8to16": "Conv2d $8{\\to}16$", "linear_256to64": "Linear $256{\\to}64$",
              "sum_reduce": "Row sum, 4096", "softmax": "Softmax, 1024",
              "layer_norm": "LayerNorm, 256"}
rows, prim = [], {}
for p in PRIM_LABEL:
    cells = [PRIM_LABEL[p]]
    for dt in ("fp32", "fp16"):
        x = ordd[(ordd.primitive == p) & (ordd.dtype == dt) & (ordd.cfg == "default")]
        mn = float(x.rows_bitident.min())
        dev = float(x.max_diff_over_scale.max())
        e64 = float(x.err_ref_vs_fp64.max())
        diffB = sorted(int(b) for b in x[x.rows_bitident < 1].B)
        prim[f"{p}|{dt}"] = {"min_rows_bitident": mn, "max_dev_rel": dev, "err_fp64_rel": e64,
                             "B_differ": diffB}
        if mn == 1.0:
            cells += ["1.000", "--"]
        else:
            cells += [f"{mn:.3f}", f"${sci(dev)}$"]
    rows.append(" & ".join(cells) + r" \\")
N["primitives"] = prim
N["s1_invariant_count"] = {dt: sum(1 for p in PRIM_LABEL if prim[f"{p}|{dt}"]["min_rows_bitident"] == 1)
                           for dt in ("fp32", "fp16")}
# regime boundaries = consecutive batch sizes whose outputs differ
reg = ordd[(ordd.cfg == "default") & (ordd.rows_bitident_vs_prevB < 1)]
N["s1_regime_boundaries"] = [f"{r.primitive}|{r.dtype}: {int(r.prevB)}->{int(r.B)}" for r in reg.itertuples()]
open(f"{OUT}/tab_primitives.tex", "w").write("\n".join(rows) + "\n")

# Fig 1 — fraction of rows identical to B=1 vs batch size (only primitives that vary)
fig, ax = plt.subplots(figsize=(3.4, 2.0))
var = [("linear_256to64", "fp32", "Linear, FP32", "o"), ("sum_reduce", "fp32", "Row sum, FP32", "s"),
       ("linear_256to64", "fp16", "Linear, FP16", "^"), ("conv2d_8to16", "fp16", "Conv2d, FP16", "D")]
for p, dt, lab, mk in var:
    x = ordd[(ordd.primitive == p) & (ordd.dtype == dt) & (ordd.cfg == "default")].sort_values("B")
    ax.plot(x.B, x.rows_bitident, marker=mk, ms=3.2 if dt == "fp32" and p == "linear_256to64" else 2.6,
            label=lab, zorder=4 if p == "linear_256to64" and dt == "fp32" else 3,
            mfc="none" if (p == "linear_256to64" and dt == "fp32") else None)
ax.set_xscale("log", base=2)
ax.set_xlabel("Batch size $B$")
ax.set_ylabel("Rows bit-identical\nto $B{=}1$")
ax.set_ylim(-0.04, 1.04)
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"])
ax.grid(alpha=0.3, lw=0.4)
ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1,
          columnspacing=1.0, handlelength=1.6)
fig.tight_layout(pad=0.2)
fig.savefig(f"{OUT}/fig_primitives.pdf")
plt.close(fig)

# ── STAGE 2: models ──────────────────────────────────────────────────────────
pairs = pd.concat([pd.read_csv(f"{D3}/stage2_pairs.csv"), pd.read_csv(f"{D4}/stage2_pairs.csv")])
perB = pd.concat([pd.read_csv(f"{D3}/stage2_perB.csv"), pd.read_csv(f"{D4}/stage2_perB.csv")])
ctrl = pd.concat([pd.read_csv(f"{D3}/stage2_control.csv"), pd.read_csv(f"{D4}/stage2_control.csv")])
shuf = pd.concat([pd.read_csv(f"{D3}/stage2_shuffle.csv"), pd.read_csv(f"{D4}/stage2_shuffle.csv")])
mcdf = pd.concat([pd.read_csv(f"{D3}/stage2_margin_cdf.csv"), pd.read_csv(f"{D4}/stage2_margin_cdf.csv")])

N["s2_pairs_total"] = int(len(pairs))
N["s2_pairs_default"] = int((pairs.cfg == "default").sum())
N["s2_bound_violations"] = int(pairs.flips_not_at_risk.sum())
N["s2_control_min"] = float(ctrl.rows_bitident_rerun.min())
N["s2_control_n"] = int(len(ctrl))
N["s2_shuffle_min"] = float(shuf.rows_bitident.min())
N["s2_shuffle_flips"] = int(shuf.flips.sum())
N["s2_shuffle_n"] = int(len(shuf))
N["s2_n_rows"] = int(perB[(perB.cfg == "default")].n_eff.max())
N["s2_batch_sizes"] = sorted(int(b) for b in perB[perB.cfg == "default"].B.unique())
# cudnn benchmark vs deterministic: identical?
k = ["workload", "dtype", "Ba", "Bb"]
jb = pairs[pairs.cfg == "cudnn_bench"].set_index(k).join(pairs[pairs.cfg == "cudnn_det"].set_index(k), rsuffix="_d")
N["s2_bench_vs_det_pairs"] = int(len(jb))
N["s2_bench_vs_det_identical"] = int(((jb.rows_bitident == jb.rows_bitident_d) & (jb.flips == jb.flips_d)
                                      & (jb.max_abs_diff == jb.max_abs_diff_d)).sum())

WL = [("cnn1d", "1-D CNN"), ("cnn2d", "2-D CNN"), ("tfm1d", "Transformer"),
      ("cnn1d_hard", "1-D CNN (B)"), ("cnn2d_hard", "2-D CNN (B)"), ("tfm1d_hard", "Transformer (B)")]
pdef = pairs[pairs.cfg == "default"]
bdef = perB[perB.cfg == "default"]
model, trows = {}, []
for wl, lab in WL:
    for dt in ("fp32", "fp16"):
        x = pdef[(pdef.workload == wl) & (pdef.dtype == dt)]
        y = bdef[(bdef.workload == wl) & (bdef.dtype == dt)]
        imax = x.flip_rate.idxmax() if x.flips.max() > 0 else None
        r = {"pairs": int(len(x)), "min_rows_bitident": float(x.rows_bitident.min()),
             "max_abs_dlogit": float(x.max_abs_diff.max()),
             "pairs_with_flip": int((x.flips > 0).sum()), "max_flips": int(x.flips.max()),
             "max_flip_rate": float(x.flip_rate.max()),
             "sum_flips": int(x.flips.sum()), "sum_expected": float(x.expected_flips.sum()),
             "sum_at_risk": int(x.at_risk.sum()),
             "fp64_flips_min": int(y.flips_vs_fp64.min()), "fp64_flips_max": int(y.flips_vs_fp64.max()),
             "fp64_n": int(y.n_vs_fp64.max()),
             "acc_min": float(y.acc.min()), "acc_max": float(y.acc.max()),
             "acc_fp64": float(y.acc_fp64_ref.iloc[0]),
             "c2w": int(x.correct_to_wrong.sum()), "w2c": int(x.wrong_to_correct.sum())}
        if x.flips.max() > 0:
            rr = x.loc[x.flip_rate.idxmax()]
            r["max_flip_pair"] = [int(rr.Ba), int(rr.Bb)]
            r["max_flip_pair_n"] = int(rr.n)
            r["spearman_pred_obs"] = float(spearmanr(x.expected_flips, x.flips)[0])
        else:
            nmax = int(x.n.max())
            r["zero_flip_upper95_rate"] = wilson_hi(0, nmax)
            r["zero_flip_n"] = nmax
        # max over pairs that do NOT involve B=8 (the outlier class in fp16 transformer)
        x8 = x[(x.Ba != 8) & (x.Bb != 8)]
        r["max_flips_excl_B8"] = int(x8.flips.max())
        r["max_flip_rate_excl_B8"] = float(x8.flip_rate.max())
        model[f"{wl}|{dt}"] = r
        pf = f"{r['pairs_with_flip']}/{r['pairs']}"
        mfr = "0" if r["max_flips"] == 0 else f"${sci(r['max_flip_rate'])}$"
        pred = (f"{r['sum_flips']}/{r['sum_expected']:.0f}" if dt == "fp16" else
                f"{r['sum_flips']}/{r['sum_expected']:.1f}")
        f64 = (f"{r['fp64_flips_min']}" if r['fp64_flips_min'] == r['fp64_flips_max']
               else f"{r['fp64_flips_min']}--{r['fp64_flips_max']}")
        trows.append(f"{lab if dt == 'fp32' else ''} & {dt.upper()} & {r['min_rows_bitident']:.3f} & "
                     f"${sci(r['max_abs_dlogit'])}$ & {pf} & {mfr} & {pred} & {f64} \\\\")
    if wl == "tfm1d":
        trows.append(r"\midrule")
N["models"] = model
open(f"{OUT}/tab_models.tex", "w").write("\n".join(trows) + "\n")

# totals used in the abstract
N["fp32_total_flips"] = int(pdef[pdef.dtype == "fp32"].flips.sum())
N["fp32_pairs"] = int((pdef.dtype == "fp32").sum())
N["fp32_row_comparisons"] = int(pdef[pdef.dtype == "fp32"].n.sum())
N["fp16_row_comparisons"] = int(pdef[pdef.dtype == "fp16"].n.sum())
N["fp32_min_rows_bitident_models"] = float(pdef[pdef.dtype == "fp32"].rows_bitident.min())
N["fp32_max_abs_dlogit"] = float(pdef[pdef.dtype == "fp32"].max_abs_diff.max())
N["fp16_max_abs_dlogit"] = float(pdef[pdef.dtype == "fp16"].max_abs_diff.max())
f16 = pdef[(pdef.dtype == "fp16") & (pdef.at_risk > 0)]
N["pred_spearman_fp16_all"] = float(spearmanr(pdef[pdef.dtype == "fp16"].expected_flips,
                                              pdef[pdef.dtype == "fp16"].flips)[0])
N["pairs_at_risk_zero"] = int((pdef.at_risk == 0).sum())
N["flips_over_at_risk_fp16"] = float(pdef[pdef.dtype == "fp16"].flips.sum() / pdef[pdef.dtype == "fp16"].at_risk.sum())
N["margin_cdf"] = mcdf.to_dict(orient="records")

# Fig 2 — transformer FP16: lower triangle rows identical, upper triangle flips per 1e5 rows
x = pdef[(pdef.workload == "tfm1d") & (pdef.dtype == "fp16")]
Bs = sorted(set(x.Ba) | set(x.Bb))
ix = {b: i for i, b in enumerate(Bs)}
Mb = np.full((len(Bs), len(Bs)), np.nan)
Mf = np.full((len(Bs), len(Bs)), np.nan)
for r in x.itertuples():
    i, j = ix[r.Ba], ix[r.Bb]
    Mb[j, i] = r.rows_bitident             # lower
    Mf[i, j] = r.flips / r.n * 1e5         # upper
fig = plt.figure(figsize=(3.4, 3.55))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.045], hspace=0.5, wspace=0.3,
                      left=0.13, right=0.97, top=0.985, bottom=0.10)
ax = fig.add_subplot(gs[0, :])
im1 = ax.imshow(Mb, cmap="Blues", vmin=0, vmax=1, origin="upper")
im2 = ax.imshow(Mf, cmap="Reds", vmin=0, vmax=np.nanmax(Mf), origin="upper")
for i in range(len(Bs)):
    ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, color="0.85", lw=0))
ax.set_xticks(range(len(Bs)))
ax.set_yticks(range(len(Bs)))
ax.set_xticklabels(Bs, rotation=90, fontsize=6)
ax.set_yticklabels(Bs, fontsize=6)
ax.set_xlabel("Batch size")
ax.set_ylabel("Batch size")
c1 = fig.colorbar(im1, cax=fig.add_subplot(gs[1, 0]), orientation="horizontal")
c1.set_label("Rows identical (lower)", fontsize=6.5)
c1.ax.tick_params(labelsize=6)
c2 = fig.colorbar(im2, cax=fig.add_subplot(gs[1, 1]), orientation="horizontal")
c2.set_label("Flips per $10^5$ rows (upper)", fontsize=6.5)
c2.ax.tick_params(labelsize=6)
fig.savefig(f"{OUT}/fig_pairs_tfm_fp16.pdf")
plt.close(fig)
# equivalence classes (batch sizes whose outputs are bit-identical on every row)
classes = []
for dt in ("fp16", "fp32"):
    xx = pdef[(pdef.workload == "tfm1d") & (pdef.dtype == dt)]
    parent = {b: b for b in Bs}

    def find(b):
        while parent[b] != b:
            b = parent[b]
        return b
    for r in xx[xx.rows_bitident == 1].itertuples():
        parent[find(r.Bb)] = find(r.Ba)
    cl = {}
    for b in Bs:
        cl.setdefault(find(b), []).append(int(b))
    N[f"tfm_{dt}_classes"] = sorted(cl.values())
x8 = x[(x.Ba == 8) | (x.Bb == 8)]
N["tfm_fp16_B8_max_rows_bitident"] = float(x8.rows_bitident.max())

# Fig 3 — predicted vs observed flips per pair (FP16, pairs with >=1 at-risk row)
fig, ax = plt.subplots(figsize=(3.4, 2.15))
mk = {"cnn1d_hard": ("o", "1-D CNN (B)"), "cnn2d_hard": ("s", "2-D CNN (B)"),
      "tfm1d": ("^", "Transformer"), "tfm1d_hard": ("v", "Transformer (B)"),
      "cnn1d": ("x", "1-D CNN")}
for wl, (m, lab) in mk.items():
    z = pdef[(pdef.workload == wl) & (pdef.dtype == "fp16") & (pdef.at_risk > 0)]
    if len(z):
        ax.scatter(z.expected_flips, z.flips, s=7, marker=m, label=lab, lw=0.6, alpha=0.75)
lim = [0, 130]
ax.plot(lim, lim, "k--", lw=0.6)
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Predicted flips per pair (Eq. 1)")
ax.set_ylabel("Observed flips per pair")
ax.grid(alpha=0.3, lw=0.4)
ax.legend(frameon=False, loc="upper left", fontsize=6.5, handletextpad=0.2)
fig.tight_layout(pad=0.2)
fig.savefig(f"{OUT}/fig_predictor.pdf")
plt.close(fig)
N["fig3_pairs_plotted"] = int(len(pdef[(pdef.dtype == "fp16") & (pdef.at_risk > 0)]))

# ── STAGE 3: padding ─────────────────────────────────────────────────────────
s3 = pd.concat([pd.read_csv(f"{D3}/stage3_padding.csv"), pd.read_csv(f"{D4}/stage3_padding.csv")])
s3["ratio"] = s3.rows_per_s_padded / s3.rows_per_s_unpadded
N["s3_padded_min_bitident"] = float(s3.padded_rows_bitident_vs_Mbatch.min())
N["s3_padded_flips"] = int(s3.padded_flips_vs_Mbatch.sum())
N["s3_unpadded_flips"] = int(s3.unpadded_flips_vs_Mbatch.sum())
N["s3_rows"] = int(s3.n.max())
N["s3_configs"] = int(len(s3))
pad, prow = {}, []
for wl, lab in WL[:3]:
    for dt in ("fp32", "fp16"):
        cells = [lab if dt == "fp32" else "", dt.upper()]
        ratios = []
        for M in (64, 256):
            z = s3[(s3.workload == wl) & (s3.dtype == dt) & (s3.M == M)]
            pad[f"{wl}|{dt}|M{M}"] = {"unpadded_min_bitident": float(z.unpadded_rows_bitident_vs_Mbatch.min()),
                                     "unpadded_flips": int(z.unpadded_flips_vs_Mbatch.sum()),
                                     "ratio_min": float(z.ratio.min()), "ratio_max": float(z.ratio.max()),
                                     "ratio_mean": float(z.ratio.mean())}
            q = pad[f"{wl}|{dt}|M{M}"]
            cells += [f"{q['unpadded_min_bitident']:.3f}"]
            ratios.append(f"{q['ratio_min']:.2f}--{q['ratio_max']:.2f}")
        cells += [str(sum(pad[f"{wl}|{dt}|M{M}"]["unpadded_flips"] for M in (64, 256)))] + ratios
        prow.append(" & ".join(cells) + r" \\")
N["padding"] = pad
for M in (64, 256):
    z = s3[s3.M == M]
    N[f"s3_ratio_M{M}"] = {"min": float(z.ratio.min()), "max": float(z.ratio.max()), "mean": float(z.ratio.mean())}
open(f"{OUT}/tab_padding.tex", "w").write("\n".join(prow) + "\n")

# throughput context (FP32, default) at B=1 and B=1024
thr = {}
for wl, _ in WL[:3]:
    z = bdef[(bdef.workload == wl) & (bdef.dtype == "fp32")].set_index("B").rows_per_s
    thr[wl] = {"B1": float(z.loc[1]), "B1024": float(z.loc[1024]), "max": float(z.max()),
               "argmax": int(z.idxmax())}
N["throughput_fp32"] = thr

# ── reproducibility checks ───────────────────────────────────────────────────
d2, d3 = pd.read_csv(f"{D2}/digests_stage1.csv"), pd.read_csv(f"{D3}/digests_stage1.csv")
mm = d2.merge(d3, on=["primitive", "dtype", "cfg", "B"], suffixes=("_a", "_b"))
N["repro_digests_compared"] = int(len(mm))
N["repro_digests_equal"] = int((mm.sha256_a == mm.sha256_b).sum())
xa, xb = np.load(f"{D2}/xgpu_logits_TeslaT4.npz"), np.load(f"{D3}/xgpu_logits_TeslaT4.npz")
com = [k for k in xa.files if k in xb.files]
N["repro_logit_arrays_compared"] = len(com)
N["repro_logit_arrays_equal"] = int(sum(np.array_equal(xa[k], xb[k]) for k in com))
N["repro_logit_rows_each"] = int(xa[com[0]].shape[0])


def parse(path):
    out = {}
    for l in open(path):
        m = re.match(r"\s+(\S+\|\S+\|\S+): control=([\d.]+)\s+worst rows_bitident vs B=1: ([\d.]+)", l)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3))
        m = re.match(r"\s+fp64 reference done: accuracy on (\d+) rows = ([\d.]+)", l)
        if m:
            out.setdefault("fp64", []).append(m.group(2))
    return out


ca, kb = parse("data/colab_v3_T4_partial_log.txt"), parse(f"{D3}/log.txt")
keys = [k for k in ca if k != "fp64"]
cenv = json.loads(open("data/colab_v3_T4_partial_log.txt").readline()[4:])
N["colab"] = {"torch": cenv["torch"], "cudnn": cenv["cudnn"],
              "driver": cenv["driver_and_max_sm_clock"].split(",")[0],
              "stage1_lines": len(keys), "stage1_lines_equal": int(sum(ca[k] == kb.get(k) for k in keys)),
              "fp64_acc_equal": ca["fp64"] == kb["fp64"][:len(ca["fp64"])], "fp64_acc": ca["fp64"]}

json.dump(N, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: N[k] for k in N if k not in ("primitives", "models", "margin_cdf", "padding", "settings")},
                 indent=1, default=float))

# ── fraction of rows that could ever flip given the worst observed FP16 perturbation ──
cert = {}
for v, wls in ((D3, ["cnn1d", "cnn2d", "cnn1d_hard", "cnn2d_hard"]), (D4, ["tfm1d", "tfm1d_hard"])):
    xz = np.load(f"{v}/xgpu_logits_TeslaT4.npz")
    for wl in wls:
        a = xz[f"{wl}|fp16|default|B1"].astype(np.float64)
        s = np.sort(a, 1)
        m = s[:, -1] - s[:, -2]
        dmax = N["models"][f"{wl}|fp16"]["max_abs_dlogit"]
        cert[wl] = {"n": int(len(m)), "two_delta": 2 * dmax,
                    "frac_margin_le_2delta": float((m <= 2 * dmax).mean()),
                    "frac_exact_tie": float((m == 0).mean())}
N["fp16_certifiable"] = cert
json.dump(N, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
print("certifiable", json.dumps(cert, indent=1))
