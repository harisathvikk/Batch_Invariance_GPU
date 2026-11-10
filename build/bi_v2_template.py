#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Batch invariance of GPU signal / image pipelines  —  v2 (paper-grade run)
# ─────────────────────────────────────────────────────────────────────────────
# QUESTION: does the batch a signal is processed in (size, position, neighbours)
#           change its output bits, its error, or its final decision?
#
# STAGE 1  primitive survey   FFT1D/2D, STFT, FIR, conv1d/conv2d, GEMM, sum,
#                             softmax, layernorm  x  21 batch sizes x fp32/fp16
# STAGE 2  models at scale    two small CNNs (1D tones, 2D gratings) on ~262k
#                             inputs: pairwise bit-equality between ALL batch
#                             sizes, decision flips, at-risk bound, flip predictor
# STAGE 3  mitigation         pad every request to a fixed batch M -> is it
#                             invariant, and what does it cost?
#
# Models are TRAINED ONCE on CPU and their weights are embedded at the bottom
# of this file, so every GPU runs the *identical* model on *identical* inputs
# (all inputs come from the CPU RNG).  No internet, no dataset, no download.
#
# Output folder: bi_v2_<GPU>/   (+ .zip)  ->  CSVs, env.json, summary.txt, log.txt
# Control knobs (env vars):  BI_QUICK=1  (2-minute smoke test)
#                            BI_STAGES=1,2,3  (which stages to run)
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, io, re, json, time, copy, math, base64, hashlib, shutil
import platform, subprocess, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = "bi_v4"

# ── embedded weights (filled in by build_final.py) ───────────────────────────
WEIGHTS_B64 = ""  # __WEIGHTS_B64__

# ── settings ─────────────────────────────────────────────────────────────────
QUICK = os.environ.get("BI_QUICK", "0") == "1"
TRAIN_ONLY = os.environ.get("BI_TRAIN_ONLY", "0") == "1"
STAGES = set(os.environ.get("BI_STAGES", "1,2,3").replace(" ", "").split(","))
SEED = 1234
WL_FILTER = [w for w in os.environ.get("BI_WL", "").replace(" ", "").split(",") if w]
SIG_TONE = float(os.environ.get("BI_SIG_TONE", "2.0"))     # noise level, tones
SIG_GRAT = float(os.environ.get("BI_SIG_GRAT", "1.5"))     # noise level, gratings

if QUICK:
    N_SIG = 256
    B_S1 = [2, 3, 8, 16, 32, 64, 128]
    N_L, N64, XG_N = 4096, 1024, 512
    B_S2 = [1, 2, 3, 8, 16, 32, 64, 256]
    CAPS = {1: 512, 2: 1024, 3: 1024}
    N_RED, CAPS_RED, B_RED = 1024, {1: 256, 2: 512}, [1, 2, 8, 32, 256]
    SHUF_B = [16, 64]
    N_PAD, PAD_M, PAD_REQ = 512, [32, 128], [1, 4, 16]
else:
    N_SIG = 2048
    B_S1 = [2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 100, 127, 128, 129,
            255, 256, 512, 1024]
    N_L, N64, XG_N = 262144, 65536, 16384
    B_S2 = [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 100, 128, 256, 512, 1024]
    CAPS = {1: 65536, 2: 131072, 3: 131072, 4: 131072}   # slow small batches: fewer rows
    N_RED, CAPS_RED, B_RED = 65536, {1: 16384, 2: 32768}, [1, 2, 8, 16, 32, 64, 256, 1024]
    SHUF_B = [16, 64, 256, 1024]
    N_PAD, PAD_M, PAD_REQ = 16384, [64, 256], [1, 2, 4, 8, 16, 32]

# ── output folder / logging (filled in main) ─────────────────────────────────
OUT = None
LOG = None
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG is not None:
        LOG.write(s + "\n")
        LOG.flush()


def el():
    return f"{(time.time() - T0) / 60:.1f} min"


# ── device ───────────────────────────────────────────────────────────────────
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_CUDA = dev.type == "cuda"
cap = torch.cuda.get_device_capability(0) if HAS_CUDA else (0, 0)


def sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def set_flags(cfg):
    torch.backends.cudnn.benchmark = cfg["bench"]
    torch.backends.cudnn.deterministic = cfg["det"]
    torch.backends.cuda.matmul.allow_tf32 = cfg["tf32"]
    torch.backends.cudnn.allow_tf32 = cfg["tf32"]


CFGS = [
    {"name": "default", "bench": False, "det": False, "tf32": False},
    {"name": "cudnn_bench", "bench": True, "det": False, "tf32": False},
    {"name": "cudnn_det", "bench": False, "det": True, "tf32": False},
]
if cap[0] >= 8:   # TF32 exists only on Ampere or newer
    CFGS.append({"name": "tf32", "bench": False, "det": False, "tf32": True})
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


# ─────────────────────────────────────────────────────────────────────────────
# models + synthetic data (all generated on CPU -> identical on every machine)
# ─────────────────────────────────────────────────────────────────────────────
class Net1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(1, 8, 9, 2, 4), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 16, 9, 2, 4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 9, 2, 4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 4))

    def forward(self, x):          # x: [B, 1024]
        return self.f(x.unsqueeze(1))


class Net2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 4))

    def forward(self, x):          # x: [B, 32, 32]
        return self.f(x.unsqueeze(1))


class TFMBlock(nn.Module):
    """Pre-norm transformer block written with plain matmul + softmax (no fused kernels)."""
    def __init__(self, d=32, h=4, ff=64):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x):                     # x: [B, T, d]
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = torch.softmax((q @ k.transpose(-1, -2)) * (d // self.h) ** -0.5, -1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, T, d))
        return x + self.f2(F.gelu(self.f1(self.n2(x))))


class TFM1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(16, 32)
        self.pos = nn.Parameter(torch.zeros(64, 32))
        self.blocks = nn.Sequential(TFMBlock(), TFMBlock())
        self.nf = nn.LayerNorm(32)
        self.head = nn.Linear(32, 4)

    def forward(self, x):                     # x: [B, 1024] -> 64 tokens of 16 samples
        z = self.emb(x.reshape(x.shape[0], 64, 16)) + self.pos
        return self.head(self.nf(self.blocks(z)).mean(1))


def make_tones(n, seed, L=1024):
    """4 classes of noisy sinusoids that differ by (jittered) frequency."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)
    f = torch.tensor([40.0, 46.0, 52.0, 58.0])[y] + torch.randn(n, generator=g) * 2.0
    ph = torch.rand(n, generator=g) * 2 * math.pi
    t = torch.arange(L, dtype=torch.float32) / L
    x = torch.sin(2 * math.pi * f[:, None] * t[None, :] + ph[:, None])
    x = x + SIG_TONE * torch.randn(n, L, generator=g)
    return (x / 2.0).float(), y


def make_gratings(n, seed, S=32):
    """4 classes of oriented gratings (0/45/90/135 deg, jittered) in noise."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)
    th = y.float() * (math.pi / 4) + torch.randn(n, generator=g) * 0.20
    k = 0.08 + torch.rand(n, generator=g) * 0.08
    ph = torch.rand(n, generator=g) * 2 * math.pi
    ax = torch.arange(S, dtype=torch.float32) - (S - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    arg = 2 * math.pi * k[:, None, None] * (xx[None] * torch.cos(th)[:, None, None]
                                            + yy[None] * torch.sin(th)[:, None, None]) + ph[:, None, None]
    x = torch.cos(arg) + SIG_GRAT * torch.randn(n, S, S, generator=g)
    return (x / 2.0).float(), y


def gen_chunked(maker, n, seed, chunk=32768):
    xs, ys = [], []
    for i, s in enumerate(range(0, n, chunk)):
        m = min(chunk, n - s)
        x, y = maker(m, seed * 1000003 + i)
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def make_tones_hard(n, seed, L=1024, centers=(43.0, 47.7, 53.5)):
    """Stress set: tone frequency sits next to a class boundary of the trained model (43, 47.7 or 53.5 Hz),
    so many inputs have a near-tied decision (high margin density)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)             # label unused for flips
    c = torch.tensor(list(centers))[torch.randint(0, 3, (n,), generator=g)]   # boundaries of the trained model
    f = c + torch.randn(n, generator=g) * 0.15
    ph = torch.rand(n, generator=g) * 2 * math.pi
    t = torch.arange(L, dtype=torch.float32) / L
    x = torch.sin(2 * math.pi * f[:, None] * t[None, :] + ph[:, None])
    x = x + SIG_TONE * torch.randn(n, L, generator=g)
    return (x / 2.0).float(), y


def make_gratings_hard(n, seed, S=32):
    """Stress set: grating orientation sits next to a class boundary (22.5/67.5/112.5 deg)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)
    c = (torch.tensor([0.5, 1.5, 2.5])[torch.randint(0, 3, (n,), generator=g)]) * (math.pi / 4)
    th = c + torch.randn(n, generator=g) * 0.04
    k = 0.08 + torch.rand(n, generator=g) * 0.08
    ph = torch.rand(n, generator=g) * 2 * math.pi
    ax = torch.arange(S, dtype=torch.float32) - (S - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    arg = 2 * math.pi * k[:, None, None] * (xx[None] * torch.cos(th)[:, None, None]
                                            + yy[None] * torch.sin(th)[:, None, None]) + ph[:, None, None]
    x = torch.cos(arg) + SIG_GRAT * torch.randn(n, S, S, generator=g)
    return (x / 2.0).float(), y


TFM_CENTERS = (43.0, 47.7, 53.5)   # __TFM_CENTERS__ (set by build step from a scan of the trained transformer)


def make_tones_hard_tfm(n, seed, L=1024):
    return make_tones_hard(n, seed, L, TFM_CENTERS)


MODEL_DEFS = {"cnn1d": (Net1D, make_tones), "cnn2d": (Net2D, make_gratings), "tfm1d": (TFM1D, make_tones)}
# Stage-2 workloads: name -> (model key, input maker).  "_hard" = same model, boundary inputs.
WORKLOADS = {"cnn1d": ("cnn1d", make_tones), "cnn2d": ("cnn2d", make_gratings),
             "cnn1d_hard": ("cnn1d", make_tones_hard), "cnn2d_hard": ("cnn2d", make_gratings_hard),
             "tfm1d": ("tfm1d", make_tones), "tfm1d_hard": ("tfm1d", make_tones_hard_tfm)}
if WL_FILTER:
    WORKLOADS = {k: v for k, v in WORKLOADS.items() if k in WL_FILTER}
ACTIVE_MODELS = sorted({v[0] for v in WORKLOADS.values()})


def train_model(kind):
    """Deterministic CPU training (used by build step; fallback if no weights)."""
    cls, maker = MODEL_DEFS[kind]
    torch.manual_seed(SEED)
    ntr = 4000 if QUICK else 40000
    ep = 1 if QUICK else 6
    X, y = gen_chunked(maker, ntr, 777)
    m = cls()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    gcpu = torch.Generator().manual_seed(SEED)
    for e in range(ep):
        m.train()
        perm = torch.randperm(ntr, generator=gcpu)
        tot = 0.0
        for i in range(0, ntr, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = F.cross_entropy(m(X[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        log(f"  [train {kind}] epoch {e + 1}/{ep} loss {tot / ntr:.4f}")
    m.eval()
    Xv, yv = gen_chunked(maker, 8192, 4242)
    with torch.no_grad():
        acc = (m(Xv).argmax(1) == yv).double().mean().item()
    log(f"  [train {kind}] validation accuracy {acc:.4f}")
    return m, acc


def pack_state(models):
    arrays = {}
    for name, m in models.items():
        for k, v in m.state_dict().items():
            arrays[f"{name}__{k}"] = v.cpu().numpy()
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode()


def unpack_state(b64):
    z = np.load(io.BytesIO(base64.b64decode("".join(b64.split()))))
    out = {}
    for key in z.files:
        name, k = key.split("__", 1)
        out.setdefault(name, {})[k] = torch.from_numpy(z[key].copy())
    return out


def load_models():
    ms, src = {}, "embedded"
    if WEIGHTS_B64.strip():
        st = unpack_state(WEIGHTS_B64)
        for kind, (cls, _) in MODEL_DEFS.items():
            if kind not in ACTIVE_MODELS:
                continue
            m = cls()
            m.load_state_dict(st[kind])
            ms[kind] = m.eval()
    else:
        src = "trained-locally-on-cpu"
        log("WARNING: no embedded weights -> training on CPU here. Weights will NOT be "
            "bit-identical across machines, so cross-GPU bit comparison is invalid.")
        for kind in MODEL_DEFS:
            ms[kind], _ = train_model(kind)
    return ms, src


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_batched(fn, X, B, n=None):
    """Run fn over X[:n] in chunks of exactly B rows. The remainder rows
    (n mod B) are DROPPED so that every compared row really ran at batch size B."""
    n = X.shape[0] if n is None else min(n, X.shape[0])
    n_eff = (n // B) * B
    if n_eff == 0:
        return None
    return torch.cat([fn(X[i:i + B]) for i in range(0, n_eff, B)], 0)


@torch.no_grad()
def timed_run(fn, X, B, n=None):
    n = X.shape[0] if n is None else min(n, X.shape[0])
    if (n // B) * B == 0:
        return None, float("nan")
    for _ in range(2):                      # warm-up (cudnn autotune, plan caches)
        fn(X[:B])
    sync()
    t = time.perf_counter()
    out = run_batched(fn, X, B, n)
    sync()
    dt = time.perf_counter() - t
    return out, out.shape[0] / dt


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def compare_arrays(ref, out, f64=None):
    n = min(ref.shape[0], out.shape[0])
    r = ref[:n].reshape(n, -1)
    o = out[:n].reshape(n, -1)
    eq = (r == o)
    rd, od = r.double(), o.double()
    diff = (rd - od).abs()
    scale = rd.abs().max().item() + 1e-30
    rec = {"n_compared": n,
           "rows_bitident": eq.all(1).double().mean().item(),
           "elems_bitident": eq.double().mean().item(),
           "max_abs_diff": diff.max().item(),
           "max_diff_over_scale": diff.max().item() / scale,
           "mean_abs_diff": diff.mean().item()}
    if f64 is not None:
        f = f64[:n].reshape(n, -1).to(rd.device)
        rec["err_ref_vs_fp64"] = (rd - f).abs().max().item() / scale
        rec["err_out_vs_fp64"] = (od - f).abs().max().item() / scale
    return rec


def save_csv(name, rows):
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, name), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — primitive survey
# ─────────────────────────────────────────────────────────────────────────────
def gen(shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def prim_fft1d(L):
    def build():
        X = gen((N_SIG, L), SEED + L)
        f = lambda x: torch.view_as_real(torch.fft.rfft(x, dim=-1))
        return dict(X=X, make=lambda dt: f, f64=f, cudnn=False)
    return build


def prim_fft2d(S):
    def build():
        X = gen((max(N_SIG // 2, 64), S, S), SEED + S)
        f = lambda x: torch.view_as_real(torch.fft.rfft2(x))
        return dict(X=X, make=lambda dt: f, f64=f, cudnn=False)
    return build


def prim_stft(L=4096, n_fft=512, hop=128):
    def build():
        X = gen((N_SIG, L), SEED + 11)

        def make(dt):
            win = torch.hann_window(n_fft, dtype=dt, device=dev)

            def fn(x):
                S = torch.stft(x, n_fft, hop_length=hop, window=win, return_complex=True)
                p = S.real.float() ** 2 + S.imag.float() ** 2     # power in fp32
                return torch.log10(p.clamp_min(1e-10))            # (v1 bug: log in fp16)
            return fn

        def f64(x):
            win = torch.hann_window(n_fft, dtype=torch.float64)
            S = torch.stft(x, n_fft, hop_length=hop, window=win, return_complex=True)
            return torch.log10((S.real ** 2 + S.imag ** 2).clamp_min(1e-10))
        return dict(X=X, make=make, f64=f64, cudnn=False)
    return build


def prim_fir(L=4096, taps=101):
    def build():
        X = gen((N_SIG, L), SEED + 21)
        w = gen((1, 1, taps), SEED + 22)
        w = w / w.abs().sum()

        def make(dt):
            wd = w.to(dev, dt)
            return lambda x: F.conv1d(x.unsqueeze(1), wd, padding=taps // 2).squeeze(1)
        f64 = lambda x: F.conv1d(x.unsqueeze(1), w.double(), padding=taps // 2).squeeze(1)
        return dict(X=X, make=make, f64=f64, cudnn=True)
    return build


def prim_conv1d_multi():
    def build():
        X = gen((N_SIG, 16, 512), SEED + 31)
        w = gen((32, 16, 9), SEED + 32) * 0.1
        b = gen((32,), SEED + 33)

        def make(dt):
            wd, bd = w.to(dev, dt), b.to(dev, dt)
            return lambda x: F.conv1d(x, wd, bd, padding=4)
        f64 = lambda x: F.conv1d(x, w.double(), b.double(), padding=4)
        return dict(X=X, make=make, f64=f64, cudnn=True)
    return build


def prim_conv2d():
    def build():
        X = gen((N_SIG, 8, 32, 32), SEED + 41)
        w = gen((16, 8, 3, 3), SEED + 42) * 0.2
        b = gen((16,), SEED + 43)

        def make(dt):
            wd, bd = w.to(dev, dt), b.to(dev, dt)
            return lambda x: F.conv2d(x, wd, bd, padding=1)
        f64 = lambda x: F.conv2d(x, w.double(), b.double(), padding=1)
        return dict(X=X, make=make, f64=f64, cudnn=True)
    return build


def prim_linear():
    def build():
        X = gen((N_SIG, 256), SEED + 51)
        w = gen((64, 256), SEED + 52) * 0.0625
        b = gen((64,), SEED + 53)

        def make(dt):
            wd, bd = w.to(dev, dt), b.to(dev, dt)
            return lambda x: F.linear(x, wd, bd)
        f64 = lambda x: F.linear(x, w.double(), b.double())
        return dict(X=X, make=make, f64=f64, cudnn=False)
    return build


def prim_simple(name):
    def build():
        if name == "sum_reduce":
            X = gen((N_SIG, 4096), SEED + 61)
            f = lambda x: x.sum(-1)
        elif name == "softmax":
            X = gen((N_SIG, 1024), SEED + 62) * 3
            f = lambda x: torch.softmax(x, -1)
        else:
            X = gen((N_SIG, 256), SEED + 63)
            f = lambda x: F.layer_norm(x, (256,))
        return dict(X=X, make=lambda dt: f, f64=f, cudnn=False)
    return build


PRIMS = {
    "fft1d_L256": prim_fft1d(256), "fft1d_L1024": prim_fft1d(1024),
    "fft1d_L4096": prim_fft1d(4096), "fft2d_64": prim_fft2d(64),
    "fft2d_128": prim_fft2d(128), "stft_logpow": prim_stft(),
    "fir_conv1d": prim_fir(), "conv1d_16to32": prim_conv1d_multi(),
    "conv2d_8to16": prim_conv2d(), "linear_256to64": prim_linear(),
    "sum_reduce": prim_simple("sum_reduce"), "softmax": prim_simple("softmax"),
    "layer_norm": prim_simple("layer_norm"),
}
if QUICK:
    PRIMS = {k: PRIMS[k] for k in ["fft1d_L1024", "stft_logpow", "conv1d_16to32",
                                    "linear_256to64", "softmax"]}


def stage1():
    log(f"\n######## STAGE 1: primitive survey ({el()}) ########")
    rows, digs = [], []
    gshuf = torch.Generator().manual_seed(SEED + 5)
    for pname, build in PRIMS.items():
        log(f"\n=== primitive {pname} ({el()}) ===")
        try:
            p = build()
        except Exception:
            log("BUILD FAILED", pname, traceback.format_exc())
            continue
        X = p["X"]
        N = X.shape[0]
        try:
            f64 = p["f64"](X.double())
        except Exception:
            f64 = None
            log("fp64 reference failed", traceback.format_exc())
        cfgs = CFGS if p["cudnn"] else CFGS[:1]
        for dname, dt in DTYPES.items():
            for cfg in cfgs:
                tag = f"{pname}|{dname}|{cfg['name']}"
                try:
                    set_flags(cfg)
                    Xd = X.to(dev, dt)
                    fn = p["make"](dt)
                    timed = cfg["name"] == "default"
                    if timed:
                        ref, rps1 = timed_run(fn, Xd, 1)
                    else:
                        ref, rps1 = run_batched(fn, Xd, 1), float("nan")
                    if not torch.isfinite(ref.float()).all():
                        log(f"  {tag}: non-finite reference - skipped")
                        continue
                    ctrl = (run_batched(fn, Xd, 1).reshape(N, -1) == ref.reshape(N, -1)).all(1).double().mean().item()
                    f64d = f64.to(dev) if f64 is not None else None
                    base = {"primitive": pname, "dtype": dname, "cfg": cfg["name"],
                            "control_rerun_B1": ctrl}
                    rows.append({**base, "B": 1, "mode": "ref", "rows_per_s": rps1,
                                 "n_compared": N})
                    digs.append({**base, "B": 1, "n": N, "sha256": sha(ref)})
                    prev, prevB, worst = None, None, 1.0
                    for B in B_S1:
                        if timed:
                            out, rps = timed_run(fn, Xd, B)
                        else:
                            out, rps = run_batched(fn, Xd, B), float("nan")
                        if out is None:
                            continue
                        rec = compare_arrays(ref, out, f64d)
                        rec.update({**base, "B": B, "mode": "ordered", "rows_per_s": rps})
                        if prev is not None:
                            rec["prevB"] = prevB
                            rec["rows_bitident_vs_prevB"] = compare_arrays(prev, out)["rows_bitident"]
                        nB = out.shape[0]               # shuffled order: position + neighbours change
                        pm = torch.randperm(nB, generator=gshuf).to(dev)
                        so = run_batched(fn, Xd[:nB][pm], B)
                        back = torch.empty_like(so)
                        back[pm] = so
                        rec["rows_bitident_shuffled_vs_ordered_sameB"] = (
                            back.reshape(nB, -1) == out.reshape(nB, -1)).all(1).double().mean().item()
                        rows.append(rec)
                        if B == B_S1[-1]:
                            digs.append({**base, "B": B, "n": nB, "sha256": sha(out)})
                        worst = min(worst, rec["rows_bitident"])
                        prev, prevB = out, B
                    log(f"  {tag}: control={ctrl:.3f}  worst rows_bitident vs B=1: {worst:.4f}")
                except Exception:
                    log(f"  {tag}: FAILED\n" + traceback.format_exc())
                finally:
                    if HAS_CUDA:
                        torch.cuda.empty_cache()
        save_csv("stage1_primitives.csv", rows)
        save_csv("digests_stage1.csv", digs)
    save_csv("stage1_primitives.csv", rows)
    save_csv("digests_stage1.csv", digs)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — models at scale
# ─────────────────────────────────────────────────────────────────────────────
S2_PAIRS, S2_PERB, S2_SHUF, S2_CTRL, S2_MARGIN = [], [], [], [], []
XG = {}


@torch.no_grad()
def pair_stats(oa, ob, y):
    """Compare logits of the SAME rows computed at two batch sizes (a = reference)."""
    n = min(oa.shape[0], ob.shape[0])
    a, b = oa[:n].double(), ob[:n].double()
    eq = (oa[:n] == ob[:n]).all(1)
    d = (a - b).abs()
    dinf = d.max(1).values
    tv, ti = a.topk(2, 1)
    margin = tv[:, 0] - tv[:, 1]
    ka, kb = a.argmax(1), b.argmax(1)      # same function on both -> identical rows can never "flip"
    flips = ka != kb
    # a flip needs some logit gap to shrink by `margin`; each logit moves <= dinf,
    # so the gap moves <= 2*dinf  ->  flips are a SUBSET of {margin < 2*dinf}.
    at_risk = (margin <= 2 * dinf) & (dinf > 0)
    gap_b = b.gather(1, ti[:, 0:1]).squeeze(1) - b.gather(1, ti[:, 1:2]).squeeze(1)
    delta = gap_b - margin
    w = 0.05
    f0 = (margin < w).sum().item() / (n * w)        # margin density near 0
    expected = n * f0 * torch.clamp(-delta, min=0).mean().item()
    ca, cb = ka == y[:n], kb == y[:n]
    k = int(flips.sum().item())
    lo, hi = wilson(k, n)
    return {"n": n, "rows_bitident": eq.double().mean().item(),
            "max_abs_diff": d.max().item(), "mean_dinf": dinf.mean().item(),
            "flips": k, "flip_rate": k / n, "flip_ci_lo": lo, "flip_ci_hi": hi,
            "at_risk": int(at_risk.sum().item()),
            "flips_not_at_risk": int((flips & ~at_risk).sum().item()),   # must be 0
            "margin_density_f0": f0, "expected_flips": expected,
            "correct_to_wrong": int((ca & ~cb).sum().item()),
            "wrong_to_correct": int((~ca & cb).sum().item())}


def margin_cdf(out, y, tag):
    a = out.double()
    tv = a.topk(2, 1).values
    m = tv[:, 0] - tv[:, 1]
    rec = dict(tag, n=int(m.numel()))
    for thr in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        rec[f"n_margin_lt_{thr:g}"] = int((m < thr).sum().item())
    return rec


def run_model_config(wl, model, Xcpu, ycpu, L64, dname, dt, cfg, N, B_list, caps, full):
    set_flags(cfg)
    Xd = Xcpu[:N].to(dev, dt)
    yd = ycpu[:N].to(dev)
    m = copy.deepcopy(model).to(dev, dt).eval()
    fn = lambda x: m(x)
    base = {"workload": wl, "dtype": dname, "cfg": cfg["name"]}
    outs = {}
    acc64 = None
    if L64 is not None:
        n64 = L64.shape[0]
        k64 = L64.argmax(1).to(dev)
        acc64 = (k64 == yd[:n64]).double().mean().item()
    for B in B_list:
        n = min(N, caps.get(B, N))
        out, rps = timed_run(fn, Xd, B, n)
        if out is None:
            continue
        if not torch.isfinite(out.float()).all():
            log(f"    {wl}|{dname}|{cfg['name']} B={B}: non-finite logits - skipped")
            continue
        outs[B] = out
        ne = out.shape[0]
        kb = out.double().argmax(1)
        rec = {**base, "B": B, "n_eff": ne, "rows_per_s": rps,
               "acc": (kb == yd[:ne]).double().mean().item()}
        if L64 is not None:
            m64 = min(L64.shape[0], ne)
            fl = int((kb[:m64] != k64[:m64]).sum().item())
            rec.update({"n_vs_fp64": m64, "flips_vs_fp64": fl,
                        "flip_rate_vs_fp64": fl / m64, "acc_fp64_ref": acc64})
        S2_PERB.append(rec)
    if not outs:
        return
    # control: rerun at the same batch size must be bit-identical
    for B in (min(outs), max(outs)):
        nc = min(4096, outs[B].shape[0])
        nc = (nc // B) * B
        if nc:
            r2 = run_batched(fn, Xd, B, nc)
            S2_CTRL.append({**base, "B": B, "n": nc,
                            "rows_bitident_rerun": (r2 == outs[B][:nc]).all(1).double().mean().item()})
    if full:
        S2_MARGIN.append(margin_cdf(outs[min(outs)], yd, {**base, "ref_B": min(outs)}))
        for B in (min(outs), max(outs)):
            XG[f"{wl}|{dname}|{cfg['name']}|B{B}"] = outs[B][:XG_N].float().cpu().numpy()
    # pairwise matrix, a<b, a = reference
    Bs = sorted(outs)
    nbad = 0
    for i, a in enumerate(Bs):
        for b in Bs[i + 1:]:
            st = pair_stats(outs[a], outs[b], yd)
            S2_PAIRS.append({**base, "Ba": a, "Bb": b, **st})
            nbad += st["flips_not_at_risk"]
    if nbad:
        log(f"    !!! SELF-CHECK FAILED: {nbad} flips outside the at-risk bound ({wl}|{dname})")
    # shuffled order: do position / neighbours matter beyond the batch size?
    gshuf = torch.Generator().manual_seed(SEED + 9)
    for B in SHUF_B:
        if B not in outs:
            continue
        nB = outs[B].shape[0]
        pm = torch.randperm(nB, generator=gshuf).to(dev)
        so = run_batched(fn, Xd[:nB][pm], B)
        back = torch.empty_like(so)
        back[pm] = so
        st = pair_stats(outs[B], back, yd)
        S2_SHUF.append({**base, "B": B, **st})
    del Xd, outs
    if HAS_CUDA:
        torch.cuda.empty_cache()


def stage2(models):
    log(f"\n######## STAGE 2: models at scale ({el()}) ########")
    for wl, (mk, maker) in WORKLOADS.items():
        log(f"\n=== workload {wl}: generating {N_L} test inputs on CPU ({el()}) ===")
        Xcpu, ycpu = gen_chunked(maker, N_L, 4242)
        model = models[mk].cpu().eval()
        L64 = None
        try:
            m64 = copy.deepcopy(model).double().eval()
            with torch.no_grad():
                L64 = torch.cat([m64(Xcpu[i:i + 1024].double()) for i in range(0, N64, 1024)])
            log(f"  fp64 reference done: accuracy on {N64} rows = "
                f"{(L64.argmax(1) == ycpu[:N64]).double().mean().item():.4f}")
        except Exception:
            log("  fp64 reference failed", traceback.format_exc())
        for dname, dt in DTYPES.items():
            for cfg in CFGS:
                full = cfg["name"] == "default"
                N = N_L if full else N_RED
                Bl = B_S2 if full else B_RED
                cp = CAPS if full else CAPS_RED
                tag = f"{wl}|{dname}|{cfg['name']}"
                t = time.time()
                try:
                    run_model_config(wl, model, Xcpu, ycpu, L64, dname, dt, cfg, N, Bl, cp, full)
                    log(f"  {tag}: done in {(time.time() - t) / 60:.1f} min   (total {el()})")
                except Exception:
                    log(f"  {tag}: FAILED\n" + traceback.format_exc())
                    if HAS_CUDA:
                        torch.cuda.empty_cache()
                save_csv("stage2_pairs.csv", S2_PAIRS)
                save_csv("stage2_perB.csv", S2_PERB)
                save_csv("stage2_shuffle.csv", S2_SHUF)
                save_csv("stage2_control.csv", S2_CTRL)
                save_csv("stage2_margin_cdf.csv", S2_MARGIN)
        del Xcpu, ycpu


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — mitigation: pad every request to a fixed batch size M
# ─────────────────────────────────────────────────────────────────────────────
S3 = []


@torch.no_grad()
def stage3(models):
    log(f"\n######## STAGE 3: fixed-size padding ({el()}) ########")
    for wl, (cls, maker) in MODEL_DEFS.items():
        if wl not in ACTIVE_MODELS:
            continue
        Xcpu, ycpu = gen_chunked(maker, N_PAD, 4242)
        model = models[wl].cpu().eval()
        for dname, dt in DTYPES.items():
            tag = f"{wl}|{dname}"
            try:
                set_flags(CFGS[0])
                Xd = Xcpu.to(dev, dt)
                m = copy.deepcopy(model).to(dev, dt).eval()
                fn = lambda x: m(x)
                unp = {}
                for M in PAD_M:
                    refM = run_batched(fn, Xd, M, N_PAD)
                    nM = refM.shape[0]
                    buf = torch.zeros((M,) + tuple(Xd.shape[1:]), dtype=dt, device=dev)
                    for Br in PAD_REQ:
                        if Br >= M:
                            continue
                        n_use = (nM // Br) * Br
                        if Br not in unp:
                            unp[Br] = timed_run(fn, Xd, Br, n_use)
                        out_un, rps_un = unp[Br]

                        def padded(xb, Br=Br, buf=buf):
                            buf[:Br] = xb
                            return fn(buf)[:Br]
                        out_p, rps_p = timed_run(padded, Xd, Br, n_use)
                        # NOTE: timed_run's warm-up calls fn(X[:B]) -> use padded explicitly
                        rec = {"workload": wl, "dtype": dname, "M": M, "B_request": Br,
                               "n": n_use,
                               "padded_rows_bitident_vs_Mbatch": (out_p == refM[:n_use]).all(1).double().mean().item(),
                               "unpadded_rows_bitident_vs_Mbatch": (out_un[:n_use] == refM[:n_use]).all(1).double().mean().item(),
                               "padded_flips_vs_Mbatch": int((out_p.double().argmax(1) != refM[:n_use].double().argmax(1)).sum().item()),
                               "unpadded_flips_vs_Mbatch": int((out_un[:n_use].double().argmax(1) != refM[:n_use].double().argmax(1)).sum().item()),
                               "rows_per_s_unpadded": rps_un, "rows_per_s_padded": rps_p}
                        S3.append(rec)
                    log(f"  {tag} M={M}: done ({el()})")
                del Xd
            except Exception:
                log(f"  {tag}: FAILED\n" + traceback.format_exc())
            finally:
                if HAS_CUDA:
                    torch.cuda.empty_cache()
            save_csv("stage3_padding.csv", S3)
    save_csv("stage3_padding.csv", S3)


# ─────────────────────────────────────────────────────────────────────────────
# summary
# ─────────────────────────────────────────────────────────────────────────────
def write_summary():
    try:
        lines = []
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 40)
        pd.set_option("display.max_rows", 400)
        f1 = os.path.join(OUT, "stage1_primitives.csv")
        if os.path.exists(f1):
            d = pd.read_csv(f1)
            d = d[d["mode"] == "ordered"]
            s = d.groupby(["primitive", "dtype", "cfg"]).agg(
                min_rows_bitident=("rows_bitident", "min"),
                max_diff_over_scale=("max_diff_over_scale", "max"),
                min_shuffled_same=("rows_bitident_shuffled_vs_ordered_sameB", "min"))
            lines += ["STAGE 1 (vs B=1)", s.to_string(), ""]
            ch = d[d["rows_bitident_vs_prevB"] < 1][["primitive", "dtype", "cfg", "prevB", "B"]]
            lines += ["STAGE 1 regime boundaries (consecutive batch sizes that differ)",
                      ch.to_string(index=False), ""]
        f2 = os.path.join(OUT, "stage2_pairs.csv")
        if os.path.exists(f2):
            d = pd.read_csv(f2)
            d1 = d[(d["Ba"] == d.groupby(["workload", "dtype", "cfg"])["Ba"].transform("min"))]
            s = d1.groupby(["workload", "dtype", "cfg"]).agg(
                min_rows_bitident=("rows_bitident", "min"), flips_total=("flips", "sum"),
                max_flip_rate=("flip_rate", "max"), max_at_risk=("at_risk", "max"),
                expected_flips_max=("expected_flips", "max"),
                bad_selfcheck=("flips_not_at_risk", "sum"))
            lines += ["STAGE 2 (reference = smallest B)", s.to_string(), ""]
        f3 = os.path.join(OUT, "stage3_padding.csv")
        if os.path.exists(f3):
            lines += ["STAGE 3 padding", pd.read_csv(f3).to_string(index=False), ""]
        lines.append(f"total minutes: {(time.time() - T0) / 60:.1f}")
        open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines))
        log("\n".join(lines[:60]))
    except Exception:
        log("summary failed", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
def main():
    global OUT, LOG
    if TRAIN_ONLY:
        ms = {}
        for kind in MODEL_DEFS:
            ms[kind], acc = train_model(kind)
        open("weights_b64.txt", "w").write(pack_state(ms))
        print("wrote weights_b64.txt")
        return
    gpu = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU"
    tag = re.sub(r"[^A-Za-z0-9]+", "", gpu)
    root = "/kaggle/working" if os.path.isdir("/kaggle/working") else (
        "/content" if os.path.isdir("/content") else ".")
    OUT = os.path.join(root, f"bi_v4_{tag}" + ("_quick" if QUICK else ""))
    os.makedirs(OUT, exist_ok=True)
    LOG = open(os.path.join(OUT, "log.txt"), "a")

    # P100 guard: PyTorch built for CUDA 13 has no Pascal kernels
    if HAS_CUDA:
        try:
            a = torch.randn(64, 64, device=dev)
            (a @ a).sum().item()
            F.conv1d(torch.randn(1, 1, 32, device=dev), torch.randn(1, 1, 3, device=dev))
            sync()
        except Exception as e:
            log(f"\nFATAL: this PyTorch build cannot run on {gpu} (sm_{cap[0]}{cap[1]}): {e}")
            log("If this is a P100, install the CUDA-12.6 build FIRST and restart:\n"
                "  pip install -q torch --index-url https://download.pytorch.org/whl/cu126\n"
                "(needs Internet ON in the notebook settings).")
            raise SystemExit(1)

    drv = ""
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
    except Exception:
        pass
    models, wsrc = load_models()
    env = {"version": VERSION, "python": sys.version, "platform": platform.platform(),
           "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "gpu": gpu, "capability": list(cap),
           "driver_and_max_sm_clock": drv, "numpy": np.__version__, "pandas": pd.__version__,
           "cpu_count": os.cpu_count(), "quick": QUICK, "stages": sorted(STAGES),
           "weights_source": wsrc,
           "weights_sha256": hashlib.sha256("".join(WEIGHTS_B64.split()).encode()).hexdigest(),
           "settings": {"N_SIG": N_SIG, "B_S1": B_S1, "N_L": N_L, "N64": N64, "B_S2": B_S2,
                        "CAPS": CAPS, "N_RED": N_RED, "B_RED": B_RED, "SHUF_B": SHUF_B,
                        "N_PAD": N_PAD, "PAD_M": PAD_M, "PAD_REQ": PAD_REQ,
                        "SIG_TONE": SIG_TONE, "SIG_GRAT": SIG_GRAT, "SEED": SEED},
           "cfgs": [c["name"] for c in CFGS]}
    json.dump(env, open(os.path.join(OUT, "env.json"), "w"), indent=2)
    log("ENV", json.dumps(env))
    if not HAS_CUDA:
        log("WARNING: no CUDA GPU found. Results are CPU-only and NOT what the paper needs.")

    for name, fn, needs_models in (("1", stage1, False), ("2", stage2, True), ("3", stage3, True)):
        if name in STAGES:
            try:
                fn(models) if needs_models else fn()
            except Exception:
                log(f"STAGE {name} CRASHED\n" + traceback.format_exc())
    if XG:
        np.savez_compressed(os.path.join(OUT, f"xgpu_logits_{tag}.npz"), **XG)
    write_summary()
    shutil.make_archive(OUT, "zip", OUT)
    log(f"\nDONE in {el()}. Download: {OUT}.zip")


if __name__ == "__main__":
    main()
