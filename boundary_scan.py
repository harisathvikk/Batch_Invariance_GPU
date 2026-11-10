# Reproduces the decision-boundary scan used to place the CNN "boundary" (hard) inputs.
# Tones: majority class vs frequency for the 1-D CNN; gratings: majority class vs orientation for the 2-D CNN.
# Run with any PyTorch (CPU is fine):  python boundary_scan.py  -> out/boundary_scan_cnn.json
import importlib.util, json, math, os
import torch
os.environ["BI_WL"] = "cnn1d,cnn2d"
spec = importlib.util.spec_from_file_location("v3", "batch_invariance_v3.py")
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
ms, _ = v.load_models()
res = {"tones_cnn1d": {}, "gratings_cnn2d": {}}
m1 = ms["cnn1d"].double().eval(); t = torch.arange(1024, dtype=torch.float32) / 1024
for f in [x * 0.5 for x in range(76, 125)]:            # 38.0 ... 62.0 cycles per window
    g = torch.Generator().manual_seed(3); ph = torch.rand(3000, generator=g) * 2 * math.pi
    X = ((torch.sin(2 * math.pi * f * t[None] + ph[:, None]) + 2 * torch.randn(3000, 1024, generator=g)) / 2)
    with torch.no_grad():
        res["tones_cnn1d"][f] = torch.bincount(m1(X.double()).argmax(1), minlength=4).tolist()
m2 = ms["cnn2d"].double().eval(); S = 32
ax = torch.arange(S, dtype=torch.float32) - (S - 1) / 2; yy, xx = torch.meshgrid(ax, ax, indexing="ij")
for deg in [x * 2.5 for x in range(0, 61)]:            # 0 ... 150 degrees
    g = torch.Generator().manual_seed(3); th = torch.full((3000,), deg * math.pi / 180)
    k = 0.08 + torch.rand(3000, generator=g) * 0.08; ph = torch.rand(3000, generator=g) * 2 * math.pi
    arg = 2 * math.pi * k[:, None, None] * (xx[None] * torch.cos(th)[:, None, None] + yy[None] * torch.sin(th)[:, None, None]) + ph[:, None, None]
    X = (torch.cos(arg) + 1.5 * torch.randn(3000, S, S, generator=g)) / 2
    with torch.no_grad():
        res["gratings_cnn2d"][deg] = torch.bincount(m2(X.double()).argmax(1), minlength=4).tolist()
os.makedirs("out", exist_ok=True)
json.dump(res, open("out/boundary_scan_cnn.json", "w"), indent=1)
for name, d in res.items():
    prev = None
    for k_, c in d.items():
        a = max(range(4), key=lambda i: c[i])
        if prev is not None and a != prev:
            print(f"{name}: majority class changes near {k_} -> {c}")
        prev = a
