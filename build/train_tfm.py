# Trains ONLY the transformer on CPU, keeps the CNN weights byte-identical to v3, scans the
# transformer's decision boundaries (for the "hard" stress set) and writes weights_b64_v4.txt
import importlib.util, io, base64, json, math, numpy as np, torch
spec = importlib.util.spec_from_file_location("v", "bi_v2_template.py"); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
old = np.load(io.BytesIO(base64.b64decode("".join(open("weights_b64.txt").read().split()))))
arrays = {k: old[k] for k in old.files}
import torch.nn.functional as F
torch.manual_seed(v.SEED)
X, y = v.gen_chunked(v.make_tones, 60000, 777)
m = v.TFM1D(); EP = 25
opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-2)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EP * ((60000 + 255) // 256))
gc = torch.Generator().manual_seed(v.SEED)
for e in range(EP):
    m.train(); perm = torch.randperm(60000, generator=gc); tot = 0.0
    for i in range(0, 60000, 256):
        idx = perm[i:i + 256]; opt.zero_grad(); loss = F.cross_entropy(m(X[idx]), y[idx]); loss.backward(); opt.step(); sch.step(); tot += loss.item() * len(idx)
    print("epoch", e + 1, round(tot / 60000, 4), flush=True)
m.eval(); Xv, yv = v.gen_chunked(v.make_tones, 8192, 4242)
with torch.no_grad(): acc = (m(Xv).argmax(1) == yv).float().mean().item()
print("val acc", acc)
for k, t in m.state_dict().items():
    arrays[f"tfm1d__{k}"] = t.cpu().numpy()
buf = io.BytesIO(); np.savez_compressed(buf, **arrays)
open("weights_b64_v4.txt", "w").write(base64.b64encode(buf.getvalue()).decode())
# boundary scan
md = m.double().eval(); t = torch.arange(1024, dtype=torch.float32) / 1024
res = {}
for f in np.arange(38, 62.01, 0.5):
    g = torch.Generator().manual_seed(3); ph = torch.rand(3000, generator=g) * 2 * math.pi
    X = ((torch.sin(2 * math.pi * float(f) * t[None] + ph[:, None]) + 2 * torch.randn(3000, 1024, generator=g)) / 2).float()
    with torch.no_grad(): res[float(f)] = torch.bincount(md(X.double()).argmax(1), minlength=4).tolist()
json.dump(res, open("tfm_scan.json", "w"))
for f, c in res.items(): print(f, c)
print("val acc", acc)
