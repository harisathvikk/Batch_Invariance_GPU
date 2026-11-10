# Parameter counts and fp64 margin statistics of the embedded models (run with any PyTorch).
import importlib.util, json, os, sys
os.environ["BI_WL"] = "cnn1d,cnn2d,tfm1d"
spec = importlib.util.spec_from_file_location("v4", "batch_invariance_v4_extra.py")
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
ms, src = v.load_models()
info = {k: {"params": int(sum(p.numel() for p in m.parameters()))} for k, m in ms.items()}
spec3 = importlib.util.spec_from_file_location("v3", "batch_invariance_v3.py")
v3 = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(v3)
ms3, _ = v3.load_models()
import torch
info["cnn_weights_identical_v3_v4"] = all(torch.equal(ms3[k].state_dict()[p], ms[k].state_dict()[p])
                                          for k in ("cnn1d", "cnn2d") for p in ms3[k].state_dict())
os.makedirs("out", exist_ok=True)
json.dump(info, open("out/model_info.json", "w"), indent=1)
print(info)
