t = open("bi_v2_template.py").read()
w = open("weights_b64_v4.txt").read().strip()
lines = "\n".join(w[i:i+100] for i in range(0, len(w), 100))
a = 'WEIGHTS_B64 = ""  # __WEIGHTS_B64__'; assert a in t
t = t.replace(a, 'WEIGHTS_B64 = """\n' + lines + '\n"""')
a = "TFM_CENTERS = (43.0, 47.7, 53.5)"; assert a in t
t = t.replace(a, "TFM_CENTERS = (43.5, 49.0, 54.5)")
# defaults for the EXTRA run: transformer only, stages 2+3 (override with env vars)
a = 'os.environ.get("BI_WL", "")'; assert a in t
t = t.replace(a, 'os.environ.get("BI_WL", "tfm1d,tfm1d_hard")')
a = 'os.environ.get("BI_STAGES", "1,2,3")'; assert a in t
t = t.replace(a, 'os.environ.get("BI_STAGES", "2,3")')
open("batch_invariance_v4_extra.py", "w").write(t)
