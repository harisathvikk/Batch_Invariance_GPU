# Batch Invariance of GPU Inference for Signal Classification

Code, raw results and analysis for the paper
**"When Does Batch Size Change the Answer? Measuring Bit- and Decision-Level Batch Invariance of GPU Inference for Signal Classification."**

The question: if the same input is processed in a batch of 1, 8 or 1,024 inputs on a GPU, are its output bits the same, and does its predicted class ever change?

## What is measured

| Stage | What | Scale |
|---|---|---|
| 1. Primitives | rFFT, 2-D rFFT, STFT log-power, FIR, Conv1d, Conv2d, Linear, row sum, softmax, LayerNorm | 13 ops × 21 batch sizes × FP32/FP16, vs. batch size 1 |
| 2. Models | 1-D CNN, 2-D CNN, small transformer; standard and decision-boundary inputs | 262,144 inputs, all 171 pairs of 19 batch sizes, FP32/FP16, FP64 reference |
| 3. Mitigation | pad every request to a fixed batch M ∈ {64, 256} | 16,384 inputs, request sizes 1–32 |

All inputs are synthetic and generated from fixed seeds on the CPU. The trained model weights are embedded in the scripts (base64), so every GPU runs exactly the same parameters on exactly the same inputs.

## Repository layout

```
batch_invariance_v3.py        harness that produced data/v3_T4_kaggle (primitives + CNNs), exact file that was run
batch_invariance_v4_extra.py  harness that produced data/v4_T4_kaggle (transformer), exact file that was run
analyze.py                    turns data/ into every number, table and figure in the paper -> out/
model_info.py                 parameter counts; checks CNN weights are identical in v3 and v4
boundary_scan.py              decision-boundary scan used to place the CNN boundary inputs
build/                        template + CPU training/build scripts and the embedded weights
  bi_v2_template.py           harness template (no weights); BI_TRAIN_ONLY=1 trains the CNNs on CPU
  train_tfm.py                CPU training recipe of the transformer + its boundary scan (tfm_scan.json)
  build_v4.py                 injects weights_b64_v4.txt into the template -> batch_invariance_v4_extra.py
data/
  v3_T4_kaggle/               main run (Kaggle, Tesla T4, PyTorch 2.11.0, CUDA 13.0, cuDNN 9.27)
  v4_T4_kaggle/               transformer run (same environment)
  v2_T4_kaggle/               earlier independent session, used only for the reproducibility check
  colab_v3_T4_partial_log.txt log of a partial rerun on Colab (PyTorch 2.14.1, cuDNN 9.28)
archive/batch_invariance_v2.py  the earlier harness version (see "Version history")
out/                          numbers.json, generated tables (.tex) and figures (.pdf)
TRACE.md                      where every number in the paper comes from
```

## Reproduce the analysis (no GPU needed)

```bash
pip install -r requirements.txt
python analyze.py            # rewrites out/numbers.json, out/tab_*.tex, out/fig_*.pdf
python model_info.py         # needs PyTorch (CPU is fine)
python boundary_scan.py      # needs PyTorch (CPU is fine)
```

## Re-run the GPU experiments

1. Open a Kaggle notebook with a GPU (we used "GPU T4 x2"; only one GPU is used). Internet is not needed.
2. Paste the whole of `batch_invariance_v3.py` into one cell and run it. A 1-minute smoke test: put
   `import os; os.environ["BI_QUICK"]="1"` at the top first.
3. Download the `bi_v3_<GPU>.zip` from the output folder. Repeat with `batch_invariance_v4_extra.py` for the transformer.
4. Unzip into `data/` and point the paths at the top of `analyze.py` to the new folders.

Run times on the T4: 45.8 min (v3) and 46.8 min (v4).
GPUs of the Pascal generation (e.g. P100) are not supported by PyTorch builds for CUDA 13; the script stops with a clear message.

## Version history (why there is a v2)

`archive/batch_invariance_v2.py` produced `data/v2_T4_kaggle`. Its *analysis* code had two bugs, both fixed in v3:
(1) the flip counter broke exact FP16 ties with two different rules, which produced spurious flips;
(2) fractions were averaged in float32, which produced spurious "regime boundaries".
Neither bug affects the raw outputs, so the v2 run is used only to compare output digests and saved logits with v3
(76/76 digests and 8/8 logit arrays identical). No v2 statistic is reported in the paper.

## Limitations

One GPU model (Turing T4) and one full software stack; synthetic data; small models. Thresholds and batch-size classes
are specific to the kernels selected on this GPU and library version. See the paper for details.

## License

MIT (see `LICENSE`).
