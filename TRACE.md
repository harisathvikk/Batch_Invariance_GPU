# Where every number in the paper comes from

`out/numbers.json` is written by `analyze.py` from the raw files in `data/`. Keys below refer to that file.
D3 = `data/v3_T4_kaggle`, D4 = `data/v4_T4_kaggle`.

| Paper statement | Value | Source |
|---|---|---|
| GPU, compute capability, PyTorch, CUDA, cuDNN, driver | Tesla T4, 7.5, 2.11.0, 13.0, 9.27 (92700), 580.178.04 | `env` (D3/env.json) |
| Run times | 45.8 min, 46.8 min | `runtime_min` (D3, D4 summary.txt) |
| 13 primitives, 21 batch sizes, 2,048 rows (1,024 for 2-D FFT) | | `settings`, PRIMS in batch_invariance_v3.py |
| 11 of 13 primitives invariant per precision | 11 / 11 | `s1_invariant_count` |
| Table I (identical fraction, max deviation) | | `primitives`, out/tab_primitives.tex |
| Row sums changed 75–79%; conv2d FP16 ~99.4% at B≥17; linear FP32 0 at all B | | D3/stage1_primitives.csv (rows_bitident) |
| Thresholds 3→4, 7→8, 15→16, 16→17, 129→255, 512→1024 | | `s1_regime_boundaries` |
| Deviation vs. FP64 error: 1.6e-7 vs 8.8e-8; 7.3e-4 vs 6.8e-4 | | `primitives[...].max_dev_rel`, `.err_fp64_rel` |
| cuDNN settings identical digests, 12 checks | 12, 0 differing | `s1_cudnn_digest_groups`, `s1_cudnn_groups_cfg_dependent` |
| 262,144 inputs; 19 model batch sizes; 171 pairs | | `s2_n_rows`, `s2_batch_sizes`, `models[*].pairs` |
| 1,026 FP32 pairs, 0 flips | 1026, 0 | `fp32_pairs`, `fp32_total_flips` |
| Up to 99.6% rows differ (worst pair 0.4% identical) | 0.00386 | `fp32_min_rows_bitident_models` |
| Largest FP32 logit change 3.6e-6; FP16 7.8e-3 | | `fp32_max_abs_dlogit`, `fp16_max_abs_dlogit` |
| Wilson 95% upper bound 1.5e-5 (n = 262,144, 0 flips) | 1.466e-5 | `models[*|fp32].zero_flip_upper95_rate` |
| FP16 rows changed ≤2.7% (1-D), ≤0.9% (2-D) | 0.9734, 0.9912 | `models[cnn1d|fp16]`, `models[cnn2d|fp16]` min_rows_bitident |
| 4–54 rows with margin < 1e-7 (FP16); none < 1e-6 (FP32) | | `margin_cdf` (stage2_margin_cdf.csv) |
| CNN boundary sets ≤2 flips per pair, 7.6e-6 | | `models[cnn1d_hard|fp16]`, `models[cnn2d_hard|fp16]` |
| FP16 vs FP64 flips 277, 49, 58, 4, 17–19, 24–27 | | `models[*].fp64_flips_min/max` (stage2_perB.csv) |
| Transformer 141/171 pairs; 82 of 262,119 (B=8 vs 33); 3.1e-4 | | `models[tfm1d|fp16]` |
| 962 correct→incorrect, 861 incorrect→correct | | `models[tfm1d|fp16].c2w/.w2c` |
| 144 shuffle checks, 0 changes; 72 rerun checks | | `s2_shuffle_n`, `s2_shuffle_min`, `s2_control_n` |
| benchmark vs deterministic identical in 336 pairs | 336/336 | `s2_bench_vs_det_pairs/_identical` |
| Batch-size classes | | `tfm_fp16_classes`, `tfm_fp32_classes` |
| B=8 shares ≤17.1% rows; worst pair without B=8: 7.6e-5 | | `tfm_fp16_B8_max_rows_bitident`, `models[tfm1d|fp16].max_flip_rate_excl_B8` |
| B=1 vs B=1024 no flips (transformer FP16) | 0 | D4/stage2_pairs.csv (Ba=1, Bb=1024) |
| 2,724 comparisons, 0 bound violations | | `s2_pairs_total`, `s2_bound_violations` |
| 1,439 of 2,052 pairs with no at-risk row | | `pairs_at_risk_zero`, `s2_pairs_default` |
| 10.3% of FP16 at-risk rows flipped | 0.1027 | `flips_over_at_risk_fp16` |
| 2Δ = 0.008–0.016; 0.06–1.5% of rows below it | | `fp16_certifiable` |
| Eq. (1): 1,995 vs 2,033; 2,358 vs 1,973; 33 vs 0; 25 vs 45 | | `models[*|fp16].sum_expected/.sum_flips` |
| Spearman 0.95, 0.92 (per workload); 0.87 (all FP16 pairs) | | `models[*].spearman_pred_obs`, `pred_spearman_fp16_all` |
| Fig. 3: 536 pairs | | `fig3_pairs_plotted` |
| Padding: 72 configs bit-identical, 0 padded flips, 32 unpadded flips | | `s3_configs`, `s3_padded_min_bitident`, `s3_padded_flips`, `s3_unpadded_flips` |
| Padding cost M=64: mean 6%, range −2% to 17%; M=256: mean 33%, max 66% | | `s3_ratio_M64`, `s3_ratio_M256`, `padding` |
| Throughput 1,777 → 285,905 inputs/s (1-D CNN, FP32) | | `throughput_fp32.cnn1d` |
| Parameters 6,132 / 6,132 / 19,876 | | out/model_info.json |
| FP64 accuracy 83.9%, 94.6%, 58.6% | | `models[*|fp32].acc_fp64` |
| 76/76 digests, 8/8 logit arrays identical across sessions | | `repro_*` |
| Colab: 38/38 summary lines, both FP64 accuracies | | `colab` |
| Boundary inputs (tones 43.0/47.7/53.5 and 43.5/49.0/54.5, s.d. 0.15; gratings 22.5/67.5/112.5°, s.d. 0.04 rad) | | make_tones_hard / make_gratings_hard in the scripts; out/boundary_scan_cnn.json; build/tfm_scan.json |
