# SpatialClaw beyond the paper's 20 benchmarks

Beyond the 20 benchmarks of the paper, SpatialClaw was evaluated on further spatial-reasoning benchmarks with the same backbone, the same tool set and one configuration for every dataset. Each number pairs the tool agent against a **no-tool chain-of-thought baseline of the same model on the same items**, so the difference isolates what the tools add. This page lists the benchmarks on which SpatialClaw comes out ahead of that baseline.

**Setup.** Backbone Qwen3.6-27B (vLLM, thinking on) for both arms. Agent = the SpatialClaw harness as released (code cells, Reconstruct, SAM3, `vlm.*` sub-queries). Baseline = `spatial_agent/entrypoints/cot_baseline.py` (chain of thought, no tools; for the benchmarks whose answers are not multiple-choice letters, CrossPoint-Bench, SpatialRGPT-Bench, ReVSI and Where2Place, it ran with `--system_prompt cot_fmt`, the format-agnostic variant of the CoT prompt, and with the default `cot` prompt otherwise). Items are a seed-42 cohort per benchmark (stratified where the official metric is category-balanced). Every benchmark is scored with its official rule (SpatialRGPT-Bench: see the note under the table); protocols, cohorts and hypotheses were pre-registered before any run, with one look at the result when the agent arm finished. Dataset sources and the expected `data/` layout for the benchmarks beyond the paper are listed in [configuration.md](configuration.md#benchmarks-beyond-the-paper).

## Results

Δ = SpatialClaw − CoT in points.

| Benchmark | Venue | Input; answer | No-tool CoT | SpatialClaw | Δ |
|---|---|---|---:|---:|---:|
| [CrossPoint-Bench](https://arxiv.org/abs/2512.04686) | arXiv 2025 | 1–2 images; points + MCQ | 43.7 | 74.6 | +30.9 |
| [SpatialRGPT-Bench](https://arxiv.org/abs/2406.01584)¹ | NeurIPS 2024 | 1 image + regions; yes/no, numeric | 53.7 | 61.8 | +8.1 |
| [ReVSI (32-frame)](https://arxiv.org/abs/2604.24300) | ICML 2026 | video, 32 frames; numeric + MCQ | 58.8 | 66.0 | +7.1 |
| [CountQA](https://arxiv.org/abs/2508.06585) | ICCV 2025 Workshop | 1 image; count | 50.3 | 56.9 | +6.7 |
| [Spatial-IQ](https://arxiv.org/abs/2607.22864) | NeurIPS 2026 | 1 image; integer | 45.5 | 48.3 | +2.8 |
| [SpinBench](https://arxiv.org/abs/2509.25390) | ICLR 2026 | 1–5 images; MCQ | 83.1 | 85.2 | +2.1 |
| [PixMo-Count](https://arxiv.org/abs/2409.17146) | CVPR 2025 | 1 image; count | 85.8 | 86.6 | +0.8 |
| [Where2Place](https://arxiv.org/abs/2406.10721) | CoRL 2024 | 1 image; points | 62.3 | 62.7 | +0.4 |
| [MultihopSpatial](https://arxiv.org/abs/2603.18892) | ECCV 2026 | 1 image; MCQ | 63.5 | 63.8 | +0.3 |

¹ SpatialRGPT-Bench is scored with a deterministic re-implementation of the official GPT-4-turbo judge (same numeric tolerance and qualitative buckets), not the judge itself; its results carry `"scoring": "deterministic_proxy (non-official; the official metric uses a GPT-4-turbo judge)"`.

## Where the gains come from

The table is ordered by Δ. The gains are CrossPoint-Bench +30.9, SpatialRGPT-Bench +8.1, ReVSI (32-frame) +7.1, CountQA +6.7, Spatial-IQ +2.8, SpinBench +2.1, PixMo-Count +0.8, Where2Place +0.4, and MultihopSpatial +0.3.

CountQA and PixMo-Count are both listed as one image and a count. CountQA goes from 50.3 to 56.9 (Δ +6.7). PixMo-Count goes from 85.8 to 86.6 (Δ +0.8).
