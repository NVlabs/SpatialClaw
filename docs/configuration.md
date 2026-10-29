# Configuration & Benchmarks

Model/dataset JSON configs, environment-variable overrides, and the list of supported benchmarks.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; Prev: [« Monitoring & logs](monitoring.md) &nbsp;·&nbsp; Next: [Architecture »](architecture.md)

---

## Configuration Reference

Loading priority (highest first): **CLI args > Model/Dataset JSON > Environment variables > Defaults.**

### Model configs (`spatial_agent/config/model/*.json`)

```json
{
    "llm_model":    "qwen3.5-397b-a17b",
    "llm_base_url": "vllm",
    "llm_api_key":  "bearer",
    "roles": {
        "main":          { "max_tokens": 131072, "temperature": 0.6, "enable_thinking": true },
        "planning":      { "max_tokens": 131072, "temperature": 0.6, "enable_thinking": true },
        "vlm":           { "max_tokens": 32768,  "temperature": 0.6, "enable_thinking": true },
        "vlm_grounding": { "max_tokens": 16384,  "temperature": 0.6, "enable_thinking": true },
        "general":       { "max_tokens": 131072, "temperature": 0.6, "enable_thinking": false }
    }
}
```

* `llm_base_url: "vllm"` — auto-discover endpoints from `spatial_agent/logs/serve.json` (load-balanced, sticky-session for prefix cache hits). With this base URL, **`llm_model` must be the `served_name`** from `vllm_manager/models.json` (e.g. `qwen3.5-397b-a17b`), **not** the HuggingFace path.
* `llm_base_url: "https://..."` — call any OpenAI-compatible HTTP endpoint. Here `llm_model` is whatever model identifier that endpoint expects (e.g. `gcp/google/gemini-3-pro`).
* `llm_api_key` accepts `${ENV_VAR}` and `${ENV_VAR:-default}` substitution; secrets stay in `.env` (see [Installation → API Keys](installation.md#api-keys--env)).
* `roles.*` — per-role hyperparameters (`main`, `planning`, `vlm`, `vlm_grounding`, `general`, `reflection`).

### Dataset configs (`spatial_agent/config/dataset/*.json`)

Per-benchmark settings: data paths, question types, frame sampling, tool list, agent-loop caps.

### Common env-var overrides

Any `SpatialAgentConfig` field can be overridden via `SPATIAL_AGENT_<FIELD>=...`. Examples:

```bash
export SPATIAL_AGENT_CONCURRENCY=16
export SPATIAL_AGENT_MAX_STEPS=30
export SPATIAL_AGENT_RECONSTRUCT_BACKEND=da3   # or pi3
```

---

## Supported Benchmarks

All 20 paper benchmarks ship as ready-to-run dataset configs under `spatial_agent/config/dataset/`:

| Category                          | Benchmarks                                                                   |
|-----------------------------------|------------------------------------------------------------------------------|
| Single-image spatial reasoning    | ERQA, Omni3D, OmniSpatial, SPBench                                           |
| Multi-view spatial reasoning      | MindCube, MMSI, SPAR-Bench                                                   |
| General spatial reasoning         | BLINK, SpatialTree, ViewSpatial                                             |
| Video spatial & 4D reasoning      | MMSI-Video, OSI-Bench, PAI-Bench, VSI-Bench-U, VSTI-Bench, DSI-Bench         |
| General video understanding       | CV-Bench, PerceptComp, Video-MME, Video-MME-v2                              |

See `spatial_agent/evals/` for the corresponding loaders and scoring scripts.

### Benchmarks beyond the paper

This branch also ships loaders and configs for 14 further benchmarks; the results are summarized in [benchmarks_beyond_paper.md](benchmarks_beyond_paper.md). None of the datasets ship with the repository: download each one from its source and place it under `data/<folder>` (the default data root is `data/`, relative to the directory you launch from, i.e. the repository root in the Quickstart). Loaders that read parquet files with embedded images dump them to disk on first load, typically under `data/<folder>/images/`. The docstring at the top of each module in `spatial_agent/evals/` describes the files it reads. Every dataset keeps its own license; check the source page before use.

| Benchmark | Config (`spatial_agent/config/dataset/`) | Source | Expected under `data/` |
|---|---|---|---|
| CV-Bench-2024 (Cambrian-1, images) | `cvbench2024.json` | HF [`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | `CV-Bench-2024/{test_2d,test_3d}.parquet` |
| MultihopSpatial | `multihopspatial.json` | HF [`etri-vilab/MultihopSpatial`](https://huggingface.co/datasets/etri-vilab/MultihopSpatial) | `MultihopSpatial/data/{multihop_test_4500.json,images/}` |
| SpinBench | `spinbench.json` | HF [`YuyouZhang/SpinBench`](https://huggingface.co/datasets/YuyouZhang/SpinBench) | `SpinBench/{test.jsonl,images/}` (unzip `images.zip`) |
| STI-Bench | `stibench.json` | HF [`MINT-SJTU/STI-Bench`](https://huggingface.co/datasets/MINT-SJTU/STI-Bench) | `STI-Bench/{data/test-00000-of-00001.parquet,video/*.mp4}` (unzip `video.zip`) |
| SpatialRGPT-Bench | `spatialrgptbench.json` | HF [`a8cheng/SpatialRGPT-Bench`](https://huggingface.co/datasets/a8cheng/SpatialRGPT-Bench) | `SpatialRGPT-Bench/{SpatialRGPT-Bench_v1.json,images/}` |
| Spatial-IQ | `spatialiq.json` | HF [`patrickqrim/spatial-iq`](https://huggingface.co/datasets/patrickqrim/spatial-iq) | `Spatial-IQ/data_*/level_1/sample_*/view_*/<variant>/{view.png,gt.json}` (the paper's evaluation set: the 750 views listed in the official code's `inference_results/subset_eval.txt` × 4 camera variants = 3,000 view directories; the loader evaluates every view directory it finds) |
| PixMo-Count | `pixmocount.json` | HF [`kcz358/pixmo-count`](https://huggingface.co/datasets/kcz358/pixmo-count) (lmms-eval re-host, images) and [`allenai/pixmo-count`](https://huggingface.co/datasets/allenai/pixmo-count) (official annotations) | `PixMo-Count/data/{test,validation}-*.parquet` (re-host, required); `PixMo-Count/official/data/{test,validation}-*.parquet` (optional: without them the loader warns and takes labels from the re-host questions; scores are unchanged) |
| CountQA | `countqa.json` | HF [`Jayant-Sravan/CountQA`](https://huggingface.co/datasets/Jayant-Sravan/CountQA) | `CountQA/data/test-*.parquet` |
| 3DSRBench | `3dsrbench.json`, `3dsrbench_c500.json` | HF [`ccvl/3DSRBench`](https://huggingface.co/datasets/ccvl/3DSRBench) (parquet plus `coco_images.zip`, the MS-COCO images it uses) | `3DSRBench/{3dsrbench_v1-00000-of-00001.parquet,coco_images/{train2017,val2017}/}` (unzip `coco_images.zip`; mirrored twins are written to `3DSRBench/flipped/` on first load) |
| CrossPoint-Bench | `crosspointbench.json` | HF [`WangYipu2002/CrossPoint-Bench`](https://huggingface.co/datasets/WangYipu2002/CrossPoint-Bench) | `CrossPoint-Bench/{CrossPoint-Bench.jsonl,image/}` |
| RefSpatial-Bench | `refspatialbench.json` | HF [`BAAI/RefSpatial-Bench`](https://huggingface.co/datasets/BAAI/RefSpatial-Bench) | `RefSpatial-Bench/{Location,Placement,Unseen}/{question.json,image/,mask/}` |
| RoboSpatial-Home | `robospatialhome.json` | HF [`chanhee-luke/RoboSpatial-Home`](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home) | `RoboSpatial-Home/data/<split>-*.parquet` |
| Where2Place | `where2place.json` | HF [`wentao-yuan/where2place`](https://huggingface.co/datasets/wentao-yuan/where2place) | `Where2Place/{point_questions.jsonl,images/,masks/}` |
| ReVSI (32-frame) | `revsi.json`, `revsi_s500.json`, `revsi_s1000.json`, `revsi_s1500.json` | HF [`3dlg-hcvc/ReVSI`](https://huggingface.co/datasets/3dlg-hcvc/ReVSI) | `ReVSI/{32_frame/test-00000-of-00001.parquet,video/32_frame/*.mp4}` (unzip `video.zip`) |

Notes:

* Configs with an `_s<N>` or `_c<N>` suffix pin the cohorts used in the campaign through `sample_ids` (`revsi_s1500.json` is the cohort behind the ReVSI row on the results page; `3dsrbench_c500.json` holds 500 base items plus their `-flip` mirrored twins); the plain config runs the full benchmark.
* Video benchmarks (STI-Bench, ReVSI) need `ffmpeg` (on `PATH`, or the `imageio-ffmpeg` package); frames are extracted once into a `.frame_cache/` folder next to the videos, falling back to the system temp directory when that folder is not writable.
* ReVSI optionally reads `ReVSI/metadata/scene_video_sizes.json` (`{scene_id: [width, height]}`, a locally generated file, e.g. with `ffprobe` over the 32-frame mp4s; not part of the HF release) to label each item portrait or landscape in the per-item breakdown; scores do not depend on it.
* SpatialRGPT-Bench is scored with a deterministic re-implementation of the official GPT-4-turbo judge (same numeric tolerance and qualitative buckets), not the judge itself; its results carry `"scoring": "deterministic_proxy (non-official; the official metric uses a GPT-4-turbo judge)"`.

---

> Next: [Architecture »](architecture.md)
