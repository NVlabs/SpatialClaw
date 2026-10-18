<div align="center">

<img src="https://raw.githubusercontent.com/SpatialClaw/spatialclaw.github.io/main/static/images/icon.svg" width="120" alt="SpatialClaw logo" />

# SpatialClaw
### Contributing to NVIDIA Nemotron 3.5 Super VL

**_From executable reasoning to training demonstrations._**

<br/>

[![README](https://img.shields.io/badge/README-181717?style=for-the-badge&logo=github&logoColor=white)](../README.md)
[![Nemotron Report](https://img.shields.io/badge/Nemotron_Report-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf)
[![SpatialClaw Paper](https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2606.13673)
[![BibTeX](https://img.shields.io/badge/Cite-BibTeX-4C8C4A?style=for-the-badge&logo=googlescholar&logoColor=white)](../README.md#citation)

<br/>

</div>

---

> 💡 **TL;DR.** SpatialClaw was used to generate **executable video-QA demonstrations for supervised fine-tuning in NVIDIA Nemotron 3.5 Super VL**, as described in the [Nemotron technical report (§3.1)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf#page=14).

---

## From tool use to training data

[SpatialClaw](https://arxiv.org/abs/2606.13673) is a training-free framework that lets a vision-language model write code, use visual tools, and inspect results in a persistent Python workspace. In this application, its recorded interactions become demonstrations for training a model.

The report describes three steps:

1. 💻 **Generate:** Kimi K2.6 uses SpatialClaw to answer video questions through executable Python actions and visual observations.
2. ✅ **Filter:** Check answer correctness, successful execution, and trajectory quality.
3. 📦 **Prepare for training:** Package the retained conversations and images for supervised fine-tuning.

> 📊 **Demonstration corpus:** **14,169 trajectories** · **81,366 assistant actions**

➡ Full details: [§3.1, “Agentic Vision Data”](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf#page=14).

---

## Reported results

[Table 7](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf#page=15) evaluates the broader agentic vision training recipe using the SpatialClaw harness:

| Training condition | Average score |
| --- | ---: |
| Without agentic vision training | 50.5 |
| With agentic vision training | **54.8** |
| **Improvement** | **+4.3 points** |

📈 Scores improve across **all 20 evaluated spatial reasoning and video benchmarks**.

> 🔍 **Reading this result.** This comparison covers the broader image and video training mixture, including SpatialClaw-generated demonstrations. It does **not isolate the effect of SpatialClaw-generated data alone**.

---

## Acknowledgments

We are honored to contribute to this effort and grateful to the **NVIDIA Nemotron team** for the opportunity, trust, and collaboration. We thank [**Seokju Cho**](https://seokju-cho.github.io/), SpatialClaw's first author, and all our collaborators for the work behind this contribution.

---

## References and citation

1. **NVIDIA (2026).** [*Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI.*](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf) See §3.1 and Table 7, pp. 14–15.
2. **Cho et al. (2026).** [*SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning.*](https://arxiv.org/abs/2606.13673) NeurIPS 2026; arXiv:2606.13673.

If you use SpatialClaw in your research, please [cite our paper](../README.md#citation). For work building on Nemotron or its training results, please also cite the Nemotron technical report as appropriate.

🚀 **[Try SpatialClaw](../README.md#quickstart)**
