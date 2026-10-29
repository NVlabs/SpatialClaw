"""Benchmark factory and registry."""

import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import BaseBenchmark


@dataclass(frozen=True)
class BenchmarkSpec:
    """Single source of truth for a benchmark registry entry."""

    module_path: Optional[str]
    class_name: Optional[str]
    data_dir: Optional[str] = None
    init_kwargs: Dict[str, Any] = field(default_factory=dict)

    def load_class(self):
        if not self.module_path or not self.class_name:
            return None
        module = import_module(self.module_path)
        return getattr(module, self.class_name)


BENCHMARK_REGISTRY: Dict[str, BenchmarkSpec] = {
    # --- Single-image spatial reasoning ---
    "erqa": BenchmarkSpec("spatial_agent.evals.erqa", "ERQABench", "ERQA"),
    "omni3d": BenchmarkSpec("spatial_agent.evals.omni3d", "Omni3DBench", "Omni3D-Bench"),
    "omnispatial": BenchmarkSpec("spatial_agent.evals.omnispatial", "OmniSpatialBench", "OmniSpatial"),
    "spbench": BenchmarkSpec("spatial_agent.evals.spbench", "SPBench", "SPBench"),
    # --- Multi-view spatial reasoning ---
    "mindcube": BenchmarkSpec("spatial_agent.evals.mindcube", "MindCubeBench", "MindCube"),
    "mmsi": BenchmarkSpec("spatial_agent.evals.mmsi", "MMSIBench", "MMSI-Bench"),
    "sparbench": BenchmarkSpec("spatial_agent.evals.sparbench", "SPARBench", "SPAR-Bench"),
    # --- General spatial reasoning ---
    "blink": BenchmarkSpec("spatial_agent.evals.blink", "BLINKBench", "BLINK"),
    "spatialtree": BenchmarkSpec("spatial_agent.evals.spatialtree", "SpatialTreeBench", "SpatialTree-Bench"),
    "viewspatial": BenchmarkSpec("spatial_agent.evals.viewspatial", "ViewSpatialBench", "ViewSpatial-Bench"),
    # --- Video spatial & 4D reasoning ---
    "mmsivideo": BenchmarkSpec("spatial_agent.evals.mmsivideo", "MMSIVideoBench", "MMSI-Video-Bench"),
    "osibench": BenchmarkSpec("spatial_agent.evals.osibench", "OSIBench", "OSI-Bench"),
    "paibench": BenchmarkSpec("spatial_agent.evals.paibench", "PAIBench", "PAI-Bench"),
    "vsibench": BenchmarkSpec("spatial_agent.evals.vsibench", "VSIBench", "VSI-Bench"),
    "vsibench_unbiased": BenchmarkSpec(
        "spatial_agent.evals.vsibench",
        "VSIBench",
        "VSI-Bench",
        init_kwargs={"variant": "unbiased"},
    ),
    "vstibench": BenchmarkSpec("spatial_agent.evals.vstibench", "VSTIBench", "vstibench"),
    "dsibench": BenchmarkSpec("spatial_agent.evals.dsibench", "DSIBench", "DSI-Bench"),
    # --- General video understanding ---
    "cvbench": BenchmarkSpec("spatial_agent.evals.cvbench", "CVBench", "CVBench"),
    "perceptioncomp": BenchmarkSpec("spatial_agent.evals.perceptioncomp", "PerceptionCompBench", "PerceptionComp"),
    "videomme": BenchmarkSpec("spatial_agent.evals.videomme", "VideoMMEBench", "Video-MME"),
    "videommev2": BenchmarkSpec("spatial_agent.evals.videommev2", "VideoMMEv2Bench", "Video-MME-v2"),
    # --- Benchmarks beyond the paper ---
    # Cambrian-1 CV-Bench (2024, single image) — distinct from the 2025 video "cvbench" above.
    "cvbench2024": BenchmarkSpec("spatial_agent.evals.cvbench2024", "CVBench2024", "CV-Bench-2024"),
    "multihopspatial": BenchmarkSpec("spatial_agent.evals.multihopspatial", "MultihopSpatialBench", "MultihopSpatial"),
    "spinbench": BenchmarkSpec("spatial_agent.evals.spinbench", "SpinBench", "SpinBench"),
    "stibench": BenchmarkSpec("spatial_agent.evals.stibench", "STIBench", "STI-Bench"),
    "spatialrgptbench": BenchmarkSpec("spatial_agent.evals.spatialrgptbench", "SpatialRGPTBench", "SpatialRGPT-Bench"),
    # single-image integer-answer benchmarks
    "spatialiq": BenchmarkSpec("spatial_agent.evals.spatialiq", "SpatialIQBench", "Spatial-IQ"),
    "pixmocount": BenchmarkSpec("spatial_agent.evals.pixmocount", "PixMoCountBench", "PixMo-Count"),
    "countqa": BenchmarkSpec("spatial_agent.evals.countqa", "CountQABench", "CountQA"),
    # single-image 3D spatial-reasoning MCQ
    "3dsrbench": BenchmarkSpec("spatial_agent.evals.dsrbench3d", "DSRBench3D", "3DSRBench"),
    "crosspointbench": BenchmarkSpec("spatial_agent.evals.crosspointbench", "CrossPointBench", "CrossPoint-Bench"),
    # pointing suite
    "refspatialbench": BenchmarkSpec("spatial_agent.evals.refspatialbench", "RefSpatialBench", "RefSpatial-Bench"),
    "robospatialhome": BenchmarkSpec("spatial_agent.evals.robospatialhome", "RoboSpatialHome", "RoboSpatial-Home"),
    "where2place": BenchmarkSpec("spatial_agent.evals.where2place", "Where2Place", "Where2Place"),
    # --- Sentinel ---
    "none": BenchmarkSpec(None, None),
}


class BenchmarkFactory:
    """Create benchmark instances by name."""

    @staticmethod
    def create_benchmark(
        benchmark_name: str,
        data_root: str = "data",
        question_type: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[BaseBenchmark]:
        if benchmark_name not in BENCHMARK_REGISTRY:
            raise ValueError(
                f"Unknown benchmark '{benchmark_name}'. "
                f"Available: {list(BENCHMARK_REGISTRY.keys())}"
            )

        spec = BENCHMARK_REGISTRY[benchmark_name]
        bench_class = spec.load_class()
        if bench_class is None:
            return None

        data_dir = spec.data_dir or benchmark_name
        data_path = os.path.join(data_root, data_dir)
        init_kwargs = {**spec.init_kwargs, **kwargs}
        return bench_class(
            data_path=data_path,
            question_type=question_type,
            **init_kwargs,
        )
