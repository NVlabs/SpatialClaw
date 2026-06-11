"""Build sample_id → category mappings from benchmark raw data.

Only reads lightweight metadata (JSON/parquet), never loads images.
All loaders return Dict[str, str] mapping sample_id to category name.
"""

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# MMSI-Video-Bench
# ---------------------------------------------------------------------------

_MMSIVIDEO_CATEGORY_MAP = {
    "Planning": "Planning",
    "Prediction": "Planning",
    "Memory Update": "Cross-Video",
    "Multi-View Integration": "Cross-Video",
    "Camera Motion": "Motion Understanding",
    "Instance Motion": "Motion Understanding",
    "Interactive Motion": "Motion Understanding",
    "Camera-Instance": "Spatial Construction",
    "Camera-Scene": "Spatial Construction",
    "Instance-Instance": "Spatial Construction",
    "Instance-Scene": "Spatial Construction",
    "Attribute": "Spatial Construction",
    "Scene-Scene": "Spatial Construction",
    "(Cross-Video) Memoery Update": "Cross-Video",
    "(Cross-Video) Multi-View Integration": "Cross-Video",
    "(Motion Understanding) Camera Motion": "Motion Understanding",
    "(Motion Understanding) Instance Motion": "Motion Understanding",
    "(Motion Understanding) Interactive Motion": "Motion Understanding",
    "(Spatial Construction) Camera-Instance Spatial Relationship": "Spatial Construction",
    "(Spatial Construction) Camera-Scene Spatial Relationship": "Spatial Construction",
    "(Spatial Construction) Instance-Instance Spatial Relationship": "Spatial Construction",
    "(Spatial Construction) Instance-Scene Spatial Relationship": "Spatial Construction",
    "(Spatial Construction) Instance/Scene Attribute": "Spatial Construction",
    "(Spatial Construction) Scene-Scene Spatial Relationship": "Spatial Construction",
}


def _load_mmsivideo(data_root: str) -> Dict[str, str]:
    path = os.path.join(data_root, "MMSI-Video-Bench", "mmsivideo.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        items = json.load(f)
    result = {}
    for item in items:
        sid = item.get("id", "")
        raw_type = item.get("type", "")
        category = _MMSIVIDEO_CATEGORY_MAP.get(raw_type, raw_type)
        result[str(sid)] = category
    return result


# ---------------------------------------------------------------------------
# MMSI-Bench
# ---------------------------------------------------------------------------

_MMSI_SUBSET_MAP = {
    "Positional Relationship (Cam.–Obj.)": "Positional Relationship",
    "Positional Relationship (Obj.–Obj.)": "Positional Relationship",
    "Positional Relationship (Obj.–Scene)": "Positional Relationship",
    "Positional Relationship (Cam.–Scene)": "Positional Relationship",
    "Positional Relationship (Scene–Scene)": "Positional Relationship",
    "Positional Relationship (Cam.–Cam.)": "Positional Relationship",
    "Motion (Cam.)": "Motion",
    "Motion (Obj.)": "Motion",
    "Attribute (Object)": "Attribute",
    "Attribute (Scene)": "Attribute",
    "MSR": "MSR",
}


def _load_mmsi(data_root: str) -> Dict[str, str]:
    path = os.path.join(data_root, "MMSI-Bench", "MMSI_Bench.parquet")
    if not os.path.isfile(path):
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["id", "question_type"])
        result = {}
        for _, row in df.iterrows():
            sid = str(row["id"])
            qtype = str(row.get("question_type", ""))
            category = _MMSI_SUBSET_MAP.get(qtype, qtype)
            result[sid] = category
        return result
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# BLINK
# ---------------------------------------------------------------------------

_BLINK_SUBTASKS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]


def _load_blink(data_root: str) -> Dict[str, str]:
    blink_dir = os.path.join(data_root, "BLINK")
    if not os.path.isdir(blink_dir):
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}
    result = {}
    for subtask in _BLINK_SUBTASKS:
        subtask_dir = os.path.join(blink_dir, subtask)
        if not os.path.isdir(subtask_dir):
            continue
        for fname in os.listdir(subtask_dir):
            if fname.startswith("val") and fname.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(subtask_dir, fname), columns=["idx"])
                for _, row in df.iterrows():
                    result[str(row["idx"])] = subtask
    return result


# ---------------------------------------------------------------------------
# MindCube
# ---------------------------------------------------------------------------

def _mindcube_get_setting(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) >= 2:
        return parts[0]
    return "other"


def _load_mindcube(data_root: str) -> Dict[str, str]:
    for fname in ["MindCube.jsonl", "MindCube_tinybench.jsonl"]:
        path = os.path.join(data_root, "MindCube", "data", "raw", fname)
        if os.path.isfile(path):
            break
    else:
        return {}
    result = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            sid = str(item.get("id", ""))
            setting = _mindcube_get_setting(sid)
            result[sid] = setting
    return result


# ---------------------------------------------------------------------------
# OmniSpatial
# ---------------------------------------------------------------------------

def _load_omnispatial(data_root: str) -> Dict[str, str]:
    for split in ["test", "full", "train"]:
        path = os.path.join(data_root, "OmniSpatial", f"OmniSpatial-{split}", "data.json")
        if os.path.isfile(path):
            break
    else:
        return {}
    with open(path) as f:
        items = json.load(f)
    result = {}
    for item in items:
        sid = str(item.get("id", ""))
        task_type = item.get("task_type", "")
        sub_task = item.get("sub_task_type", "")
        result[sid] = sub_task if sub_task else task_type
    return result


# ---------------------------------------------------------------------------
# OSI-Bench
# ---------------------------------------------------------------------------

def _load_osibench(data_root: str) -> Dict[str, str]:
    path = os.path.join(data_root, "OSI-Bench", "data.parquet")
    if not os.path.isfile(path):
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["index", "category"])
        result = {}
        for _, row in df.iterrows():
            result[str(row["index"])] = str(row["category"])
        return result
    except (ImportError, KeyError):
        return {}


# ---------------------------------------------------------------------------
# VSI-Bench
# ---------------------------------------------------------------------------

def _load_vsibench(data_root: str) -> Dict[str, str]:
    for fname in ["test.jsonl", "test_debiased.parquet"]:
        path = os.path.join(data_root, "VSI-Bench", fname)
        if os.path.isfile(path):
            break
    else:
        return {}
    result = {}
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                sid = str(item.get("id", ""))
                result[sid] = item.get("question_type", "unknown")
    elif path.endswith(".parquet"):
        try:
            import pandas as pd
            df = pd.read_parquet(path, columns=["id", "question_type"])
            for _, row in df.iterrows():
                result[str(row["id"])] = str(row["question_type"])
        except ImportError:
            pass
    return result


# ---------------------------------------------------------------------------
# VSTIBench
# ---------------------------------------------------------------------------

def _load_vstibench(data_root: str) -> Dict[str, str]:
    path = os.path.join(data_root, "vstibench", "test.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        items = json.load(f)
    result = {}
    for item in items:
        sid = str(item.get("id", ""))
        result[sid] = item.get("question_type", "unknown")
    return result


# ---------------------------------------------------------------------------
# SPAR-Bench
# ---------------------------------------------------------------------------

_SPAR_COGNITIVE_LEVELS = {
    "obj_spatial_relation_2d": "Low",
    "obj_spatial_relation_proximity": "Low",
    "position_matching": "Low",
    "depth_prediction_mono": "Low",
    "depth_prediction_stereo": "Low",
    "distance_prediction_mono": "Low",
    "distance_prediction_stereo": "Low",
    "obj_spatial_relation_3d": "Middle",
    "spatial_imagination_rotation": "Middle",
    "spatial_imagination_folding": "Middle",
    "spatial_imagination_cross_section": "Middle",
    "distance_infer_center_2d": "Middle",
    "distance_infer_center_3d": "Middle",
    "camera_motion_infer": "Middle",
    "distance_prediction_3d": "Middle",
    "distance_prediction_map": "Middle",
    "spatial_imagination_perspective": "High",
    "spatial_imagination_mirror": "High",
    "view_change_infer": "High",
    "obj_spatial_relation_abstract": "High",
}


def _load_sparbench(data_root: str) -> Dict[str, str]:
    spar_dir = os.path.join(data_root, "SPAR-Bench", "data")
    if not os.path.isdir(spar_dir):
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}
    result = {}
    for fname in os.listdir(spar_dir):
        if fname.startswith("test") and fname.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(spar_dir, fname), columns=["id", "task"])
            for _, row in df.iterrows():
                sid = str(row["id"])
                task = str(row["task"])
                result[sid] = task
    return result


# ---------------------------------------------------------------------------
# Omni3D-Bench
# ---------------------------------------------------------------------------

def _load_omni3d(data_root: str) -> Dict[str, str]:
    path = os.path.join(data_root, "Omni3D-Bench", "annotations.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        items = json.load(f)
    result = {}
    for item in items:
        sid = str(item.get("question_index", ""))
        result[sid] = item.get("answer_type", "unknown")
    return result


# ---------------------------------------------------------------------------
# SPBench
# ---------------------------------------------------------------------------

def _load_spbench(data_root: str) -> Dict[str, str]:
    spbench_dir = os.path.join(data_root, "SPBench")
    if not os.path.isdir(spbench_dir):
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}
    result = {}
    for fname in os.listdir(spbench_dir):
        if not fname.endswith(".parquet"):
            continue
        # Extract subset from filename: SPBench-SI.parquet -> "SI"
        subset = fname.replace("SPBench-", "").replace(".parquet", "")
        df = pd.read_parquet(os.path.join(spbench_dir, fname), columns=["id", "question_type"])
        for _, row in df.iterrows():
            sid = f"{subset}_{row['id']}"
            result[sid] = str(row["question_type"])
    return result


# ---------------------------------------------------------------------------
# SpatialTree-Bench
# ---------------------------------------------------------------------------

def _load_spatialtree(data_root: str) -> Dict[str, str]:
    path = os.path.join(data_root, "SpatialTree-Bench", "annotations_plain.parquet")
    if not os.path.isfile(path):
        return {}
    try:
        import json as _json
        import pandas as pd
    except ImportError:
        return {}
    result = {}
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        sid = str(row["session_id"])
        extra = {}
        if row.get("extra_info"):
            try:
                extra = _json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
            except (ValueError, TypeError):
                pass
        # Use metricfunc as category — determines scoring method
        mf = extra.get("metricfunc", "unknown")
        result[sid] = mf
    return result


# SpatialTree per-sample metadata cache (metricfunc + question_type)
_spatialtree_meta_cache: Dict[str, Dict[str, str]] = {}


def _load_spatialtree_meta(data_root: str) -> Dict[str, Dict[str, str]]:
    """Load SpatialTree per-sample metadata: {session_id: {metricfunc, question_type}}."""
    global _spatialtree_meta_cache
    if _spatialtree_meta_cache:
        return _spatialtree_meta_cache
    path = os.path.join(data_root, "SpatialTree-Bench", "annotations_plain.parquet")
    if not os.path.isfile(path):
        return {}
    try:
        import json as _json
        import pandas as pd
    except ImportError:
        return {}
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        sid = str(row["session_id"])
        extra = {}
        if row.get("extra_info"):
            try:
                extra = _json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
            except (ValueError, TypeError):
                pass
        _spatialtree_meta_cache[sid] = {
            "metricfunc": extra.get("metricfunc", ""),
            "question_type": row.get("question_type", ""),
        }
    return _spatialtree_meta_cache


def get_spatialtree_meta(data_root: str) -> Dict[str, Dict[str, str]]:
    """Get SpatialTree per-sample metadata. Cached after first load."""
    return _load_spatialtree_meta(data_root)


# ---------------------------------------------------------------------------
# Numerical category registry (categories scored by MRA, not accuracy)
# ---------------------------------------------------------------------------

_NUMERICAL_CATEGORIES: Dict[str, set] = {
    "osibench": {
        "absolute_speed", "absolute_displacement", "absolute_distance",
        "object_3d_localization", "depth_aware_counting", "trajectory_length",
    },
    "vsibench": {
        "object_abs_distance", "object_counting",
        "object_size_estimation", "room_size_estimation",
    },
    "vsibench_unbiased": {
        "object_abs_distance", "object_counting",
        "object_size_estimation", "room_size_estimation",
    },
    "vstibench": {
        "camera_obj_abs_dist", "camera_displacement", "camera_obj_dist_change",
    },
    "sparbench": {
        "depth_prediction_oc", "depth_prediction_oo",
        "depth_prediction_oc_mv", "depth_prediction_oo_mv",
        "distance_prediction_oc", "distance_prediction_oo",
        "distance_prediction_oc_mv", "distance_prediction_oo_mv",
    },
    "spbench": {
        "object_abs_distance", "object_counting",
        "object_size_estimation", "room_size_estimation",
    },
    "spatialtree": {
        "meanrelativeacc",
    },
}


def is_numerical_category(benchmark: str, category: str) -> bool:
    """Check if a category uses numerical (MRA) scoring."""
    cats = _NUMERICAL_CATEGORIES.get(benchmark)
    if cats is None:
        return False
    return category in cats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Map benchmark config name → loader function
_LOADERS = {
    "mmsivideo": lambda dr: _load_mmsivideo(dr),
    "mmsi": lambda dr: _load_mmsi(dr),
    "blink": lambda dr: _load_blink(dr),
    "mindcube": lambda dr: _load_mindcube(dr),
    "omnispatial": lambda dr: _load_omnispatial(dr),
    "osibench": lambda dr: _load_osibench(dr),
    "vsibench": lambda dr: _load_vsibench(dr),
    "vsibench_unbiased": lambda dr: _load_vsibench(dr),
    "vstibench": lambda dr: _load_vstibench(dr),
    "sparbench": lambda dr: _load_sparbench(dr),
    "omni3d": lambda dr: _load_omni3d(dr),
    "spbench": lambda dr: _load_spbench(dr),
    "spatialtree": lambda dr: _load_spatialtree(dr),
}

# Cache: benchmark_name → {sample_id: category}
_cache: Dict[str, Dict[str, str]] = {}


def get_category_map(benchmark: str, data_root: str) -> Dict[str, str]:
    """Get sample_id → category mapping for a benchmark.

    Results are cached after first load.
    """
    if benchmark in _cache:
        return _cache[benchmark]

    loader = _LOADERS.get(benchmark)
    if loader is None:
        _cache[benchmark] = {}
        return {}

    try:
        mapping = loader(data_root)
    except Exception as e:
        print(f"[category_maps] Warning: failed to load {benchmark}: {e}")
        mapping = {}

    _cache[benchmark] = mapping
    return mapping


def clear_cache():
    """Clear the category map cache."""
    global _spatialtree_meta_cache
    _cache.clear()
    _spatialtree_meta_cache = {}
