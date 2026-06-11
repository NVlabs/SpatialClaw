"""ERQA (Embodied Reasoning Question Answer) data loader.

Data structure (HuggingFace parquet):
    data/ERQA/data/test-00000-of-00001.parquet   (400 examples)

Parquet columns:
    - question_id: str       — e.g. "ERQA_1"
    - question: str          — question text with choices
    - question_type: str     — question category
    - answer: str            — ground truth answer (single letter: A, B, C, D)
    - visual_indices: int[]  — character positions where images are inserted
    - images: list[dict]     — each dict has 'bytes' (image data) and 'path'

Also supports legacy TFRecord format (data/ERQA/data/erqa.tfrecord).

Evaluation follows the official eval_harness.py from the ERQA repo:
    - Exact match: prediction.replace(".", "").strip().lower() == answer.strip().lower()
    - Breakdowns: overall, single-image, multi-image, per question_type

TFRecord parsing is done without tensorflow using struct-based binary parsing.
"""

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample, save_embedded_image
from spatial_agent.evals.scoring import get_prediction, write_results_summary


# ---------------------------------------------------------------------------
# Lightweight TFRecord + tf.train.Example parser (no tensorflow dependency)
# ---------------------------------------------------------------------------

def _masked_crc32c(data: bytes) -> int:
    """Compute masked CRC-32C as used by TFRecord format."""
    import binascii
    # TFRecord uses CRC-32C (Castagnoli). Python's binascii.crc32 is CRC-32 (ISO),
    # but we only need the CRC for integrity — skip verification for simplicity.
    return 0  # placeholder; we don't verify CRCs


def _read_tfrecord(path: str) -> Generator[bytes, None, None]:
    """Yield raw serialized Example bytes from a TFRecord file.

    TFRecord format per record:
        uint64 length
        uint32 masked_crc32_of_length
        byte   data[length]
        uint32 masked_crc32_of_data
    """
    with open(path, "rb") as f:
        while True:
            # Read length (8 bytes, little-endian uint64)
            len_bytes = f.read(8)
            if len(len_bytes) == 0:
                break  # EOF
            if len(len_bytes) < 8:
                break
            (length,) = struct.unpack("<Q", len_bytes)
            # Skip CRC of length (4 bytes)
            f.read(4)
            # Read data
            data = f.read(length)
            if len(data) < length:
                break
            # Skip CRC of data (4 bytes)
            f.read(4)
            yield data


def _parse_example(data: bytes) -> Dict[str, Any]:
    """Parse a serialized tf.train.Example protobuf into a dict.

    Returns a dict mapping feature name to its value(s).
    FixedLenFeature strings → bytes
    VarLenFeature strings → List[bytes]
    VarLenFeature int64 → List[int]
    """
    # tf.train.Example is: { features: Features }
    # Features is: { feature: map<string, Feature> }
    # Feature is oneof: { bytes_list, float_list, int64_list }
    #
    # Protobuf wire format:
    #   field_number << 3 | wire_type
    #   wire_type 0 = varint, 2 = length-delimited

    result = {}

    def read_varint(buf, pos):
        val = 0
        shift = 0
        while pos < len(buf):
            b = buf[pos]
            pos += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if (b & 0x80) == 0:
                break
        return val, pos

    def parse_feature(buf):
        """Parse a Feature message → (type, values)."""
        pos = 0
        values = []
        feat_type = None  # 'bytes', 'float', 'int64'
        while pos < len(buf):
            tag, pos = read_varint(buf, pos)
            field_num = tag >> 3
            wire_type = tag & 0x07
            if wire_type == 2:  # length-delimited
                length, pos = read_varint(buf, pos)
                inner = buf[pos:pos + length]
                pos += length
                if field_num == 1:  # bytes_list
                    feat_type = "bytes"
                    # BytesList: repeated bytes value = 1
                    ipos = 0
                    while ipos < len(inner):
                        itag, ipos = read_varint(inner, ipos)
                        if (itag >> 3) == 1 and (itag & 0x07) == 2:
                            blen, ipos = read_varint(inner, ipos)
                            values.append(inner[ipos:ipos + blen])
                            ipos += blen
                        else:
                            break
                elif field_num == 2:  # float_list
                    feat_type = "float"
                    # FloatList: repeated float value = 1 (packed)
                    ipos = 0
                    while ipos + 4 <= len(inner):
                        (v,) = struct.unpack("<f", inner[ipos:ipos + 4])
                        values.append(v)
                        ipos += 4
                elif field_num == 3:  # int64_list
                    feat_type = "int64"
                    # Int64List: repeated int64 value = 1 (packed as varints)
                    ipos = 0
                    while ipos < len(inner):
                        v, ipos = read_varint(inner, ipos)
                        # Decode as signed (zigzag is not used in tf.train.Example)
                        if v > 0x7FFFFFFFFFFFFFFF:
                            v -= 0x10000000000000000
                        values.append(v)
            elif wire_type == 0:  # varint
                _, pos = read_varint(buf, pos)
            else:
                break
        return feat_type, values

    def parse_features(buf):
        """Parse a Features message → dict of feature name to Feature."""
        pos = 0
        while pos < len(buf):
            tag, pos = read_varint(buf, pos)
            field_num = tag >> 3
            wire_type = tag & 0x07
            if wire_type == 2 and field_num == 1:
                # map entry (key=string, value=Feature)
                length, pos = read_varint(buf, pos)
                entry = buf[pos:pos + length]
                pos += length
                # Parse map entry
                epos = 0
                key = None
                feat_buf = None
                while epos < len(entry):
                    etag, epos = read_varint(entry, epos)
                    efn = etag >> 3
                    ewt = etag & 0x07
                    if ewt == 2:
                        elen, epos = read_varint(entry, epos)
                        edata = entry[epos:epos + elen]
                        epos += elen
                        if efn == 1:  # key (string)
                            key = edata.decode("utf-8")
                        elif efn == 2:  # value (Feature)
                            feat_buf = edata
                    elif ewt == 0:
                        _, epos = read_varint(entry, epos)
                    else:
                        break
                if key is not None and feat_buf is not None:
                    feat_type, values = parse_feature(feat_buf)
                    result[key] = (feat_type, values)
            elif wire_type == 2:
                length, pos = read_varint(buf, pos)
                pos += length
            elif wire_type == 0:
                _, pos = read_varint(buf, pos)
            else:
                break

    # Example message: field 1 = Features
    pos = 0
    while pos < len(data):
        tag, pos = read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2 and field_num == 1:
            length, pos = read_varint(data, pos)
            features_buf = data[pos:pos + length]
            pos += length
            parse_features(features_buf)
        elif wire_type == 2:
            length, pos = read_varint(data, pos)
            pos += length
        elif wire_type == 0:
            _, pos = read_varint(data, pos)
        else:
            break

    return result


# ---------------------------------------------------------------------------
# ERQA Benchmark
# ---------------------------------------------------------------------------

@dataclass
class ERQASample(BaseBenchmarkSample):
    """ERQA sample with visual interleaving metadata."""

    visual_indices: List[int] = field(default_factory=list)
    num_images: int = 0


class ERQABench(BaseBenchmark):
    """ERQA benchmark loader (Google DeepMind Embodied Reasoning QA)."""

    data_specific_prompt = (
        "This question is from the ERQA (Embodied Reasoning Question Answer) benchmark, "
        "which evaluates multimodal spatial reasoning and world knowledge in robotics scenarios.\n\n"
        "The question may contain <image> tags indicating where images should be viewed in context. "
        "Questions are multiple-choice with answer options embedded in the question text.\n\n"
        "Answer with ONLY the correct option letter (e.g., A, B, C, or D). "
        "Do not include any explanation or additional text in your final answer."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._images_dir = os.path.join(data_path, "extracted_images")
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        # Try parquet first (HuggingFace format), fall back to legacy tfrecord
        parquet_dir = os.path.join(self.data_path, "data")
        parquet_files = sorted(
            f for f in (os.listdir(parquet_dir) if os.path.isdir(parquet_dir) else [])
            if f.endswith(".parquet")
        )

        if parquet_files:
            self._read_parquet(parquet_dir, parquet_files)
        else:
            tfrecord_path = os.path.join(self.data_path, "data", "erqa.tfrecord")
            if not os.path.exists(tfrecord_path):
                print(f"[Warning] ERQA data not found at {self.data_path}/data/ "
                      f"(looked for .parquet and erqa.tfrecord)")
                return
            self._read_tfrecord(tfrecord_path)

    def _read_parquet(self, parquet_dir: str, parquet_files: List[str]) -> None:
        import pandas as pd

        os.makedirs(self._images_dir, exist_ok=True)
        type_counts: Dict[str, int] = {}

        dfs = [pd.read_parquet(os.path.join(parquet_dir, f)) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

        for idx, row in df.iterrows():
            question = str(row.get("question", ""))
            answer = str(row.get("answer", ""))
            q_type = str(row.get("question_type", "Unknown"))

            # Apply question_type filter
            if self.question_type_filter and q_type not in self.question_type_filter:
                continue

            # Parse visual_indices
            vi_raw = row.get("visual_indices")
            visual_indices = list(vi_raw) if vi_raw is not None else []

            # Extract images to disk
            imgs_raw = row.get("images")
            image_paths = []
            if imgs_raw is not None:
                sample_dir = os.path.join(self._images_dir, f"sample_{idx:04d}")
                for img_idx, img_entry in enumerate(imgs_raw):
                    if isinstance(img_entry, dict):
                        img_bytes = img_entry.get("bytes")
                    else:
                        img_bytes = img_entry
                    if img_bytes is None:
                        continue
                    img_path = os.path.join(sample_dir, f"img_{img_idx}.png")
                    save_embedded_image(img_path, img_bytes, convert_rgb=True)
                    image_paths.append(img_path)

            # Build question text with <image> placeholders at visual_indices
            interleaved_question = self._interleave_question(
                question, visual_indices, len(image_paths)
            )

            sample_id = str(row.get("question_id", f"erqa_{idx:04d}"))
            sample = ERQASample(
                sample_id=sample_id,
                question=interleaved_question,
                question_type=q_type,
                images=image_paths,
                answer=answer,
                visual_indices=visual_indices[:len(image_paths)],
                num_images=len(image_paths),
            )
            self.data.append(sample)
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
        print(f"[ERQA] Loaded {len(self.data)} samples from parquet ({type_str})")

    def _read_tfrecord(self, tfrecord_path: str) -> None:
        os.makedirs(self._images_dir, exist_ok=True)
        type_counts: Dict[str, int] = {}

        for idx, raw_record in enumerate(_read_tfrecord(tfrecord_path)):
            parsed = _parse_example(raw_record)

            # Extract fields
            _, q_vals = parsed.get("question", ("bytes", []))
            question = q_vals[0].decode("utf-8") if q_vals else ""

            _, a_vals = parsed.get("answer", ("bytes", []))
            answer = a_vals[0].decode("utf-8") if a_vals else ""

            _, img_vals = parsed.get("image/encoded", ("bytes", []))

            _, vi_vals = parsed.get("visual_indices", ("int64", []))
            visual_indices = list(vi_vals)

            _, qt_vals = parsed.get("question_type", ("bytes", []))
            q_type = qt_vals[0].decode("utf-8") if qt_vals else "Unknown"

            # Apply question_type filter
            if self.question_type_filter and q_type not in self.question_type_filter:
                continue

            # Extract images to disk
            image_paths = []
            sample_dir = os.path.join(self._images_dir, f"sample_{idx:04d}")
            for img_idx, img_bytes in enumerate(img_vals):
                img_path = os.path.join(sample_dir, f"img_{img_idx}.png")
                save_embedded_image(img_path, img_bytes, convert_rgb=True)
                image_paths.append(img_path)

            # Build question text with <image> placeholders at visual_indices
            interleaved_question = self._interleave_question(
                question, visual_indices, len(image_paths)
            )

            sample = ERQASample(
                sample_id=f"erqa_{idx:04d}",
                question=interleaved_question,
                question_type=q_type,
                images=image_paths,
                answer=answer,
                visual_indices=visual_indices[:len(image_paths)],
                num_images=len(image_paths),
            )
            self.data.append(sample)
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
        print(f"[ERQA] Loaded {len(self.data)} samples from tfrecord ({type_str})")

    @staticmethod
    def _interleave_question(
        question: str, visual_indices: List[int], num_images: int
    ) -> str:
        """Insert <image> placeholders into question text at visual_indices positions.

        Follows the same interleaving logic as the official eval_harness.py.
        Only the first ``num_images`` entries of ``visual_indices`` are used
        (official code does ``zip(pil_images, visual_indices)``).
        """
        if num_images == 0:
            return question

        # Only use as many indices as there are images (matches official zip behavior)
        used_indices = visual_indices[:num_images]

        # No indices or all zero → images go before question
        if len(used_indices) == 0 or all(idx == 0 for idx in used_indices):
            prefix = " ".join("<image>" for _ in range(num_images))
            return f"{prefix}\n{question}"

        # Sort (image_idx, char_position) by char_position
        pairs = sorted(enumerate(used_indices), key=lambda x: x[1])

        parts = []
        last_pos = 0
        for _, char_idx in pairs:
            if char_idx == 0:
                parts.append("<image>")
            else:
                clamped = min(char_idx, len(question))
                if clamped > last_pos:
                    parts.append(question[last_pos:clamped])
                parts.append("<image>")
                last_pos = clamped

        # Remaining text
        if last_pos < len(question):
            parts.append(question[last_pos:])

        result = "".join(parts)
        if not result.strip():
            # Fallback: all indices beyond question length
            prefix = " ".join("<image>" for _ in range(num_images))
            return f"{prefix}\n{question}"
        return result

    # ------------------------------------------------------------------
    # Answer extraction — follows official eval_harness.py exactly
    # ------------------------------------------------------------------

    def extract_answer(self, prediction: str) -> str:
        """Extract answer following the official ERQA evaluation protocol.

        Official logic (eval_harness.py line 548):
            response_text.replace(".", "").strip().lower() == answer.strip().lower()

        We normalize the prediction the same way for comparison.
        """
        if not prediction:
            return ""
        return prediction.replace(".", "").strip().lower()

    def evaluate_single(
        self, sample: BaseBenchmarkSample, prediction: str
    ) -> Optional[float]:
        """Evaluate a single prediction using official exact match."""
        pred = self.extract_answer(prediction)
        gt = sample.answer.strip().lower()
        return 1.0 if (pred and pred == gt) else 0.0

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        total = 0
        correct = 0
        single_image_total = 0
        single_image_correct = 0
        multi_image_total = 0
        multi_image_correct = 0
        type_stats: Dict[str, Dict[str, int]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)

            pred = self.extract_answer(pred_raw)
            gt = sample.answer.strip().lower()
            is_correct = pred != "" and pred == gt

            total += 1
            if is_correct:
                correct += 1

            # Single vs multi-image breakdown
            if sample.num_images <= 1:
                single_image_total += 1
                if is_correct:
                    single_image_correct += 1
            else:
                multi_image_total += 1
                if is_correct:
                    multi_image_correct += 1

            # Per question_type breakdown
            qt = sample.question_type
            if qt not in type_stats:
                type_stats[qt] = {"total": 0, "correct": 0}
            type_stats[qt]["total"] += 1
            if is_correct:
                type_stats[qt]["correct"] += 1

            detailed.append({
                "id": sid,
                "question_type": qt,
                "num_images": sample.num_images,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": pred,
                "correct": is_correct,
            })

        # Compute accuracies
        overall_acc = correct / max(total, 1)
        single_acc = single_image_correct / max(single_image_total, 1)
        multi_acc = multi_image_correct / max(multi_image_total, 1)

        per_type = {}
        for qt, stats in sorted(type_stats.items()):
            per_type[qt] = {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / max(stats["total"], 1),
            }

        results = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": overall_acc,
            "single_image_accuracy": single_acc,
            "single_image_correct": single_image_correct,
            "single_image_total": single_image_total,
            "multi_image_accuracy": multi_acc,
            "multi_image_correct": multi_image_correct,
            "multi_image_total": multi_image_total,
            "per_question_type": per_type,
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print(f"Benchmark: ERQA (Embodied Reasoning QA)")
        print(f"Total: {results['total_samples']}")
        print(f"Overall accuracy: {results['overall_accuracy']:.4f} "
              f"({results['correct_samples']}/{results['total_samples']})")
        print(f"\nSingle-image: {results['single_image_correct']}/{results['single_image_total']} "
              f"({results['single_image_accuracy']:.4f})")
        print(f"Multi-image:  {results['multi_image_correct']}/{results['multi_image_total']} "
              f"({results['multi_image_accuracy']:.4f})")
        pt = results.get("per_question_type", {})
        if pt:
            print(f"\nPer question type:")
            for qt, stats in pt.items():
                print(f"  {qt}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")
        print(f"{'='*60}\n")
