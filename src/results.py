"""
results.py文件
　　该文件负责保存Demo阶段的车辆截图、CSV摘要与JSON清单。
"""

import csv      as csv
import json     as json
import os       as os
from typing     import Any, Dict, List

import cv2      as cv2
import numpy    as np


SUMMARY_FIELDS = [
    "vehicle_id",
    "track_ids",
    "plate_text",
    "plate_confidence",
    "first_frame",
    "last_frame",
    "crop_count",
    "output_dir",
]

OCR_DEBUG_FIELDS = [
    "prefix",
    "variant_path",
    "raw_text",
    "normalized_text",
    "confidence",
]

APPEARANCE_REVIEW_FIELDS = [
    "vehicle_id",
    "check_scope",
    "track_id",
    "crop_path",
    "nearest_crop_path",
    "similarity",
    "warning",
]


def crop_with_padding(frame_bgr: np.ndarray, bbox: List[float], padding_ratio: float = 0.08) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2<=x1 or y2<=y1:
        return np.empty((0, 0, 3), dtype=frame_bgr.dtype)

    bw, bh = x2-x1, y2-y1
    pad_x, pad_y = bw*padding_ratio, bh*padding_ratio
    x1 = int(max(0, x1-pad_x))
    y1 = int(max(0, y1-pad_y))
    x2 = int(min(width, x2+pad_x))
    y2 = int(min(height, y2+pad_y))
    if x2<=x1 or y2<=y1:
        return np.empty((0, 0, 3), dtype=frame_bgr.dtype)
    return frame_bgr[y1:y2, x1:x2].copy()


def save_vehicle_crop(frame_bgr: np.ndarray, bbox: List[float], output_dir: str, track_id: int,
                      frame_index: int, padding_ratio: float = 0.08) -> str:
    os.makedirs(output_dir, exist_ok=True)
    crop = crop_with_padding(frame_bgr, bbox, padding_ratio=padding_ratio)
    if crop.size==0:
        return ""
    crop_path = os.path.join(output_dir, f"track_{track_id:04d}_frame_{frame_index:06d}.jpg")
    if not cv2.imwrite(crop_path, crop):
        return ""
    return crop_path


def write_summary_csv(records: List[Dict[str, Any]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "vehicle_id": record.get("vehicle_id", ""),
                "track_ids": "|".join(str(item) for item in record.get("track_ids", [])),
                "plate_text": record.get("plate_text", "UNKNOWN"),
                "plate_confidence": f"{record.get('plate_confidence', 0.0):.4f}",
                "first_frame": record.get("first_frame", ""),
                "last_frame": record.get("last_frame", ""),
                "crop_count": len(record.get("crop_paths", [])),
                "output_dir": record.get("output_dir", ""),
            })


def write_ocr_debug_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OCR_DEBUG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "prefix": row.get("prefix", ""),
                "variant_path": row.get("variant_path", ""),
                "raw_text": row.get("raw_text", ""),
                "normalized_text": row.get("normalized_text", ""),
                "confidence": f"{float(row.get('confidence', 0.0)):.4f}",
            })


def write_appearance_review_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=APPEARANCE_REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            similarity = row.get("similarity", "")
            writer.writerow({
                "vehicle_id": row.get("vehicle_id", ""),
                "check_scope": row.get("check_scope", ""),
                "track_id": row.get("track_id", ""),
                "crop_path": row.get("crop_path", ""),
                "nearest_crop_path": row.get("nearest_crop_path", ""),
                "similarity": "" if similarity=="" else f"{float(similarity):.4f}",
                "warning": row.get("warning", ""),
            })


def write_manifest(records: List[Dict[str, Any]], manifest_path: str, config: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    payload = {
        "config": config,
        "vehicles": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def draw_label(frame: np.ndarray, bbox: List[float], label: str, color: tuple) -> None:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, label, (x1, max(30, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
