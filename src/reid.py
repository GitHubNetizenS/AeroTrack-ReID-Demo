"""
reid.py文件
　　该文件封装车辆外观特征提取与聚类。用于将同一辆车在不同角度下的截图归档到同一目录。
"""

import os       as os
import re       as re
import shutil   as shutil
from typing     import Any, Dict, List

import cv2      as cv2
import numpy    as np
import torch    as torch
from sklearn.cluster import DBSCAN

from src.postprocess import CHINESE_PLATE_PATTERN


def safe_folder_name(name: str) -> str:
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name.strip() or "UNKNOWN"


def parse_track_id_from_crop_path(crop_path: str) -> str:
    match = re.search(r"track_(\d+)_", os.path.basename(str(crop_path)))
    if not match:
        return ""
    return str(int(match.group(1)))


def read_image_bgr(image_path: str) -> Any:
    if not image_path or not os.path.exists(image_path):
        return None
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size==0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class VehicleReID:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config.get("reid", {})
        self.cluster_eps = self.config.get("cluster_eps", 0.35)
        self.min_samples = self.config.get("min_samples", 1)
        self.merge_by_appearance = self.config.get("merge_by_appearance", False)
        self.appearance_quality_check = self.config.get("appearance_quality_check", True)
        self.appearance_warning_threshold = float(self.config.get("appearance_warning_threshold", 0.45))
        self.appearance_track_warning_threshold = float(self.config.get("appearance_track_warning_threshold", 0.50))
        self.appearance_review_max_crops_per_vehicle = int(self.config.get("appearance_review_max_crops_per_vehicle", 32))
        self.device = config.get("model", {}).get("device", "cuda:0")
        self.device = self.device if torch.cuda.is_available() and "cuda" in self.device else "cpu"
        self.model = None
        self.transforms = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            from torchvision import models, transforms
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            model.to(self.device)
            self.model = model
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as exc:
            print(f"ResNet50 ReID模型不可用，将使用颜色直方图特征：{exc}")
            self.model = None
            self.transforms = None

    def extract_feature(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.model is None or self.transforms is None:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist.astype(np.float32)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transforms(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(tensor).cpu().numpy()[0]
        norm = np.linalg.norm(feature)
        if norm>0:
            feature = feature/norm
        return feature.astype(np.float32)

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm<=0 or right_norm<=0:
            return 0.0
        return float(np.dot(left, right)/(left_norm*right_norm))

    def sample_crop_paths(self, crop_paths: List[str]) -> List[str]:
        max_count = max(1, self.appearance_review_max_crops_per_vehicle)
        unique_paths = sorted(set(crop_paths))
        if len(unique_paths)<=max_count:
            return unique_paths
        indices = np.linspace(0, len(unique_paths)-1, max_count).round().astype(int)
        return [unique_paths[int(index)] for index in indices]

    def build_appearance_review(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.appearance_quality_check:
            return []

        review_rows = []
        for record in records:
            vehicle_id = str(record.get("vehicle_id", ""))
            crop_paths = self.sample_crop_paths(record.get("crop_paths", []))
            feature_items = []
            for crop_path in crop_paths:
                image = read_image_bgr(crop_path)
                if image is None:
                    review_rows.append({
                        "vehicle_id": vehicle_id,
                        "check_scope": "read_image",
                        "track_id": parse_track_id_from_crop_path(crop_path),
                        "crop_path": crop_path,
                        "nearest_crop_path": "",
                        "similarity": "",
                        "warning": "image_read_failed",
                    })
                    continue
                feature_items.append({
                    "crop_path": crop_path,
                    "track_id": parse_track_id_from_crop_path(crop_path),
                    "feature": self.extract_feature(image),
                })

            review_rows.extend(self.review_feature_group(
                vehicle_id,
                "vehicle",
                feature_items,
                self.appearance_warning_threshold,
            ))

            track_ids = sorted(set(item["track_id"] for item in feature_items if item["track_id"]))
            for track_id in track_ids:
                track_items = [item for item in feature_items if item["track_id"]==track_id]
                review_rows.extend(self.review_feature_group(
                    vehicle_id,
                    "track",
                    track_items,
                    self.appearance_track_warning_threshold,
                ))

        return review_rows

    def review_feature_group(self, vehicle_id: str, check_scope: str, feature_items: List[Dict[str, Any]],
                             threshold: float) -> List[Dict[str, Any]]:
        rows = []
        if len(feature_items)<2:
            for item in feature_items:
                rows.append({
                    "vehicle_id": vehicle_id,
                    "check_scope": check_scope,
                    "track_id": item.get("track_id", ""),
                    "crop_path": item.get("crop_path", ""),
                    "nearest_crop_path": "",
                    "similarity": "",
                    "warning": "single_crop",
                })
            return rows

        for index, item in enumerate(feature_items):
            best_similarity = -1.0
            best_crop_path = ""
            for other_index, other_item in enumerate(feature_items):
                if index==other_index:
                    continue
                similarity = self.cosine_similarity(item["feature"], other_item["feature"])
                if similarity>best_similarity:
                    best_similarity = similarity
                    best_crop_path = other_item["crop_path"]
            rows.append({
                "vehicle_id": vehicle_id,
                "check_scope": check_scope,
                "track_id": item.get("track_id", ""),
                "crop_path": item.get("crop_path", ""),
                "nearest_crop_path": best_crop_path,
                "similarity": best_similarity,
                "warning": "low_similarity" if best_similarity<threshold else "",
            })
        return rows

    def cluster(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []

        if not self.merge_by_appearance:
            plate_to_label = {}
            next_label = 0
            for record in records:
                plate_text = record.get("plate_text", "UNKNOWN")
                plate_confidence = float(record.get("plate_confidence", 0.0))
                if plate_text!="UNKNOWN" and plate_confidence>=0.8:
                    if plate_text not in plate_to_label:
                        plate_to_label[plate_text] = next_label
                        next_label += 1
                    record["cluster_label"] = plate_to_label[plate_text]
                else:
                    record["cluster_label"] = next_label
                    next_label += 1
            return records

        feature_list = []
        for record in records:
            image = read_image_bgr(record["best_crop_path"])
            if image is None:
                feature_list.append(np.zeros((384,), dtype=np.float32))
                continue
            feature_list.append(self.extract_feature(image))

        features = np.vstack(feature_list)
        metric = "cosine" if self.model is not None else "euclidean"
        labels = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples, metric=metric).fit_predict(features)

        plate_to_label = {}
        next_label = int(labels.max()+1) if len(labels)>0 else 0
        for index, record in enumerate(records):
            plate_text = record.get("plate_text", "UNKNOWN")
            if plate_text!="UNKNOWN":
                if plate_text not in plate_to_label:
                    plate_to_label[plate_text] = labels[index] if labels[index]>=0 else next_label
                    if labels[index]<0:
                        next_label += 1
                labels[index] = plate_to_label[plate_text]

        if np.any(labels<0):
            for index, label in enumerate(labels):
                if label<0:
                    labels[index] = next_label
                    next_label += 1

        for index, record in enumerate(records):
            record["cluster_label"] = int(labels[index])
        return records


def export_clustered_records(records: List[Dict[str, Any]], vehicles_dir: str) -> List[Dict[str, Any]]:
    os.makedirs(vehicles_dir, exist_ok=True)
    label_to_vehicle = {}
    exported = []

    for record in sorted(records, key=lambda item: (item.get("cluster_label", 0), item.get("first_frame", 0))):
        label = record.get("cluster_label", 0)
        plate_text = str(record.get("plate_text", ""))
        record_vehicle_id = str(record.get("vehicle_id", ""))
        if CHINESE_PLATE_PATTERN.match(plate_text):
            vehicle_id = safe_folder_name(plate_text)
        elif record_vehicle_id and not record_vehicle_id.startswith("vehicle_"):
            vehicle_id = safe_folder_name(record_vehicle_id)
        else:
            if label not in label_to_vehicle:
                label_to_vehicle[label] = f"vehicle_{len(label_to_vehicle)+1:04d}"
            vehicle_id = label_to_vehicle[label]
        vehicle_dir = os.path.join(vehicles_dir, vehicle_id)
        os.makedirs(vehicle_dir, exist_ok=True)

        crop_paths = []
        for crop_path in record.get("crop_paths", []):
            if not os.path.exists(crop_path):
                continue
            dst = os.path.join(vehicle_dir, os.path.basename(crop_path))
            if os.path.abspath(crop_path)!=os.path.abspath(dst):
                shutil.copy2(crop_path, dst)
            crop_paths.append(dst)

        record["vehicle_id"] = vehicle_id
        record["output_dir"] = vehicle_dir
        record["crop_paths"] = crop_paths
        exported.append(record)

    return exported
