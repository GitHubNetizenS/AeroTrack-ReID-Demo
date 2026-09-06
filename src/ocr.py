"""
ocr.py文件
　　该文件封装车牌OCR识别。默认使用PaddleOCR；若环境未安装PaddleOCR，则保留车牌裁剪图并返回UNKNOWN。
"""

import os       as os
import re       as re
import json     as json
import subprocess as subprocess
import sys      as sys
from typing     import Any, Dict, List, Tuple

import cv2      as cv2
import numpy    as np


PLATE_PATTERN = re.compile(r"([\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}|[A-Z][A-Z0-9]{5,6})")
PARTIAL_PLATE_PATTERN = re.compile(r"([A-Z0-9]{5,7})")
OCR_STOP_WORDS = {
    "STATIC",
    "MOVING",
    "ID",
    "TRACK",
    "CAR",
    "BUS",
    "TRUCK",
    "DJI",
    "M4T",
}
OCR_WATERMARK_PATTERN = re.compile(
    r"(DJI|M4T|\d{4}[-/]\d{1,2}[-/]\d{1,2}|[NSWE]\s*\d{2,3}\.\d+|\d{2}\.\d{3,}\s*[NSWE])",
    re.IGNORECASE,
)
PADDLE_ENV_DEFAULTS = {
    "FLAGS_use_onednn": "0",
    "FLAGS_use_mkldnn": "0",
    "FLAGS_use_pir_api": "0",
    "FLAGS_enable_pir_api": "0",
    "FLAGS_enable_pir_in_executor": "0",
    "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


class PlateOCR:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config.get("ocr", {})
        self.enabled = self.config.get("enabled", True)
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.device = self.config.get("device", "cpu")
        self.fallback_to_cpu = self.config.get("fallback_to_cpu", True)
        self.ocr_version = self.config.get("ocr_version", "")
        self.enable_mkldnn = self.config.get("enable_mkldnn", False)
        self.plate_detector_enabled = self.config.get("plate_detector_enabled", True)
        self.max_plate_candidates = self.config.get("max_plate_candidates", 5)
        self.plate_candidate_min_area_ratio = self.config.get("plate_candidate_min_area_ratio", 0.0008)
        self.plate_candidate_max_area_ratio = self.config.get("plate_candidate_max_area_ratio", 0.12)
        self.plate_candidate_aspect_ratio_range = self.config.get("plate_candidate_aspect_ratio_range", [1.6, 7.5])
        self.plate_candidate_min_center_y_ratio = self.config.get("plate_candidate_min_center_y_ratio", 0.08)
        self.plate_candidate_min_top_margin_ratio = self.config.get("plate_candidate_min_top_margin_ratio", 0.025)
        self.plate_preprocess_scale = self.config.get("plate_preprocess_scale", 2.5)
        self.plate_preprocess_modes = self.config.get("plate_preprocess_modes", ["raw", "clahe", "sharp"])
        self.min_ocr_crop_width = self.config.get("min_ocr_crop_width", 18)
        self.min_ocr_crop_height = self.config.get("min_ocr_crop_height", 8)
        self.max_ocr_crop_pixels = self.config.get("max_ocr_crop_pixels", 900000)
        self.max_preprocess_long_side = self.config.get("max_preprocess_long_side", 960)
        self.allow_partial_plate_text = self.config.get("allow_partial_plate_text", True)
        self.isolate_paddle_process = self.config.get("isolate_paddle_process", True)
        self.ocr_timeout_sec = self.config.get("ocr_timeout_sec", 30)
        self.max_subprocess_failures = self.config.get("max_subprocess_failures", 20)
        self.subprocess_failures = 0
        self.debug_rows: List[Dict[str, Any]] = []
        self.ocr = None

        if not self.enabled:
            return

        for key, value in PADDLE_ENV_DEFAULTS.items():
            os.environ[key] = value

        if self.isolate_paddle_process:
            print("PaddleOCR将以子进程隔离模式运行，避免Paddle C++段错误终止主流程。")
            return

        try:
            from paddleocr import PaddleOCR
            self.ocr = self._build_paddleocr(PaddleOCR)
        except Exception as exc:
            print(f"PaddleOCR不可用，车牌文本将标记为UNKNOWN：{exc}")
            self.ocr = None

    def _build_paddleocr(self, paddleocr_cls):
        lang = self.config.get("lang", "ch")
        device_options = [self.device]
        if self.device!="cpu" and self.fallback_to_cpu:
            device_options.append("cpu")

        common_kwargs = {}
        if self.ocr_version:
            common_kwargs["ocr_version"] = self.ocr_version
        if self.enable_mkldnn:
            common_kwargs["enable_mkldnn"] = True

        init_options = [
            {"use_angle_cls": True, "lang": lang, "show_log": False},
            {"use_angle_cls": True, "lang": lang},
            {"use_textline_orientation": True, "lang": lang},
            {"lang": lang},
        ]
        last_error = None
        for device in device_options:
            for kwargs in init_options:
                kwargs = dict(kwargs)
                kwargs.update(common_kwargs)
                if device:
                    kwargs["device"] = device
                try:
                    print(f"PaddleOCR初始化设备：{device or 'default'}")
                    return paddleocr_cls(**kwargs)
                except Exception as exc:
                    last_error = exc
                    unknown_arg_error = "Unknown argument" in str(exc)
                    if not unknown_arg_error:
                        continue

                    legacy_kwargs = dict(kwargs)
                    legacy_kwargs.pop("device", None)
                    legacy_kwargs.pop("ocr_version", None)
                    legacy_kwargs.pop("enable_mkldnn", None)
                    legacy_kwargs["use_gpu"] = str(device).startswith("gpu")
                    try:
                        print(f"PaddleOCR使用旧版use_gpu参数初始化：use_gpu={legacy_kwargs['use_gpu']}")
                        return paddleocr_cls(**legacy_kwargs)
                    except Exception as legacy_exc:
                        last_error = legacy_exc
        raise last_error

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.upper().replace(" ", "").replace("-", "")
        text = text.replace("·", "").replace(".", "").replace(":", "")
        for stop_word in OCR_STOP_WORDS:
            text = text.replace(stop_word, "")
        match = PLATE_PATTERN.search(text)
        if match:
            candidate = match.group(0)
            if any(char.isdigit() for char in candidate):
                return candidate
        return "UNKNOWN"

    @staticmethod
    def _looks_like_watermark(text: str) -> bool:
        return bool(OCR_WATERMARK_PATTERN.search(str(text)))

    def _normalize_ocr_text(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if normalized!="UNKNOWN" or not self.allow_partial_plate_text:
            return normalized

        text = text.upper().replace(" ", "").replace("-", "")
        text = text.replace("·", "").replace(".", "").replace(":", "")
        for stop_word in OCR_STOP_WORDS:
            text = text.replace(stop_word, "")

        for match in PARTIAL_PLATE_PATTERN.finditer(text):
            candidate = match.group(0)
            digit_count = sum(char.isdigit() for char in candidate)
            if digit_count>=3:
                return candidate
        return "UNKNOWN"

    def consume_debug_rows(self) -> List[Dict[str, Any]]:
        rows = self.debug_rows
        self.debug_rows = []
        return rows

    @staticmethod
    def _candidate_plate_regions(vehicle_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        height, width = vehicle_bgr.shape[:2]
        candidates = []
        candidates.append((int(width*0.05), int(height*0.62), int(width*0.95), height))
        candidates.append((int(width*0.15), int(height*0.68), int(width*0.85), height))
        lower_y = int(height*0.35)
        candidates.append((0, lower_y, width, height))
        candidates.append((int(width*0.05), int(height*0.45), int(width*0.95), height))
        candidates.append((int(width*0.15), int(height*0.55), int(width*0.85), height))
        candidates.append((0, int(height*0.2), width, int(height*0.85)))
        candidates.append((0, 0, width, height))
        return candidates

    @staticmethod
    def _clip_box(box: Tuple[int, int, int, int], width: int, height: int, pad_ratio: float = 0.08) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        box_width = max(1, x2-x1)
        box_height = max(1, y2-y1)
        pad_x = int(box_width*pad_ratio)
        pad_y = int(box_height*pad_ratio)
        x1 = max(0, x1-pad_x)
        y1 = max(0, y1-pad_y)
        x2 = min(width, x2+pad_x)
        y2 = min(height, y2+pad_y)
        return x1, y1, x2, y2

    @staticmethod
    def _box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0, inter_x2-inter_x1)*max(0, inter_y2-inter_y1)
        area_a = max(1, (ax2-ax1)*(ay2-ay1))
        area_b = max(1, (bx2-bx1)*(by2-by1))
        return inter_area/(area_a+area_b-inter_area+1e-6)

    def _dedupe_candidates(self, candidates: List[Tuple[int, int, int, int, str]], limit: int = None) -> List[Tuple[int, int, int, int, str]]:
        deduped = []
        for candidate in candidates:
            box = candidate[:4]
            if any(self._box_iou(box, kept[:4])>0.72 for kept in deduped):
                continue
            deduped.append(candidate)
            if limit is not None and len(deduped)>=limit:
                break
        return deduped

    def _detect_plate_candidates(self, vehicle_bgr: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        if not self.plate_detector_enabled or vehicle_bgr.size==0:
            return []

        height, width = vehicle_bgr.shape[:2]
        image_area = max(1, height*width)
        aspect_min, aspect_max = self.plate_candidate_aspect_ratio_range
        hsv = cv2.cvtColor(vehicle_bgr, cv2.COLOR_BGR2HSV)

        color_ranges = [
            (np.array([90, 50, 50]), np.array([135, 255, 255])),   # blue plates
            (np.array([35, 70, 70]), np.array([90, 255, 255])),    # green new-energy plates
            (np.array([15, 55, 70]), np.array([38, 255, 255])),    # yellow plates
        ]
        mask = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w<=0 or h<=0:
                continue
            area_ratio = (w*h)/image_area
            aspect_ratio = w/max(1, h)
            if area_ratio<self.plate_candidate_min_area_ratio or area_ratio>self.plate_candidate_max_area_ratio:
                continue
            if aspect_ratio<aspect_min or aspect_ratio>aspect_max:
                continue
            if y/max(1, height)<self.plate_candidate_min_top_margin_ratio:
                continue
            center_y_ratio = (y+h*0.5)/max(1, height)
            if center_y_ratio<self.plate_candidate_min_center_y_ratio:
                continue

            box = self._clip_box((x, y, x+w, y+h), width, height)
            lower_position_score = y/max(1, height)
            color_area_score = cv2.contourArea(contour)/max(1, w*h)
            score = area_ratio*4.0 + lower_position_score*0.35 + color_area_score*0.25
            scored_candidates.append((score, *box, "color"))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = [(x1, y1, x2, y2, source) for _, x1, y1, x2, y2, source in scored_candidates]

        fixed_regions = [(x1, y1, x2, y2, "region") for x1, y1, x2, y2 in self._candidate_plate_regions(vehicle_bgr)]
        fixed_regions = self._dedupe_candidates(fixed_regions)
        candidates = self._dedupe_candidates(candidates)

        selected = []
        for candidate in fixed_regions[:2] + candidates + fixed_regions[2:]:
            if any(self._box_iou(candidate[:4], kept[:4])>0.72 for kept in selected):
                continue
            selected.append(candidate)
            if len(selected)>=self.max_plate_candidates:
                break
        return selected

    def _preprocess_plate_variants(self, plate_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        if not self._is_valid_ocr_image(plate_bgr):
            return []

        height, width = plate_bgr.shape[:2]
        scale = max(1.0, float(self.plate_preprocess_scale))
        target_width = max(width, int(width*scale))
        target_height = max(height, int(height*scale))
        long_side = max(target_width, target_height)
        if long_side>self.max_preprocess_long_side:
            resize_scale = self.max_preprocess_long_side/long_side
            target_width = max(1, int(target_width*resize_scale))
            target_height = max(1, int(target_height*resize_scale))
        resized = cv2.resize(plate_bgr, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        modes = set(getattr(self, "plate_preprocess_modes", ["raw", "clahe", "sharp", "binary"]))
        variants = []
        if "raw" in modes:
            variants.append(("raw", resized))

        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        if "clahe" in modes:
            variants.append(("clahe", enhanced))

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
        if "sharp" in modes:
            variants.append(("sharp", sharpened))

        if "binary" in modes:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                9,
            )
            variants.append(("binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))
        return variants

    def _is_valid_ocr_image(self, image_bgr: np.ndarray) -> bool:
        if image_bgr is None or image_bgr.size==0:
            return False
        height, width = image_bgr.shape[:2]
        if width<self.min_ocr_crop_width or height<self.min_ocr_crop_height:
            return False
        if width*height>self.max_ocr_crop_pixels:
            return False
        return True

    @staticmethod
    def _flatten_ocr_result(result) -> List[Tuple[str, float]]:
        items = []
        if result is None:
            return items

        if isinstance(result, dict):
            for key in ("rec_texts", "texts"):
                if key in result:
                    scores = result.get("rec_scores", result.get("scores", [0.0]*len(result[key])))
                    for text, score in zip(result[key], scores):
                        items.append((str(text), float(score)))
                    return items
            for value in result.values():
                items.extend(PlateOCR._flatten_ocr_result(value))
            return items

        if isinstance(result, (list, tuple)):
            if len(result)==2 and isinstance(result[0], str):
                try:
                    return [(result[0], float(result[1]))]
                except Exception:
                    return []
            for value in result:
                items.extend(PlateOCR._flatten_ocr_result(value))
            return items

        json_data = getattr(result, "json", None)
        if isinstance(json_data, dict):
            return PlateOCR._flatten_ocr_result(json_data)
        return items

    def _run_ocr(self, crop: np.ndarray, crop_path: str) -> List[Tuple[str, float]]:
        if self.isolate_paddle_process:
            return self._run_ocr_subprocess(crop_path)

        if self.ocr is None:
            return []

        if hasattr(self.ocr, "predict"):
            try:
                result = self.ocr.predict(crop_path)
                return self._flatten_ocr_result(result)
            except Exception as exc:
                print(f"PaddleOCR新版predict接口识别失败，尝试旧版ocr接口：{exc}")

        try:
            result = self.ocr.ocr(crop, cls=True)
            return self._flatten_ocr_result(result)
        except TypeError:
            result = self.ocr.ocr(crop)
            return self._flatten_ocr_result(result)

    def _run_ocr_subprocess(self, crop_path: str) -> List[Tuple[str, float]]:
        return self._run_ocr_subprocess_batch([crop_path]).get(crop_path, [])

    def _run_ocr_subprocess_batch(self, crop_paths: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        if not crop_paths:
            return {}

        if self.subprocess_failures>=self.max_subprocess_failures:
            return {crop_path: [] for crop_path in crop_paths}

        env = os.environ.copy()
        for key, value in PADDLE_ENV_DEFAULTS.items():
            env[key] = value

        command = [
            sys.executable,
            "-m",
            "src.ocr_worker",
            "--lang",
            self.config.get("lang", "ch"),
            "--device",
            self.device,
        ]
        if self.ocr_version:
            command.extend(["--ocr-version", self.ocr_version])
        if self.enable_mkldnn:
            command.append("--enable-mkldnn")
        if self.fallback_to_cpu:
            command.append("--fallback-to-cpu")
        for crop_path in crop_paths:
            command.extend(["--image", crop_path])

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=float(self.ocr_timeout_sec),
                env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"OCR子进程超时，跳过当前批次的{len(crop_paths)}个候选区域。")
            self.subprocess_failures += 1
            return {crop_path: [] for crop_path in crop_paths}

        if completed.returncode!=0:
            stderr = (completed.stderr or "").strip()
            if stderr:
                stderr = " | ".join(stderr.splitlines()[-3:])
            print(f"OCR子进程失败(returncode={completed.returncode})，跳过当前批次：{stderr or completed.returncode}")
            self.subprocess_failures += 1
            if self.subprocess_failures>=self.max_subprocess_failures:
                print("OCR子进程连续失败次数过多，本次运行将跳过后续OCR，仅保留车牌候选图。")
            return {crop_path: [] for crop_path in crop_paths}

        try:
            stdout = (completed.stdout or "").strip()
            if not stdout.startswith("{"):
                stdout = stdout[stdout.rfind("{"):]
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            print("OCR子进程输出无法解析，跳过当前批次。")
            self.subprocess_failures += 1
            return {crop_path: [] for crop_path in crop_paths}
        self.subprocess_failures = 0
        items_by_image = payload.get("items_by_image")
        if items_by_image is None and len(crop_paths)==1:
            items_by_image = {crop_paths[0]: payload.get("items", [])}

        result: Dict[str, List[Tuple[str, float]]] = {}
        for crop_path in crop_paths:
            result[crop_path] = [
                (str(item["text"]), float(item["confidence"]))
                for item in (items_by_image or {}).get(crop_path, [])
            ]
        return result

    def recognize(self, vehicle_bgr: np.ndarray, output_dir: str, prefix: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        best = {
            "plate_text": "UNKNOWN",
            "plate_confidence": 0.0,
            "plate_crop_path": "",
            "reject_crop": False,
        }
        variant_jobs = []

        candidates = self._detect_plate_candidates(vehicle_bgr)
        for index, (x1, y1, x2, y2, source) in enumerate(candidates):
            crop = vehicle_bgr[y1:y2, x1:x2]
            if not self._is_valid_ocr_image(crop):
                continue

            plate_crop_path = os.path.join(output_dir, f"{prefix}_plate_{index:02d}_{source}.jpg")
            cv2.imwrite(plate_crop_path, crop)
            if not best["plate_crop_path"]:
                best["plate_crop_path"] = plate_crop_path

            if self.ocr is None and not self.isolate_paddle_process:
                continue

            variants = self._preprocess_plate_variants(crop)
            for variant_name, variant_image in variants:
                variant_path = os.path.join(output_dir, f"{prefix}_plate_{index:02d}_{source}_{variant_name}.jpg")
                cv2.imwrite(variant_path, variant_image)
                variant_jobs.append((variant_path, variant_image))

        if self.isolate_paddle_process:
            ocr_batch = self._run_ocr_subprocess_batch([path for path, _ in variant_jobs])
        else:
            ocr_batch = {}

        for variant_path, variant_image in variant_jobs:
            try:
                if self.isolate_paddle_process:
                    ocr_items = ocr_batch.get(variant_path, [])
                else:
                    ocr_items = self._run_ocr(variant_image, variant_path)
            except Exception as exc:
                print(f"OCR识别失败，跳过当前候选区域：{exc}")
                continue

            for text, confidence in ocr_items:
                normalized = self._normalize_ocr_text(text)
                confidence = float(confidence)
                if self._looks_like_watermark(text):
                    best["reject_crop"] = True
                self.debug_rows.append({
                    "prefix": prefix,
                    "variant_path": variant_path,
                    "raw_text": str(text),
                    "normalized_text": normalized,
                    "confidence": confidence,
                })
                if normalized!="UNKNOWN" and confidence>=best["plate_confidence"]:
                    best = {
                        "plate_text": normalized,
                        "plate_confidence": confidence,
                        "plate_crop_path": variant_path,
                        "reject_crop": best.get("reject_crop", False),
                    }

        if best["plate_confidence"]<self.min_confidence:
            best["plate_text"] = "UNKNOWN"
        return best

    @staticmethod
    def _select_crop_paths(crop_paths: List[str], max_crops: int) -> List[str]:
        if len(crop_paths)<=max_crops:
            return crop_paths

        index_by_path = {path: index for index, path in enumerate(crop_paths)}
        uniform_count = max(1, max_crops//2)
        uniform_indexes = np.linspace(0, len(crop_paths)-1, uniform_count).astype(int).tolist()
        selected = {crop_paths[index] for index in uniform_indexes}

        remaining_slots = max_crops-len(selected)
        if remaining_slots>0:
            paths_by_size = sorted(
                crop_paths,
                key=lambda path: os.path.getsize(path) if os.path.exists(path) else 0,
                reverse=True,
            )
            for path in paths_by_size:
                selected.add(path)
                if len(selected)>=max_crops:
                    break

        return sorted(selected, key=lambda path: index_by_path[path])

    def recognize_many(self, crop_paths: List[str], output_dir: str, prefix: str) -> Dict[str, Any]:
        max_crops = self.config.get("max_crops_per_track", 8)
        if not crop_paths:
            return {
                "plate_text": "UNKNOWN",
                "plate_confidence": 0.0,
                "plate_crop_path": "",
            }

        selected_paths = self._select_crop_paths(crop_paths, max_crops)

        best = {
            "plate_text": "UNKNOWN",
            "plate_confidence": 0.0,
            "plate_crop_path": "",
            "plate_observations": [],
            "rejected_crop_paths": [],
        }
        vote_scores: Dict[str, float] = {}
        vote_best: Dict[str, Dict[str, Any]] = {}
        observations = []
        rejected_crop_paths = []

        for index, crop_path in enumerate(selected_paths):
            image = cv2.imread(crop_path)
            if image is None or image.size==0:
                continue
            result = self.recognize(image, output_dir, f"{prefix}_sample_{index:02d}")
            plate_text = result.get("plate_text", "UNKNOWN")
            if result.get("reject_crop", False) and (plate_text=="UNKNOWN" or not PLATE_PATTERN.match(plate_text)):
                rejected_crop_paths.append(crop_path)
            confidence = float(result.get("plate_confidence", 0.0))
            if plate_text=="UNKNOWN":
                continue
            observations.append({
                "source_crop_path": crop_path,
                "plate_text": plate_text,
                "plate_confidence": confidence,
                "plate_crop_path": result.get("plate_crop_path", ""),
            })
            vote_scores[plate_text] = vote_scores.get(plate_text, 0.0) + confidence
            if plate_text not in vote_best or confidence>vote_best[plate_text]["plate_confidence"]:
                vote_best[plate_text] = result

        if vote_scores:
            best_plate = max(vote_scores.items(), key=lambda item: item[1])[0]
            best = vote_best[best_plate]
        best["plate_observations"] = observations
        best["rejected_crop_paths"] = rejected_crop_paths
        return best
