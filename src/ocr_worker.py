"""
ocr_worker.py文件
　　该文件在独立Python进程中执行PaddleOCR，避免Paddle C++段错误终止主Demo流程。
"""

import argparse
import contextlib
import json
import os
import sys

import cv2

from src.ocr import PADDLE_ENV_DEFAULTS, PlateOCR


def build_paddleocr(lang: str, device: str, fallback_to_cpu: bool, ocr_version: str, enable_mkldnn: bool):
    from paddleocr import PaddleOCR

    device_options = [device]
    if device!="cpu" and fallback_to_cpu:
        device_options.append("cpu")

    common_kwargs = {}
    if enable_mkldnn:
        common_kwargs["enable_mkldnn"] = True
    if ocr_version:
        common_kwargs["ocr_version"] = ocr_version

    init_options = [
        {"use_angle_cls": True, "lang": lang, "show_log": False},
        {"use_angle_cls": True, "lang": lang},
        {"use_textline_orientation": True, "lang": lang},
        {"lang": lang},
    ]
    last_error = None
    for selected_device in device_options:
        for kwargs in init_options:
            kwargs = dict(kwargs)
            kwargs.update(common_kwargs)
            if selected_device:
                kwargs["device"] = selected_device
            try:
                return PaddleOCR(**kwargs)
            except Exception as exc:
                last_error = exc
                if "Unknown argument" not in str(exc):
                    continue

                legacy_kwargs = dict(kwargs)
                legacy_kwargs.pop("device", None)
                legacy_kwargs.pop("ocr_version", None)
                legacy_kwargs.pop("enable_mkldnn", None)
                legacy_kwargs["use_gpu"] = str(selected_device).startswith("gpu")
                try:
                    return PaddleOCR(**legacy_kwargs)
                except Exception as legacy_exc:
                    last_error = legacy_exc
    raise last_error


def run_one_image(ocr, image_path: str) -> list:
    image = cv2.imread(image_path)
    if image is None:
        return []

    if hasattr(ocr, "predict"):
        try:
            result = ocr.predict(image_path)
            items = PlateOCR._flatten_ocr_result(result)
            return [{"text": text, "confidence": confidence} for text, confidence in items]
        except Exception:
            pass

    try:
        result = ocr.ocr(image, cls=True)
    except TypeError:
        result = ocr.ocr(image)
    items = PlateOCR._flatten_ocr_result(result)
    return [{"text": text, "confidence": confidence} for text, confidence in items]


def run_worker(image_paths: list, lang: str, device: str, fallback_to_cpu: bool, ocr_version: str, enable_mkldnn: bool) -> dict:
    for key, value in PADDLE_ENV_DEFAULTS.items():
        os.environ[key] = value

    ocr = build_paddleocr(lang, device, fallback_to_cpu, ocr_version, enable_mkldnn)
    items_by_image = {}
    for image_path in image_paths:
        items_by_image[image_path] = run_one_image(ocr, image_path)
    payload = {"items_by_image": items_by_image}
    if len(image_paths)==1:
        payload["items"] = items_by_image.get(image_paths[0], [])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ocr-version", default="")
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--fallback-to-cpu", action="store_true")
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        payload = run_worker(args.image, args.lang, args.device, args.fallback_to_cpu, args.ocr_version, args.enable_mkldnn)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
