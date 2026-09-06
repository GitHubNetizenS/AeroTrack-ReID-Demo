"""
demo_pipeline.py文件
    该文件为完整Demo流水线。用于处理无人机视频流，完成辅路车辆检测、跟踪、静止判定、车牌OCR、ReID聚类归档与Demo视频导出。
"""

import os       as os
import cv2      as cv2
import numpy    as np

from baseline_sahi_inference import load_config
from src.detector import SAHIYOLOv11Detector
from src.ocr import PlateOCR
from src.postprocess import consolidate_vehicle_records
from src.reid import VehicleReID, export_clustered_records
from src.results import (
    draw_label,
    save_vehicle_crop,
    write_appearance_review_csv,
    write_manifest,
    write_ocr_debug_csv,
    write_summary_csv,
)
from src.roi import get_segment_for_frame, interpolate_roi
from src.tracker import AeroTracker


def ensure_video_available(video_path: str) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件{video_path}不存在，请检查configs/demo_config.yaml中的io.video_path。")


def write_output_frame(writer, frame, output_scale: float) -> None:
    if output_scale!=1.0:
        frame = cv2.resize(frame, None, fx=output_scale, fy=output_scale, interpolation=cv2.INTER_AREA)
    writer.write(frame)


def parse_track(track: np.ndarray) -> tuple:
    x1, y1, x2, y2 = [float(item) for item in track[:4]]
    track_id = int(track[4])
    confidence = float(track[5]) if len(track)>5 else 0.0
    cls_id = int(track[6]) if len(track)>6 else -1
    return [x1, y1, x2, y2], track_id, confidence, cls_id


def maybe_save_static_crop(frame, bbox, track_id, frame_index, state, runtime_config, raw_crops_dir) -> None:
    max_crops = runtime_config.get("max_crops_per_track", 8)
    save_every = runtime_config.get("save_static_every", 30)
    if state.crop_paths:
        last_crop_name = os.path.basename(state.crop_paths[-1])
        try:
            last_frame = int(last_crop_name.split("_frame_")[-1].split(".")[0])
            if frame_index-last_frame<save_every:
                return
        except Exception:
            pass

    crop_path = save_vehicle_crop(frame, bbox, raw_crops_dir, track_id, frame_index)
    if not crop_path:
        return
    state.crop_paths.append(crop_path)
    if len(state.crop_paths)>max_crops:
        removed_path = state.crop_paths.pop(0)
        if state.best_crop_path==removed_path:
            state.best_crop_path = ""
            state.best_confidence = 0.0
    if state.best_crop_path=="" or state.best_confidence<=state.confidences[-1]:
        state.best_crop_path = crop_path
        state.best_confidence = state.confidences[-1]


def build_supplemental_crop_schedule(config: dict) -> dict:
    crop_config = config.get("supplemental_crops", {})
    if not crop_config.get("enabled", False):
        return {}

    schedule = {}
    for index, item in enumerate(crop_config.get("items", []), start=1):
        if "frame" not in item or "bbox" not in item:
            raise ValueError("supplemental_crops.items中的每一项都必须包含frame和bbox。")
        bbox = item["bbox"]
        if len(bbox)!=4:
            raise ValueError("supplemental_crops.items[].bbox必须为[x1, y1, x2, y2]。")

        normalized = dict(item)
        normalized["frame"] = int(item["frame"])
        normalized["bbox"] = [float(value) for value in bbox]
        normalized["track_id"] = int(item.get("track_id", 900000+index))
        normalized["padding_ratio"] = float(item.get("padding_ratio", 0.02))
        schedule.setdefault(normalized["frame"], []).append(normalized)
    return schedule


def maybe_save_supplemental_crops(frame, frame_index, schedule, raw_crops_dir, supplemental_records) -> None:
    for item in schedule.get(frame_index, []):
        track_id = item["track_id"]
        crop_path = save_vehicle_crop(
            frame,
            item["bbox"],
            raw_crops_dir,
            track_id,
            frame_index,
            padding_ratio=item.get("padding_ratio", 0.02),
        )
        if not crop_path:
            print(f"补充截图失败：frame={frame_index}, track_id={track_id}", flush=True)
            continue

        plate_hint = str(item.get("plate_hint", "")).strip()
        record = supplemental_records.setdefault(track_id, {
            "track_ids": [track_id],
            "plate_text": plate_hint or "UNKNOWN",
            "plate_confidence": 0.99 if plate_hint else 0.0,
            "plate_crop_path": "",
            "manual_plate_hint": plate_hint,
            "first_frame": frame_index,
            "last_frame": frame_index,
            "crop_paths": [],
            "best_crop_path": crop_path,
            "best_confidence": 1.0,
            "supplemental_crop": True,
        })
        record["first_frame"] = min(record["first_frame"], frame_index)
        record["last_frame"] = max(record["last_frame"], frame_index)
        record["crop_paths"].append(crop_path)
        if not record.get("best_crop_path"):
            record["best_crop_path"] = crop_path
        print(f"补充截图已保存：frame={frame_index}, track_id={track_id}, path={crop_path}", flush=True)


def render_recent_states(frame, tracker, frame_index, detect_interval, skip_track_ids=None):
    skip_track_ids = skip_track_ids or set()
    for state in tracker.track_states.values():
        if state.track_id in skip_track_ids:
            continue
        if not state.bboxes:
            continue
        if frame_index-state.last_frame>detect_interval:
            continue
        stationary = tracker.is_stationary(state.track_id)
        color = (0, 165, 255) if stationary else (255, 0, 0)
        status_label = "STATIC" if stationary else "MOVING"
        label = f"ID:{state.track_id} {status_label}"
        draw_label(frame, state.bboxes[-1], label, color)


def run_demo(config_path: str = "configs/demo_config.yaml") -> None:
    config = load_config(config_path)
    io_config = config.get("io", {})
    runtime_config = config.get("runtime", {})
    segments = config.get("filter", {}).get("flight_segments", [])

    video_path = io_config["video_path"]
    out_video_path = io_config.get("output_video", "outputs/demo/demo_annotated.mp4")
    output_dir = io_config.get("output_dir", "outputs/demo")
    raw_crops_dir = io_config.get("raw_crops_dir", os.path.join(output_dir, "raw_crops"))
    vehicles_dir = io_config.get("vehicles_dir", os.path.join(output_dir, "vehicles"))
    summary_csv = io_config.get("summary_csv", os.path.join(output_dir, "summary.csv"))
    manifest_json = io_config.get("manifest_json", os.path.join(output_dir, "manifest.json"))
    ocr_debug_csv = io_config.get("ocr_debug_csv", os.path.join(output_dir, "ocr_debug.csv"))
    appearance_review_csv = io_config.get("appearance_review_csv", os.path.join(output_dir, "appearance_review.csv"))

    ensure_video_available(video_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_crops_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

    detector = SAHIYOLOv11Detector(config=config)
    tracker = AeroTracker(config=config)
    plate_ocr = PlateOCR(config=config)
    supplemental_crop_schedule = build_supplemental_crop_schedule(config)
    supplemental_records = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"视频文件{video_path}打开失败。")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = runtime_config.get("start_frame", 0)
    end_frame = min(runtime_config.get("end_frame", total_frames-1), total_frames-1)
    detect_interval = max(1, runtime_config.get("detect_interval", 10))
    output_scale = float(runtime_config.get("output_scale", 1.0))
    output_width = int(width*output_scale)
    output_height = int(height*output_scale)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (output_width, output_height))
    frame_index = start_frame
    current_segment_name = None
    diagnostics = {}

    while cap.isOpened() and frame_index<=end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frame = frame.copy()
        maybe_save_supplemental_crops(raw_frame, frame_index, supplemental_crop_schedule, raw_crops_dir, supplemental_records)

        print(f"[第{frame_index}帧/共{end_frame}帧]：")
        segment = get_segment_for_frame(segments, frame_index)
        action = "suspend" if segment is None else segment.get("action", "track")

        if action=="suspend":
            cv2.putText(frame, "U-TURN: TRACKING SUSPENDED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 5)
            write_output_frame(out, frame, output_scale)
            frame_index += 1
            continue

        diagnostics.setdefault(segment["name"], {
            "frames": 0,
            "detect_frames": 0,
            "detections_after_filter": 0,
            "tracks": 0,
            "static_tracks": 0,
            "saved_crops": 0,
        })
        diagnostics[segment["name"]]["frames"] += 1
        current_roi = interpolate_roi(segment, frame_index)
        if current_roi is None:
            raise ValueError(f"第{frame_index}帧未找到有效ROI配置。")

        if segment["name"]!=current_segment_name:
            current_segment_name = segment["name"]
            tracker.reset_segment(keep_track_states=True)
            tracker.update_base_roi(current_roi.tolist())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker.start_frame(frame_rgb)
        should_detect = (frame_index-start_frame)%detect_interval==0

        if should_detect:
            raw_result = detector.predict(frame_rgb)
            dets_np = detector.collect_detections(raw_result, frame.shape, current_roi)
            diagnostics[segment["name"]]["detect_frames"] += 1
            diagnostics[segment["name"]]["detections_after_filter"] += int(len(dets_np))
        else:
            dets_np = np.empty((0, 6), dtype=np.float32)

        tracks = tracker.update_tracks(dets_np, frame_rgb)
        diagnostics[segment["name"]]["tracks"] += int(len(tracks))
        cv2.polylines(frame, [current_roi], isClosed=True, color=(0, 255, 0), thickness=4)

        rendered_track_ids = set()
        for track in tracks:
            bbox, track_id, confidence, cls_id = parse_track(track)
            rendered_track_ids.add(track_id)
            state = tracker.update_track_state(track_id, bbox, confidence, frame_index)
            stationary = tracker.is_stationary(track_id)
            color = (0, 165, 255) if stationary else (255, 0, 0)
            status_label = "STATIC" if stationary else "MOVING"
            label = f"ID:{track_id} {status_label} {confidence:.2f}"
            draw_label(frame, bbox, label, color)

            if stationary:
                before_count = len(state.crop_paths)
                maybe_save_static_crop(raw_frame, bbox, track_id, frame_index, state, runtime_config, raw_crops_dir)
                diagnostics[segment["name"]]["static_tracks"] += 1
                diagnostics[segment["name"]]["saved_crops"] += max(0, len(state.crop_paths)-before_count)

        render_recent_states(frame, tracker, frame_index, detect_interval, rendered_track_ids)

        tracker.finish_frame()
        write_output_frame(out, frame, output_scale)
        frame_index += 1

    cap.release()
    out.release()

    records = tracker.get_confirmed_records()
    if supplemental_records:
        records.extend(supplemental_records.values())
        print(f"补充截图记录已加入后处理：共{len(supplemental_records)}条。", flush=True)
    ocr_records = [record for record in records if record.get("crop_paths")]
    print(f"视频帧处理结束，进入OCR后处理：共{len(records)}条确认track，其中{len(ocr_records)}条包含车辆截图。", flush=True)
    print(
        "OCR模式："
        f"{'子进程隔离' if config.get('ocr', {}).get('isolate_paddle_process', True) else '主进程复用'}，"
        f"每条track最多{config.get('ocr', {}).get('max_crops_per_track', 8)}张crop参与投票。",
        flush=True,
    )

    for index, record in enumerate(ocr_records, start=1):
        if not record.get("crop_paths"):
            continue
        print(
            f"OCR进度：{index}/{len(ocr_records)}，"
            f"track_ids={record.get('track_ids', [])}，crop_count={len(record.get('crop_paths', []))}",
            flush=True,
        )
        ocr_result = plate_ocr.recognize_many(record["crop_paths"], raw_crops_dir, f"track_{record['track_ids'][0]:04d}")
        manual_plate_hint = record.get("manual_plate_hint", "")
        if manual_plate_hint and ocr_result.get("plate_text", "UNKNOWN")=="UNKNOWN":
            ocr_result["plate_text"] = manual_plate_hint
            ocr_result["plate_confidence"] = max(float(ocr_result.get("plate_confidence", 0.0)), 0.99)
        record.update(ocr_result)

    write_ocr_debug_csv(plate_ocr.consume_debug_rows(), ocr_debug_csv)
    print(f"OCR调试日志：{ocr_debug_csv}", flush=True)
    print("OCR后处理结束，开始ReID/聚类。", flush=True)
    reid = VehicleReID(config=config)
    clustered_records = reid.cluster(records)
    print("ReID/聚类结束，开始车牌归并与结果导出。", flush=True)
    consolidated_records = consolidate_vehicle_records(clustered_records, config)
    exported_records = export_clustered_records(consolidated_records, vehicles_dir)
    print(f"结果导出结束：共{len(exported_records)}个车辆目录写入摘要。", flush=True)
    appearance_review_rows = reid.build_appearance_review(exported_records)
    write_appearance_review_csv(appearance_review_rows, appearance_review_csv)
    print(f"外观一致性审查：{appearance_review_csv}", flush=True)
    write_summary_csv(exported_records, summary_csv)
    config["_diagnostics"] = diagnostics
    write_manifest(exported_records, manifest_json, config)

    print("完整Demo处理结束！")
    print(f"标注视频：{out_video_path}")
    print(f"车辆归档目录：{vehicles_dir}")
    print(f"结果摘要：{summary_csv}")
    print(f"清单文件：{manifest_json}")


if __name__ == "__main__":
    run_demo()
