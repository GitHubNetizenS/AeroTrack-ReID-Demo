"""
roi.py文件
　　该文件封装辅路ROI的关键帧插值与空间过滤工具函数。阶段3之后的视频处理均通过这些函数获得当前帧的辅路区域。
"""

from typing import Any, Dict, List, Optional

import cv2      as cv2
import numpy    as np


def _polygon_from_segment(segment: Dict[str, Any]) -> Optional[np.ndarray]:
    if "polygon" in segment:
        return np.array(segment["polygon"], dtype=np.float32)
    if "roi_polygon" in segment:
        return np.array(segment["roi_polygon"], dtype=np.float32)
    return None


def get_segment_for_frame(segments: List[Dict[str, Any]], frame_index: int) -> Optional[Dict[str, Any]]:
    for segment in segments:
        if segment.get("start_frame", 0)<=frame_index<=segment.get("end_frame", 0):
            return segment
    return None


def interpolate_roi(segment: Dict[str, Any], frame_index: int) -> Optional[np.ndarray]:
    keyframes = segment.get("roi_keyframes", [])
    if not keyframes:
        polygon = _polygon_from_segment(segment)
        if polygon is None:
            return None
        return polygon.astype(np.int32)

    sorted_keyframes = sorted(keyframes, key=lambda item: item["frame"])
    if frame_index<=sorted_keyframes[0]["frame"]:
        return np.array(sorted_keyframes[0]["polygon"], dtype=np.int32)
    if frame_index>=sorted_keyframes[-1]["frame"]:
        return np.array(sorted_keyframes[-1]["polygon"], dtype=np.int32)

    for left, right in zip(sorted_keyframes[:-1], sorted_keyframes[1:]):
        if left["frame"]<=frame_index<=right["frame"]:
            left_polygon = np.array(left["polygon"], dtype=np.float32)
            right_polygon = np.array(right["polygon"], dtype=np.float32)
            if left_polygon.shape!=right_polygon.shape:
                raise ValueError("ROI关键帧的多边形顶点数量必须一致！")
            span = max(right["frame"]-left["frame"], 1)
            alpha = (frame_index-left["frame"])/span
            polygon = left_polygon*(1-alpha) + right_polygon*alpha
            return polygon.astype(np.int32)

    polygon = _polygon_from_segment(segment)
    if polygon is None:
        return None
    return polygon.astype(np.int32)


def bbox_roi_overlap_ratio(bbox: List[float], roi_polygon: np.ndarray, frame_shape: tuple) -> float:
    if roi_polygon is None or len(roi_polygon)<3:
        return 1.0

    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(width-1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height-1, y1))
    y2 = max(0, min(height, y2))
    if x2<=x1 or y2<=y1:
        return 0.0

    local_polygon = roi_polygon.astype(np.int32).copy()
    local_polygon[:, 0] -= x1
    local_polygon[:, 1] -= y1
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)
    return float(np.count_nonzero(mask))/float(mask.size)


def bbox_center_inside_roi(bbox: List[float], roi_polygon: np.ndarray) -> bool:
    if roi_polygon is None or len(roi_polygon)<3:
        return True
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    return cv2.pointPolygonTest(roi_polygon.astype(np.int32), (cx, cy), False)>=0
