"""
tracker.py文件
　　该Python文件封装无人机视频车辆跟踪、局部背景运动补偿与静止车辆判定。
"""

import argparse     as argparse
from dataclasses    import dataclass, field
from typing         import Any, Dict, List, Optional, Tuple

import cv2          as cv2
import numpy        as np
import torch        as torch

try:
    from ultralytics.trackers.bot_sort import BOTSORT
except Exception:
    BOTSORT = None


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    bboxes: List[List[float]] = field(default_factory=list)
    centers: List[Tuple[float, float]] = field(default_factory=list)
    world_centers: List[Tuple[float, float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    residual_votes: List[bool] = field(default_factory=list)
    residual_values: List[float] = field(default_factory=list)
    crop_paths: List[str] = field(default_factory=list)
    best_crop_path: str = ""
    best_confidence: float = 0.0
    plate_text: str = "UNKNOWN"
    plate_confidence: float = 0.0
    plate_crop_path: str = ""

    def is_stationary(self, required_votes: int, min_observations: int) -> bool:
        if len(self.residual_votes)<min_observations:
            return False
        return sum(self.residual_votes)>=required_votes


class AeroTracker:
    """
    AeroTracker类
    　　负责BoT-SORT多目标跟踪、局部背景光流补偿、静止投票和车辆历史状态管理。
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        print("src/tracker.py/AeroTracker/__init__：")
        print("跟踪器对象初始化开始！")

        self.config = config
        self.device = config.get("model", {}).get("device", "cuda:0")
        self.tracking_config = config.get("tracking", {})
        self.stationary_config = config.get("stationary", {})
        self.base_roi: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.curr_gray: Optional[np.ndarray] = None
        self.total_M = np.eye(3)
        self.track_states: Dict[int, TrackState] = {}
        self.fallback_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_fallback_id = self.tracking_config.get("fallback_start_id", 10000)
        self.fallback_iou_thresh = self.tracking_config.get("fallback_iou_thresh", 0.25)
        self.fallback_second_iou_thresh = self.tracking_config.get("fallback_second_iou_thresh", 0.12)
        self.fallback_max_age = self.tracking_config.get("fallback_max_age", 30)
        self.use_fallback_iou_tracker = self.tracking_config.get("fallback_iou_tracker", True)
        self.fallback_high_thresh = self.tracking_config.get("fallback_high_thresh", 0.45)
        self.fallback_low_thresh = self.tracking_config.get("fallback_low_thresh", 0.10)

        self.min_observations = self.stationary_config.get("min_observations", 12)
        self.vote_window = self.stationary_config.get("vote_window", 12)
        self.required_votes = self.stationary_config.get("required_votes", 8)
        self.max_residual_px = self.stationary_config.get("max_residual_px", 8.0)

        if BOTSORT is None:
            raise ImportError("未安装ultralytics，无法初始化BoT-SORT跟踪器。请先安装ultralytics。")

        tracker_args = {
            "tracker_type": self.tracking_config.get("tracker_type", "botsort"),
            "track_high_thresh": self.tracking_config.get("track_high_thresh", 0.35),
            "track_low_thresh": self.tracking_config.get("track_low_thresh", 0.1),
            "new_track_thresh": self.tracking_config.get("new_track_thresh", 0.35),
            "track_buffer": self.tracking_config.get("track_buffer", 90),
            "match_thresh": self.tracking_config.get("match_thresh", 0.8),
            "fuse_score": self.tracking_config.get("fuse_score", True),
            "gmc_method": self.tracking_config.get("gmc_method", "sparseOptFlow"),
            "proximity_thresh": self.tracking_config.get("proximity_thresh", 0.5),
            "appearance_thresh": self.tracking_config.get("appearance_thresh", 0.25),
            "with_reid": self.tracking_config.get("with_reid", False),
        }
        args = argparse.Namespace(**tracker_args)
        self.tracker_args = args
        self.frame_rate = self.tracking_config.get("frame_rate", 30)
        self.tracker = BOTSORT(args=args, frame_rate=self.frame_rate)

        print("跟踪器对象初始化完毕！")
        print('='*25)

    def reset_segment(self, keep_track_states: bool = True) -> None:
        print("src/tracker.py/AeroTracker/reset_segment：")
        print("重置当前飞行段跟踪器开始！")
        self.tracker = BOTSORT(args=self.tracker_args, frame_rate=self.frame_rate)
        self.prev_gray = None
        self.curr_gray = None
        self.total_M = np.eye(3)
        self.fallback_tracks.clear()
        if not keep_track_states:
            self.track_states.clear()
        print("重置当前飞行段跟踪器完毕！")
        print('='*25)

    def update_base_roi(self, new_roi_coords: list) -> None:
        print("src/tracker.py/AeroTracker/update_base_roi：")
        print("动态更新基准ROI开始！")
        self.base_roi = np.array(new_roi_coords, dtype=np.float32)
        self.prev_gray = None
        self.curr_gray = None
        self.total_M = np.eye(3)
        print("动态更新基准ROI完毕！")
        print('='*25)

    def start_frame(self, frame_rgb: np.ndarray) -> None:
        self.curr_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        self._update_global_motion()

    def finish_frame(self) -> None:
        if self.curr_gray is not None:
            self.prev_gray = self.curr_gray
        self.curr_gray = None

    def _update_global_motion(self) -> None:
        if self.prev_gray is None or self.curr_gray is None:
            return

        pts_prev = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=20)
        if pts_prev is None:
            return

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.curr_gray, pts_prev, None)
        if pts_curr is None or status is None:
            return

        idx = np.where(1==status)[0]
        if len(idx)<10:
            return

        m_affine, _ = cv2.estimateAffinePartial2D(pts_prev[idx], pts_curr[idx], method=cv2.RANSAC,
                                                  ransacReprojThreshold=3.0)
        if m_affine is None:
            return

        m_step = np.eye(3)
        m_step[:2, :] = m_affine
        self.total_M = m_step @ self.total_M

    def process_cmc_and_get_roi(self, curr_frame: np.ndarray) -> np.ndarray:
        print("src/tracker.py/AeroTracker/process_cmc_and_get_roi：")
        print("计算CMC与获取自适应ROI开始！")
        self.start_frame(curr_frame)
        if self.base_roi is None:
            raise ValueError("base_roi尚未初始化，请先调用update_base_roi。")
        print("计算CMC与获取自适应ROI完毕！")
        print('='*25)
        return self.base_roi.astype(np.int32)

    def estimate_local_background_delta(self, bbox: List[float]) -> np.ndarray:
        if self.prev_gray is None or self.curr_gray is None:
            return np.zeros(2, dtype=np.float32)

        height, width = self.curr_gray.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        bw, bh = max(1, x2-x1), max(1, y2-y1)
        pad_x, pad_y = int(bw*0.8), int(bh*0.8)
        rx1, ry1 = max(0, x1-pad_x), max(0, y1-pad_y)
        rx2, ry2 = min(width, x2+pad_x), min(height, y2+pad_y)
        if rx2-rx1<20 or ry2-ry1<20:
            return np.zeros(2, dtype=np.float32)

        mask = np.zeros((ry2-ry1, rx2-rx1), dtype=np.uint8)
        mask[:, :] = 255
        inner_x1 = max(0, x1-rx1)
        inner_y1 = max(0, y1-ry1)
        inner_x2 = min(mask.shape[1], x2-rx1)
        inner_y2 = min(mask.shape[0], y2-ry1)
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

        prev_crop = self.prev_gray[ry1:ry2, rx1:rx2]
        curr_crop = self.curr_gray[ry1:ry2, rx1:rx2]
        pts_prev = cv2.goodFeaturesToTrack(prev_crop, maxCorners=80, qualityLevel=0.01, minDistance=8, mask=mask)
        if pts_prev is None or len(pts_prev)<6:
            return np.zeros(2, dtype=np.float32)

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_crop, curr_crop, pts_prev, None)
        if pts_curr is None or status is None:
            return np.zeros(2, dtype=np.float32)

        valid = np.where(1==status)[0]
        if len(valid)<6:
            return np.zeros(2, dtype=np.float32)

        deltas = pts_curr[valid].reshape(-1, 2) - pts_prev[valid].reshape(-1, 2)
        return np.median(deltas, axis=0).astype(np.float32)

    def pixel_to_world(self, xy: Tuple[float, float]) -> np.ndarray:
        point = np.array([xy[0], xy[1], 1.0], dtype=np.float32)
        try:
            world_xy = (np.linalg.inv(self.total_M) @ point)[:2]
        except np.linalg.LinAlgError:
            world_xy = point[:2]
        return world_xy.astype(np.float32)

    def update_track_state(self, track_id: int, bbox: List[float], confidence: float, frame_index: int) -> TrackState:
        cx, cy = (bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0
        world_xy = self.pixel_to_world((cx, cy))
        state = self.track_states.get(track_id)
        if state is None:
            state = TrackState(track_id=track_id, first_frame=frame_index, last_frame=frame_index)
            self.track_states[track_id] = state

        if state.world_centers:
            last_world_xy = np.array(state.world_centers[-1], dtype=np.float32)
            residual = float(np.linalg.norm(world_xy-last_world_xy))
            state.residual_values.append(residual)
            state.residual_votes.append(residual<=self.max_residual_px)
            if len(state.residual_votes)>self.vote_window:
                state.residual_votes.pop(0)
                state.residual_values.pop(0)

        state.last_frame = frame_index
        state.bboxes.append([float(item) for item in bbox])
        state.centers.append((cx, cy))
        state.world_centers.append((float(world_xy[0]), float(world_xy[1])))
        state.confidences.append(float(confidence))
        if confidence>=state.best_confidence:
            state.best_confidence = float(confidence)
        return state

    def is_stationary(self, track_id: int, current_xy: Tuple[float, float] = (0.0, 0.0)) -> bool:
        state = self.track_states.get(track_id)
        if state is None:
            return False
        return state.is_stationary(self.required_votes, self.min_observations)

    def update_tracks(self, dets_np: np.ndarray, frame_rgb: np.ndarray) -> np.ndarray:
        print("src/tracker.py/AeroTracker/update_tracks：")
        print("跟踪器更新开始！")
        mock_boxes = MockBoxes(dets_np)
        tracks = self.tracker.update(mock_boxes, frame_rgb)

        if tracks is None or 0==len(tracks):
            if self.use_fallback_iou_tracker and dets_np is not None and len(dets_np)>0:
                fallback_tracks = self.update_fallback_tracks(dets_np)
                print("BoT-SORT未返回轨迹，已启用ByteTrack风格兜底跟踪器。")
                print("跟踪器更新完毕！")
                print('='*25)
                return fallback_tracks
            print("跟踪器更新完毕！")
            print('='*25)
            return np.empty((0, 7), dtype=np.float32)

        tracks_np = np.array(tracks, dtype=np.float32)
        print("跟踪器更新完毕！")
        print('='*25)
        return tracks_np

    @staticmethod
    def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2-x1)*max(0.0, y2-y1)
        area_a = max(0.0, box_a[2]-box_a[0])*max(0.0, box_a[3]-box_a[1])
        area_b = max(0.0, box_b[2]-box_b[0])*max(0.0, box_b[3]-box_b[1])
        union = area_a + area_b - inter
        if union<=0:
            return 0.0
        return float(inter/union)

    def update_fallback_tracks(self, dets_np: np.ndarray) -> np.ndarray:
        dets_np = dets_np[dets_np[:, 4]>=self.fallback_low_thresh]
        if len(dets_np)==0:
            self.age_unmatched_fallback_tracks(set())
            return np.empty((0, 7), dtype=np.float32)

        order = np.argsort(-dets_np[:, 4])
        dets_np = dets_np[order]
        high_dets = dets_np[dets_np[:, 4]>=self.fallback_high_thresh]
        low_dets = dets_np[dets_np[:, 4]<self.fallback_high_thresh]
        assigned_track_ids = set()
        output_tracks = []

        for det in high_dets:
            output_track = self.match_or_create_fallback_track(det, assigned_track_ids, self.fallback_iou_thresh)
            assigned_track_ids.add(int(output_track[4]))
            output_tracks.append(output_track)

        for det in low_dets:
            output_track = self.match_existing_fallback_track(det, assigned_track_ids, self.fallback_second_iou_thresh)
            if output_track is None:
                continue
            assigned_track_ids.add(int(output_track[4]))
            output_tracks.append(output_track)

        self.age_unmatched_fallback_tracks(assigned_track_ids)

        if not output_tracks:
            return np.empty((0, 7), dtype=np.float32)
        return np.array(output_tracks, dtype=np.float32)

    def match_or_create_fallback_track(self, det: np.ndarray, assigned_track_ids: set, iou_thresh: float) -> List[float]:
        matched = self.match_existing_fallback_track(det, assigned_track_ids, iou_thresh)
        if matched is not None:
            return matched

        bbox = det[:4].astype(np.float32)
        confidence = float(det[4])
        cls_id = float(det[5])
        track_id = self.next_fallback_id
        self.next_fallback_id += 1
        self.fallback_tracks[track_id] = {
            "bbox": bbox,
            "age": 0,
            "confidence": confidence,
            "cls_id": cls_id,
        }
        return [bbox[0], bbox[1], bbox[2], bbox[3], track_id, confidence, cls_id]

    def match_existing_fallback_track(self, det: np.ndarray, assigned_track_ids: set, iou_thresh: float) -> Optional[List[float]]:
        bbox = det[:4].astype(np.float32)
        confidence = float(det[4])
        cls_id = float(det[5])
        best_track_id = None
        best_iou = 0.0
        for track_id, track in self.fallback_tracks.items():
            if track_id in assigned_track_ids:
                continue
            iou = self.bbox_iou(bbox, track["bbox"])
            if iou>best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is None or best_iou<iou_thresh:
            return None

        self.fallback_tracks[best_track_id] = {
            "bbox": bbox,
            "age": 0,
            "confidence": confidence,
            "cls_id": cls_id,
        }
        return [bbox[0], bbox[1], bbox[2], bbox[3], best_track_id, confidence, cls_id]

    def age_unmatched_fallback_tracks(self, assigned_track_ids: set) -> None:
        for track_id in list(self.fallback_tracks.keys()):
            if track_id in assigned_track_ids:
                continue
            self.fallback_tracks[track_id]["age"] += 1
            if self.fallback_tracks[track_id]["age"]>self.fallback_max_age:
                del self.fallback_tracks[track_id]

    def get_confirmed_records(self) -> List[Dict[str, Any]]:
        records = []
        for state in self.track_states.values():
            if not state.crop_paths:
                continue
            records.append({
                "track_ids": [state.track_id],
                "plate_text": state.plate_text,
                "plate_confidence": state.plate_confidence,
                "plate_crop_path": state.plate_crop_path,
                "first_frame": state.first_frame,
                "last_frame": state.last_frame,
                "crop_paths": state.crop_paths,
                "best_crop_path": state.best_crop_path or state.crop_paths[0],
                "best_confidence": state.best_confidence,
            })
        return records


class MockBoxes:
    """
    MockBoxes类
    　　将SAHI产生的NumPy数组伪装成Ultralytics跟踪器期望的Boxes对象结构。
    """
    def __init__(self, dets_np: np.ndarray) -> None:
        if dets_np is None or 0==len(dets_np):
            self.xyxy = torch.empty((0, 4), dtype=torch.float32)
            self.conf = torch.empty((0,), dtype=torch.float32)
            self.cls = torch.empty((0,), dtype=torch.float32)
        else:
            self.xyxy = torch.as_tensor(dets_np[:, :4], dtype=torch.float32)
            self.conf = torch.as_tensor(dets_np[:, 4], dtype=torch.float32)
            self.cls = torch.as_tensor(dets_np[:, 5], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "MockBoxes":
        subset = MockBoxes.__new__(MockBoxes)
        subset.xyxy = self.xyxy[idx]
        subset.conf = self.conf[idx]
        subset.cls = self.cls[idx]
        return subset

    @property
    def xywh(self):
        if 0==len(self.xyxy):
            return torch.empty((0, 4), dtype=torch.float32)
        w = self.xyxy[:, 2] - self.xyxy[:, 0]
        h = self.xyxy[:, 3] - self.xyxy[:, 1]
        x_c = self.xyxy[:, 0] + w/2
        y_c = self.xyxy[:, 1] + h/2
        return torch.stack((x_c, y_c, w, h), dim=-1)
