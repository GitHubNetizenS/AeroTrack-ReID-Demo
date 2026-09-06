"""
detector.py文件
　　该Python文件定义了目标检测相关的算法封装类。主要包含了结合YOLOv11与SAHI（切片辅助超推理）技术的检测器实现，用于处理无人机航拍的高分辨率（如4K）
图像，解决小目标漏检与显存溢出问题，并在预测后置阶段增加了基于多边形 ROI（感兴趣区域）的空间过滤和类别过滤。
"""

import os       as os
import cv2      as cv2
import time     as time
import numpy    as np
from typing             import Any, Dict, List, Optional

from src.roi import bbox_roi_overlap_ratio

try:
    from sahi               import AutoDetectionModel
    from sahi.predict       import get_sliced_prediction
    from sahi.prediction    import PredictionResult
except Exception:
    AutoDetectionModel = None
    get_sliced_prediction = None
    PredictionResult = Any

"""
SAHIYOLOv11Detector类
　　该类读取外部传入的配置字典，初始化YOLOv11大模型并集成SAHI框架，提供针对单张高分辨率图像的切片推理接口，并支持将推理结果可视化保存。
"""
class SAHIYOLOv11Detector:
    """
    __init__函数
    　　初始化检测器对象，加载预训练权重至指定计算设备，并解析过滤配置。
    Args:
        config(Dict[str, Any])  从YAML解析的配置字典，包含“model”与“sahi”的相关参数配置
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        print("src/detector.py/SAHIYOLOv11Detector/__init__：")
        print("检测器对象初始化开始！")

        self.config = config
        self.model_config = config.get("model", {})
        self.sahi_config = config.get("sahi", {})
        self.filter_config = config.get("filter", {})
        self.device = self.model_config.get("device", "cuda:0")
        self.model_path = self.model_config.get("weights_path", "models/weights/yolo11l.pt")
        self.target_classes = self.filter_config.get("target_classes", [2, 5, 7])
        self.roi_polygon = np.array(self.filter_config.get("roi_polygon", []), dtype=np.int32)
        self.min_confidence = self.filter_config.get("min_confidence", self.model_config.get("confidence_threshold", 0.3))
        self.min_area_ratio = self.filter_config.get("min_area_ratio", 0.00008)
        self.max_area_ratio = self.filter_config.get("max_area_ratio", 0.15)
        self.aspect_ratio_range = self.filter_config.get("aspect_ratio_range", [0.45, 4.5])
        self.min_roi_overlap = self.filter_config.get("min_roi_overlap", 0.35)

        if AutoDetectionModel is None:
            raise ImportError("未安装sahi或其依赖，无法初始化SAHIYOLOv11Detector。请先安装sahi与ultralytics。")

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type=self.model_config.get("type", "yolov8"),
            model_path=self.model_path,
            confidence_threshold=self.model_config.get("confidence_threshold", 0.3),
            device=self.device
        )

        print("检测器对象初始化完毕！")
        print('='*25)


    """
    predict函数
    　　接收一帧图像，执行基于SAHI机制的切片辅助推理，并记录耗时。
    Args:
        frame_rgb(np.ndarray)   形状为(H, W, 3)的RGB格式图像矩阵
    Returns:
        PredictionResult    包含检测框坐标、类别及置信度的SAHI预测结果对象（未过滤）
    """
    def predict(self, frame_rgb: np.ndarray) -> PredictionResult:
        print("src/detector.py/SAHIYOLOv11Detector/predict：")
        print("切片预测开始！")

        slice_h = self.sahi_config.get("slice_height", 1024)
        slice_w = self.sahi_config.get("slice_width", 1024)
        overlap_h = self.sahi_config.get("overlap_height_ratio", 0.2)
        overlap_w = self.sahi_config.get("overlap_width_ratio", 0.2)
        start_time = time.time()
        result = get_sliced_prediction(
            frame_rgb,
            self.detection_model,
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap_h,
            overlap_width_ratio=overlap_w
        )
        elapsed_time = time.time() - start_time

        print(f"切片预测结束！共耗时{elapsed_time:.3f}秒。")
        print('=' * 25)

        return result


    def _is_valid_detection(self, bbox: List[float], score: float, category_id: int,
                            frame_shape: tuple, roi_polygon: Optional[np.ndarray]) -> bool:
        if self.target_classes and category_id not in self.target_classes:
            return False
        if score<self.min_confidence:
            return False

        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        box_w = max(0.0, x2-x1)
        box_h = max(0.0, y2-y1)
        if box_w<=0 or box_h<=0:
            return False

        area_ratio = (box_w*box_h)/float(width*height)
        if area_ratio<self.min_area_ratio or area_ratio>self.max_area_ratio:
            return False

        aspect_ratio = box_w/box_h
        min_aspect, max_aspect = self.aspect_ratio_range
        if aspect_ratio<min_aspect or aspect_ratio>max_aspect:
            return False

        if roi_polygon is not None and len(roi_polygon)>=3:
            overlap = bbox_roi_overlap_ratio(bbox, roi_polygon, frame_shape)
            if overlap<self.min_roi_overlap:
                return False

        return True


    def collect_detections(self, result: PredictionResult, frame_shape: tuple,
                           roi_polygon: Optional[np.ndarray] = None) -> np.ndarray:
        dets_list = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            score = float(obj.score.value)
            category_id = int(obj.category.id)
            box = [float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)]
            if self._is_valid_detection(box, score, category_id, frame_shape, roi_polygon):
                dets_list.append([box[0], box[1], box[2], box[3], score, category_id])

        if not dets_list:
            return np.empty((0, 6), dtype=np.float32)
        return np.array(dets_list, dtype=np.float32)


    """
    filter_predictions函数
    　　根据指定的类别ID和ROI多边形坐标对检测框进行物理过滤。
    Args:
        result(PredictionResult)    原始预测结果
    Returns:
        PredictionResult    经过过滤后的预测结果
    """
    def filter_predictions(self, result: PredictionResult) -> PredictionResult:
        print("src/detector.py/SAHIYOLOv11Detector/filter_predictions：")
        print("切片过滤开始！")
        filtered_objects = []
        result_image = getattr(result, "image", None)
        frame_shape = result_image.shape if result_image is not None else (2160, 3840, 3)

        for obj in result.object_prediction_list:
            # 操作1：类别过滤
            score = float(obj.score.value)
            category_id = int(obj.category.id)
            bbox = obj.bbox
            box = [float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)]
            if not self._is_valid_detection(box, score, category_id, frame_shape, self.roi_polygon):
                continue
            filtered_objects.append(obj)

        result.object_prediction_list = filtered_objects

        print(f"切片过滤结束！最终保留目标数量为{len(filtered_objects)}个。")
        print('=' * 25)

        return result


    """
    export_visuals 函数
    　　将包含检测结果（Bounding Boxes）的图像导出到本地硬盘。
    Args:
        result(PredictionResult)    调用predict()函数后返回的结果对象
        output_dir(str)             可视化图像输出的目录路径
        file_name(str)              导出的文件名称（不含扩展名）
    """
    @staticmethod
    def export_visuals(result: PredictionResult, output_dir: str, file_name: str) -> None:
        print("src/detector.py/SAHIYOLOv11Detector/export_visuals：")
        print("图像导出开始！")
        os.makedirs(output_dir, exist_ok=True)
        result.export_visuals(export_dir=output_dir, file_name=file_name)
        print(f"图像导出结束！结果已保存至{os.path.join(output_dir, file_name+'.png')}。")
        print('=' * 25)


    """
    export_visuals_with_roi 函数
    　　导出带有检测框和ROI多边形边界的图像，方便直观校验过滤区域，将包含检测结果（Bounding Boxes）的图像导出到本地硬盘。
    Args:
        result(PredictionResult)    调用predict()函数后返回的结果对象
        output_dir(str)             可视化图像输出的目录路径
        file_name(str)              导出的文件名称（不含扩展名）
        roi_polygon(np.ndarray)     用于绘制辅助线的ROI坐标组
    """
    @staticmethod
    def export_visuals_with_roi(result: PredictionResult, output_dir: str, file_name: str, roi_polygon: np.ndarray) -> None:
        print("src/detector.py/SAHIYOLOv11Detector/export_visuals_with_roi：")
        print("图像导出开始！")
        os.makedirs(output_dir, exist_ok=True)
        result.export_visuals(export_dir=output_dir, file_name=file_name)

        img_path = os.path.join(output_dir, f"{file_name}.png")
        if os.path.exists(img_path) and len(roi_polygon)>=3:
            img = cv2.imread(img_path)

            cv2.polylines(img, [roi_polygon.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=4)
            cv2.imwrite(img_path, img)
        print(f"图像导出结束！结果已保存至{os.path.join(output_dir, file_name+'.png')}。")
        print('=' * 25)
