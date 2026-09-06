"""
baseline_sahi_inference.py文件
　　该Python文件是AeroTrack-ReID项目第1阶段的主执行入口。通过读取外部YAML配置文件，提取目标视频的第1帧，调用检测模块执行推理，并保存验证结果。
"""

import os       as os
import yaml     as yaml
import cv2      as cv2
import numpy    as np
from src.detector import SAHIYOLOv11Detector

"""
load_config 函数
　　读取并解析YAML格式的配置文件。
Args:
    config_path(str)    配置文件在系统中的绝对或相对路径
Returns:
    dict    解析后的Python字典
"""
def load_config(config_path: str) -> dict:

    print("baseline_sahi_inference.py/load_config：")
    print("读取配置文件开始！")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件{config_path}！")
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("读取配置文件结束！")
    print('=' * 25)

    return config


"""
extract_first_frame 函数
　　从给定路径的视频文件中提取第一帧图像，并转换为RGB色彩空间。
Args:
    video_path(str)     视频文件的系统路径
Returns:
    np.ndarray  RGB色彩空间的图像矩阵(H, W, 3)
"""
def extract_first_frame(video_path: str) -> np.ndarray:
    print("baseline_sahi_inference.py/extract_first_frame：")
    print("提取第1帧图像开始！")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件{video_path}不存在！")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2550)
    ret, frame = cap.read()

    cap.release()
    if not ret:
        raise ValueError("视频读取失败或视频为空！")
    # OpenCV默认使用BGR，深度学习及SAHI常规要求使用RGB。
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("提取第1帧图像结束！")
    print('=' * 25)

    return frame_rgb


if "__main__"==__name__:
    # 步骤1：动态加载配置
    CONFIG_PATH = "configs/phase1_config.yaml"
    config = load_config(CONFIG_PATH)
    io_config = config.get("io", {})
    video_path = io_config.get("video_path")
    out_dir = io_config.get("output_dir")
    out_name = io_config.get("output_filename")
    # 步骤2：准备数据并读取视频的截帧
    frame_rgb = extract_first_frame(video_path)
    # 步骤3：实例化检测器
    detector = SAHIYOLOv11Detector(config=config)
    # 步骤4：执行推理流水线
    prediction_result = detector.predict(frame_rgb)
    # 步骤5：生成结果
    detector.export_visuals(prediction_result, out_dir, out_name)