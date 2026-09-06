"""
roi_calibrator.py文件
　　该工具用于在关键帧上点击辅路ROI顶点，并输出可复制到YAML配置中的polygon坐标。
"""

import argparse     as argparse
import os           as os

import cv2          as cv2

from baseline_sahi_inference import load_config


def read_frame(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"无法读取第{frame_index}帧。")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo_config.yaml")
    parser.add_argument("--frame", type=int, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = config["io"]["video_path"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件{video_path}不存在。")

    frame = read_frame(video_path, args.frame)
    points = []
    preview = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(preview, (x, y), 8, (0, 255, 0), -1)
            if len(points)>1:
                cv2.line(preview, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 3)
            cv2.imshow("ROI Calibrator", preview)

    cv2.namedWindow("ROI Calibrator", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI Calibrator", preview)
    cv2.setMouseCallback("ROI Calibrator", on_mouse)

    print("请按顺时针或逆时针点击ROI顶点。按Enter输出坐标，按Esc取消。")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key==13:
            break
        if key==27:
            points = []
            break

    cv2.destroyAllWindows()
    print(f"frame: {args.frame}")
    print(f"polygon: {points}")


if __name__ == "__main__":
    main()
