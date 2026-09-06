"""
manual_crop_calibrator.py文件
    该工具用于在指定帧上手工框选车辆补充截图，并输出可复制到
    configs/demo_config.yaml 中 supplemental_crops.items 的bbox配置。
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
    parser.add_argument("--track-id", type=int, default=90003)
    parser.add_argument("--plate", default="")
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--padding-ratio", type=float, default=0.02)
    parser.add_argument("--save-frame", default="")
    parser.add_argument("--bbox", nargs=4, type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = config["io"]["video_path"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件{video_path}不存在。")

    frame = read_frame(video_path, args.frame)

    if args.save_frame:
        output_dir = os.path.dirname(args.save_frame)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(args.save_frame, frame)
        print(f"已保存原始帧：{args.save_frame}")

    if args.bbox:
        bbox = [
            min(args.bbox[0], args.bbox[2]),
            min(args.bbox[1], args.bbox[3]),
            max(args.bbox[0], args.bbox[2]),
            max(args.bbox[1], args.bbox[3]),
        ]
        print_config(args, bbox)
        return

    if args.save_frame:
        print("如需无界面输出配置，请再次运行并添加--bbox x1 y1 x2 y2。")
        return

    display_scale = max(0.05, min(float(args.display_scale), 1.0))
    display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    preview = display_frame.copy()
    points = []

    def to_original_xy(x, y):
        return [int(round(x/display_scale)), int(round(y/display_scale))]

    def redraw():
        nonlocal preview
        preview = display_frame.copy()
        for point in points:
            scaled_point = (int(round(point[0]*display_scale)), int(round(point[1]*display_scale)))
            cv2.circle(preview, scaled_point, 6, (0, 255, 0), -1)
        if len(points)==2:
            p1 = (int(round(points[0][0]*display_scale)), int(round(points[0][1]*display_scale)))
            p2 = (int(round(points[1][0]*display_scale)), int(round(points[1][1]*display_scale)))
            cv2.rectangle(preview, p1, p2, (0, 255, 0), 2)
        cv2.imshow("Manual Crop Calibrator", preview)

    def on_mouse(event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            if len(points)>=2:
                points.clear()
            points.append(to_original_xy(x, y))
            redraw()

    cv2.namedWindow("Manual Crop Calibrator", cv2.WINDOW_NORMAL)
    cv2.imshow("Manual Crop Calibrator", preview)
    cv2.setMouseCallback("Manual Crop Calibrator", on_mouse)

    print("请点击补充截图框的左上角和右下角。按Enter输出bbox，按R重选，按Esc取消。")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13):
            break
        if key in (ord("r"), ord("R")):
            points.clear()
            redraw()
        if key==27:
            points = []
            break

    cv2.destroyAllWindows()
    if len(points)!=2:
        print("未输出bbox。")
        return

    x1, y1 = points[0]
    x2, y2 = points[1]
    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    print_config(args, bbox)


def print_config(args, bbox) -> None:
    print("请复制下面内容到configs/demo_config.yaml的supplemental_crops.items中：")
    print(f"  - frame: {args.frame}")
    print(f"    track_id: {args.track_id}")
    if args.plate:
        print(f"    plate_hint: \"{args.plate}\"")
    print(f"    bbox: {bbox}")
    print(f"    padding_ratio: {args.padding_ratio}")


if __name__ == "__main__":
    main()
