"""
demo_viewer.py文件
    该文件提供Windows桌面可视化界面，用于离线展示outputs/demo中的最终Demo结果。
"""

import base64       as base64
import json         as json
import os           as os
import re           as re
import threading    as threading
import time         as time
import tkinter      as tk
from pathlib        import Path
from tkinter        import ttk, messagebox
from typing         import Any, Dict, List, Optional

import cv2          as cv2
import numpy        as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_ROOT/"outputs"/"demo"/"manifest.json"
VIDEO_SIZE = (620, 350)
MAIN_IMAGE_SIZE = (760, 430)
THUMBNAIL_SIZE = (160, 112)
THUMBNAIL_CARD_SIZE = (176, 144)


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT/path
    return path


def parse_frame_from_path(path_text: str) -> int:
    match = re.search(r"_frame_(\d+)", path_text)
    if not match:
        return -1
    return int(match.group(1))


def read_image_bgr(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size==0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_to_fit(image_bgr: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if width<=0 or height<=0:
        return image_bgr
    scale = min(max_width/width, max_height/height, 1.0)
    new_width = max(1, int(width*scale))
    new_height = max(1, int(height*scale))
    if new_width==width and new_height==height:
        return image_bgr
    return cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def fit_image_to_box(image_bgr: np.ndarray, box_width: int, box_height: int,
                     background_bgr: tuple = (15, 23, 42), zoom: float = 1.0,
                     allow_upscale: bool = True, pan_x: int = 0, pan_y: int = 0) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    canvas = np.full((box_height, box_width, 3), background_bgr, dtype=np.uint8)
    if width<=0 or height<=0:
        return canvas

    base_scale = min(box_width/width, box_height/height)
    if not allow_upscale:
        base_scale = min(base_scale, 1.0)
    scale = max(0.05, base_scale*max(0.1, zoom))
    new_width = max(1, int(width*scale))
    new_height = max(1, int(height*scale))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    x_offset = int((box_width-new_width)//2 + pan_x)
    y_offset = int((box_height-new_height)//2 + pan_y)
    dst_x1 = max(0, x_offset)
    dst_y1 = max(0, y_offset)
    src_x1 = max(0, -x_offset)
    src_y1 = max(0, -y_offset)
    copy_width = min(new_width-src_x1, box_width-dst_x1)
    copy_height = min(new_height-src_y1, box_height-dst_y1)
    if copy_width<=0 or copy_height<=0:
        return canvas
    canvas[dst_y1:dst_y1+copy_height, dst_x1:dst_x1+copy_width] = resized[
        src_y1:src_y1+copy_height,
        src_x1:src_x1+copy_width,
    ]
    return canvas


def image_to_photo(image_bgr: np.ndarray, box_width: int, box_height: int,
                   background_bgr: tuple = (15, 23, 42), zoom: float = 1.0,
                   allow_upscale: bool = True, pan_x: int = 0, pan_y: int = 0,
                   fast: bool = False) -> tk.PhotoImage:
    fitted = fit_image_to_box(image_bgr, box_width, box_height, background_bgr, zoom, allow_upscale, pan_x, pan_y)
    ext = ".ppm" if fast else ".png"
    success, encoded = cv2.imencode(ext, fitted)
    if not success:
        raise ValueError("图像编码失败。")
    if fast:
        return tk.PhotoImage(data=encoded.tobytes(), format="PPM")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=data, format="PNG")


def format_confidence(value: Any) -> str:
    try:
        return f"{float(value)*100:.2f}%"
    except Exception:
        return "-"


def format_video_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, remain_seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{remain_seconds:02d}"


class DemoViewerApp:
    def __init__(self, root: tk.Tk, manifest_path: Path = DEFAULT_MANIFEST) -> None:
        self.root = root
        self.manifest_path = manifest_path
        self.manifest = self.load_manifest()
        self.config = self.manifest.get("config", {})
        self.vehicles = self.prepare_vehicles(self.manifest.get("vehicles", []))
        self.selected_vehicle_index = 0
        self.selected_crop_index = 0
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_path = resolve_project_path(self.config.get("io", {}).get("output_video", "outputs/demo/demo_annotated.mp4"))
        self.video_total_frames = 0
        self.video_fps = 30.0
        self.video_total_seconds = 0.0
        self.video_current_frame = 0
        self.video_playing = False
        self.video_seek_job = None
        self.video_user_seeking = False
        self.video_seek_generation = 0
        self.video_seek_in_progress = False
        self.updating_video_scale = False
        self.video_play_start_time = 0.0
        self.video_play_start_frame = 0
        self.video_photo: Optional[tk.PhotoImage] = None
        self.main_photo: Optional[tk.PhotoImage] = None
        self.main_image_bgr: Optional[np.ndarray] = None
        self.main_image_zoom = 1.0
        self.main_image_pan_x = 0
        self.main_image_pan_y = 0
        self.main_drag_start_x = 0
        self.main_drag_start_y = 0
        self.main_drag_origin_x = 0
        self.main_drag_origin_y = 0
        self.thumbnail_photos: List[tk.PhotoImage] = []

        self.root.title("AeroTrack-ReID Demo Viewer")
        self.root.geometry("1460x900")
        self.root.minsize(1180, 760)
        self.root.configure(bg="#eef2f5")

        self.configure_styles()
        self.build_layout()
        self.load_video()
        self.render_summary()
        self.render_vehicle_list()
        self.select_vehicle(0)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"未找到Demo结果清单：{self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def prepare_vehicles(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        vehicles = []
        for record in records:
            crop_items = []
            for crop_path in record.get("crop_paths", []):
                frame = parse_frame_from_path(crop_path)
                crop_items.append({
                    "path": resolve_project_path(crop_path),
                    "name": Path(crop_path).name,
                    "frame": frame,
                    "phase": self.get_phase_name(frame),
                })
            crop_items.sort(key=lambda item: item["frame"])
            vehicles.append({
                "vehicle_id": record.get("vehicle_id", ""),
                "plate_text": record.get("plate_text", "UNKNOWN"),
                "plate_confidence": record.get("plate_confidence", 0.0),
                "first_frame": record.get("first_frame", 0),
                "last_frame": record.get("last_frame", 0),
                "track_ids": record.get("track_ids", []),
                "crop_count": len(crop_items),
                "best_crop_path": resolve_project_path(record.get("best_crop_path", "")) if record.get("best_crop_path") else None,
                "crop_items": crop_items,
            })
        return vehicles

    def get_phase_name(self, frame: int) -> str:
        if 0<=frame<=2070:
            return "去程"
        if 2071<=frame<=2550:
            return "调头"
        if frame>=2551:
            return "返程"
        return "补充"

    def configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f5")
        style.configure("Panel.TFrame", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("TLabel", background="#eef2f5", foreground="#17202a")
        style.configure("Panel.TLabel", background="#ffffff", foreground="#17202a")
        style.configure("Title.TLabel", font=("Microsoft YaHei", 18, "bold"), background="#102027", foreground="#ffffff")
        style.configure("Subtitle.TLabel", font=("Microsoft YaHei", 9), background="#102027", foreground="#cbd5e1")
        style.configure("MetricValue.TLabel", font=("Microsoft YaHei", 22, "bold"), background="#ffffff", foreground="#0b5f59")
        style.configure("MetricLabel.TLabel", font=("Microsoft YaHei", 9), background="#ffffff", foreground="#64748b")
        style.configure("Section.TLabel", font=("Microsoft YaHei", 12, "bold"), background="#ffffff", foreground="#17202a")
        style.configure("TButton", font=("Microsoft YaHei", 9), padding=(10, 6))
        style.configure("Accent.TButton", font=("Microsoft YaHei", 10, "bold"), foreground="#ffffff", background="#0f766e")
        style.map("Accent.TButton", background=[("active", "#0b5f59")])
        style.configure("Treeview", font=("Microsoft YaHei", 10), rowheight=28)
        style.configure("Treeview.Heading", font=("Microsoft YaHei", 10, "bold"))

    def build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#102027", height=72)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        ttk.Label(header, text="AeroTrack-ReID Demo 展示台", style="Title.TLabel").pack(anchor=tk.W, padx=24, pady=(12, 0))
        ttk.Label(
            header,
            text="无人机静止车辆识别 · 车牌OCR · 多视角归档结果离线展示",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, padx=24, pady=(4, 0))

        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=14)
        main.columnconfigure(0, weight=5, minsize=660)
        main.columnconfigure(1, weight=7, minsize=660)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=0)
        left.rowconfigure(1, weight=0)
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        self.build_video_panel(left)
        self.build_metric_panel(left)
        self.build_vehicle_panel(left)
        self.build_detail_panel(main)

    def build_video_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="标注视频", style="Section.TLabel").grid(row=0, column=0, sticky="w", padx=12, pady=(10, 8))
        self.video_canvas = tk.Canvas(
            panel,
            bg="#111827",
            width=VIDEO_SIZE[0],
            height=VIDEO_SIZE[1],
            highlightthickness=0,
        )
        self.video_canvas.grid(row=1, column=0, padx=12)

        controls = ttk.Frame(panel, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=10)
        controls.columnconfigure(2, weight=1)
        self.play_button = ttk.Button(controls, text="播放", style="Accent.TButton", command=self.toggle_video)
        self.play_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="重播", command=self.restart_video).grid(row=0, column=1, padx=(0, 8))
        self.video_scale = ttk.Scale(controls, from_=0, to=1, orient=tk.HORIZONTAL, command=self.on_video_seek)
        self.video_scale.grid(row=0, column=2, sticky="ew")
        self.video_scale.bind("<ButtonPress-1>", self.on_video_seek_press)
        self.video_scale.bind("<ButtonRelease-1>", self.on_video_seek_release)
        self.video_frame_label = ttk.Label(controls, text="F0/0 · 00:00/00:00", style="Panel.TLabel", width=24)
        self.video_frame_label.grid(row=0, column=3, padx=(8, 0))

    def build_metric_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame")
        panel.grid(row=1, column=0, sticky="ew", pady=10)
        for column in range(4):
            panel.columnconfigure(column, weight=1)
        self.metric_labels = {}
        metrics = [
            ("vehicle_count", "车辆目录"),
            ("crop_count", "车辆截图"),
            ("frame_count", "处理帧数"),
            ("detect_interval", "检测间隔"),
        ]
        for column, (key, label) in enumerate(metrics):
            cell = ttk.Frame(panel, style="Panel.TFrame")
            cell.grid(row=0, column=column, sticky="ew", padx=8, pady=10)
            value_label = ttk.Label(cell, text="0", style="MetricValue.TLabel")
            value_label.pack(anchor=tk.CENTER)
            ttk.Label(cell, text=label, style="MetricLabel.TLabel").pack(anchor=tk.CENTER)
            self.metric_labels[key] = value_label

    def build_vehicle_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame")
        panel.grid(row=2, column=0, sticky="nsew")
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="车辆列表", style="Section.TLabel").grid(row=0, column=0, sticky="w", padx=12, pady=(10, 8))
        columns = ("plate", "crops", "frames")
        self.vehicle_tree = ttk.Treeview(panel, columns=columns, show="headings", selectmode="browse")
        self.vehicle_tree.heading("plate", text="车牌号")
        self.vehicle_tree.heading("crops", text="截图")
        self.vehicle_tree.heading("frames", text="帧范围")
        self.vehicle_tree.column("plate", width=130, anchor=tk.CENTER)
        self.vehicle_tree.column("crops", width=70, anchor=tk.CENTER)
        self.vehicle_tree.column("frames", width=160, anchor=tk.CENTER)
        self.vehicle_tree.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.vehicle_tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 12))
        self.vehicle_tree.configure(yscrollcommand=scrollbar.set)
        self.vehicle_tree.bind("<<TreeviewSelect>>", self.on_vehicle_select)

    def build_detail_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame")
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=0)
        panel.rowconfigure(1, weight=0)
        panel.rowconfigure(2, weight=0)
        panel.rowconfigure(3, weight=1)

        top = ttk.Frame(panel, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 8))
        top.columnconfigure(0, weight=1)
        self.detail_title = ttk.Label(top, text="车辆详情", style="Section.TLabel")
        self.detail_title.grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="打开车辆目录", command=self.open_vehicle_folder).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(top, text="打开当前图片", command=self.open_current_image).grid(row=0, column=2, padx=(8, 0))

        preview_row = ttk.Frame(panel, style="Panel.TFrame")
        preview_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        preview_row.columnconfigure(0, weight=1)
        preview_row.columnconfigure(1, weight=0)
        self.main_image_label = tk.Label(preview_row, bg="#0f172a", width=68, height=26)
        self.main_image_label.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.main_image_label.bind("<MouseWheel>", self.on_main_image_mousewheel)
        self.main_image_label.bind("<Button-4>", self.on_main_image_mousewheel)
        self.main_image_label.bind("<Button-5>", self.on_main_image_mousewheel)
        self.main_image_label.bind("<ButtonPress-1>", self.on_main_image_drag_start)
        self.main_image_label.bind("<B1-Motion>", self.on_main_image_drag_move)

        info = ttk.Frame(preview_row, style="Panel.TFrame")
        info.grid(row=0, column=1, sticky="ns")
        self.info_labels = {}
        info_rows = [
            ("plate", "车牌号"),
            ("confidence", "OCR置信度"),
            ("tracks", "Track ID"),
            ("frames", "帧范围"),
            ("crops", "截图数量"),
            ("current", "当前图片"),
        ]
        for row, (key, label) in enumerate(info_rows):
            ttk.Label(info, text=label, style="MetricLabel.TLabel").grid(row=row*2, column=0, sticky="w", pady=(0, 2))
            value = ttk.Label(info, text="-", style="Panel.TLabel", wraplength=260, font=("Microsoft YaHei", 10, "bold"))
            value.grid(row=row*2+1, column=0, sticky="w", pady=(0, 12))
            self.info_labels[key] = value

        gallery_header = ttk.Frame(panel, style="Panel.TFrame")
        gallery_header.grid(row=2, column=0, sticky="ew", padx=14)
        gallery_header.columnconfigure(0, weight=1)
        ttk.Label(gallery_header, text="多视角截图", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.crop_filter_var = tk.StringVar()
        filter_entry = ttk.Entry(gallery_header, textvariable=self.crop_filter_var, width=24)
        filter_entry.grid(row=0, column=1, sticky="e")
        filter_entry.insert(0, "")
        self.crop_filter_var.trace_add("write", lambda *_: self.render_gallery())

        self.gallery_canvas = tk.Canvas(panel, bg="#ffffff", highlightthickness=0)
        self.gallery_canvas.grid(row=3, column=0, sticky="nsew", padx=14, pady=(8, 14))
        gallery_scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.gallery_canvas.yview)
        gallery_scrollbar.grid(row=3, column=1, sticky="ns", pady=(8, 14))
        self.gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)
        self.gallery_frame = ttk.Frame(self.gallery_canvas, style="Panel.TFrame")
        self.gallery_window = self.gallery_canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        self.gallery_frame.bind("<Configure>", self.on_gallery_configure)
        self.gallery_canvas.bind("<Configure>", self.on_gallery_canvas_configure)

    def load_video(self) -> None:
        if not self.video_path.exists():
            self.show_video_message(f"未找到视频：{self.video_path}")
            return
        self.video_capture = cv2.VideoCapture(str(self.video_path))
        if not self.video_capture.isOpened():
            self.show_video_message("视频打开失败")
            return
        self.video_total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_fps = float(self.video_capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.video_total_seconds = max(0.0, (self.video_total_frames-1)/max(self.video_fps, 1.0))
        self.configure_video_scale_range()
        self.show_video_frame(0)

    def show_video_message(self, message: str) -> None:
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            VIDEO_SIZE[0]//2,
            VIDEO_SIZE[1]//2,
            text=message,
            fill="#ffffff",
            width=VIDEO_SIZE[0]-40,
            font=("Microsoft YaHei", 10),
        )

    def configure_video_scale_range(self) -> None:
        self.updating_video_scale = True
        self.video_scale.configure(from_=0, to=max(0.0, self.video_total_seconds))
        self.video_scale.set(self.frame_to_seconds(self.video_current_frame))
        self.updating_video_scale = False

    def frame_to_seconds(self, frame_index: int) -> float:
        return frame_index/max(self.video_fps, 1.0)

    def seconds_to_frame(self, seconds: float) -> int:
        return int(round(seconds*max(self.video_fps, 1.0)))

    def frame_to_scale_value(self, frame_index: int) -> float:
        return self.frame_to_seconds(frame_index)

    def update_video_status_label(self, frame_index: int) -> None:
        total_frame = max(0, self.video_total_frames-1)
        current_time = format_video_time(self.frame_to_seconds(frame_index))
        total_time = format_video_time(self.video_total_seconds)
        self.video_frame_label.configure(text=f"F{frame_index}/{total_frame} · {current_time}/{total_time}")

    def render_summary(self) -> None:
        total_crops = sum(vehicle["crop_count"] for vehicle in self.vehicles)
        runtime = self.config.get("runtime", {})
        self.metric_labels["vehicle_count"].configure(text=str(len(self.vehicles)))
        self.metric_labels["crop_count"].configure(text=str(total_crops))
        self.metric_labels["frame_count"].configure(text=str(runtime.get("end_frame", "-")))
        self.metric_labels["detect_interval"].configure(text=str(runtime.get("detect_interval", "-")))

    def render_vehicle_list(self) -> None:
        for item_id in self.vehicle_tree.get_children():
            self.vehicle_tree.delete(item_id)
        for index, vehicle in enumerate(self.vehicles):
            self.vehicle_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    vehicle["vehicle_id"],
                    vehicle["crop_count"],
                    f'{vehicle["first_frame"]}-{vehicle["last_frame"]}',
                ),
            )
        if self.vehicles:
            self.vehicle_tree.selection_set("0")

    def on_vehicle_select(self, event=None) -> None:
        selection = self.vehicle_tree.selection()
        if not selection:
            return
        self.select_vehicle(int(selection[0]))

    def select_vehicle(self, index: int) -> None:
        if not self.vehicles:
            return
        self.selected_vehicle_index = max(0, min(index, len(self.vehicles)-1))
        self.selected_crop_index = 0
        if str(self.selected_vehicle_index) not in self.vehicle_tree.selection():
            self.vehicle_tree.selection_set(str(self.selected_vehicle_index))
        self.render_vehicle_detail()

    def get_selected_vehicle(self) -> Optional[Dict[str, Any]]:
        if not self.vehicles:
            return None
        return self.vehicles[self.selected_vehicle_index]

    def get_filtered_crops(self, vehicle: Dict[str, Any]) -> List[Dict[str, Any]]:
        keyword = self.crop_filter_var.get().strip().lower()
        crops = vehicle["crop_items"]
        if not keyword:
            return crops
        return [
            crop for crop in crops
            if keyword in crop["name"].lower() or keyword in str(crop["frame"]) or keyword in crop["phase"]
        ]

    def render_vehicle_detail(self) -> None:
        vehicle = self.get_selected_vehicle()
        if vehicle is None:
            return
        crops = self.get_filtered_crops(vehicle)
        self.selected_crop_index = min(self.selected_crop_index, max(0, len(crops)-1))
        crop = crops[self.selected_crop_index] if crops else None

        self.detail_title.configure(text=f'{vehicle["vehicle_id"]} · 车辆详情')
        self.info_labels["plate"].configure(text=vehicle["plate_text"])
        self.info_labels["confidence"].configure(text=format_confidence(vehicle["plate_confidence"]))
        self.info_labels["tracks"].configure(text=" / ".join(str(item) for item in vehicle["track_ids"]))
        self.info_labels["frames"].configure(text=f'{vehicle["first_frame"]} - {vehicle["last_frame"]}')
        self.info_labels["crops"].configure(text=f'{vehicle["crop_count"]} 张')

        if crop is None:
            self.info_labels["current"].configure(text="-")
            self.main_image_bgr = None
            self.main_image_zoom = 1.0
            self.main_image_pan_x = 0
            self.main_image_pan_y = 0
            self.main_image_label.configure(image="", text="无截图", fg="#ffffff")
        else:
            self.info_labels["current"].configure(text=crop["name"])
            self.render_main_image(crop["path"])
        self.render_gallery()

    def render_main_image(self, image_path: Path) -> None:
        image = read_image_bgr(image_path)
        if image is None:
            self.main_image_bgr = None
            self.main_image_label.configure(image="", text=f"图片读取失败：{image_path.name}", fg="#ffffff")
            return
        self.main_image_bgr = image
        self.main_image_zoom = 1.0
        self.main_image_pan_x = 0
        self.main_image_pan_y = 0
        self.refresh_main_image()

    def refresh_main_image(self) -> None:
        if self.main_image_bgr is None:
            return
        self.main_photo = image_to_photo(
            self.main_image_bgr,
            MAIN_IMAGE_SIZE[0],
            MAIN_IMAGE_SIZE[1],
            zoom=self.main_image_zoom,
            allow_upscale=False,
            pan_x=self.main_image_pan_x,
            pan_y=self.main_image_pan_y,
        )
        self.main_image_label.configure(image=self.main_photo, text="")

    def on_main_image_mousewheel(self, event) -> None:
        if self.main_image_bgr is None:
            return
        if getattr(event, "num", None)==5 or getattr(event, "delta", 0)<0:
            factor = 0.9
        else:
            factor = 1.1
        self.main_image_zoom = max(0.5, min(4.0, self.main_image_zoom*factor))
        self.refresh_main_image()

    def on_main_image_drag_start(self, event) -> None:
        if self.main_image_bgr is None:
            return
        self.main_drag_start_x = event.x
        self.main_drag_start_y = event.y
        self.main_drag_origin_x = self.main_image_pan_x
        self.main_drag_origin_y = self.main_image_pan_y

    def on_main_image_drag_move(self, event) -> None:
        if self.main_image_bgr is None:
            return
        self.main_image_pan_x = self.main_drag_origin_x + event.x - self.main_drag_start_x
        self.main_image_pan_y = self.main_drag_origin_y + event.y - self.main_drag_start_y
        self.refresh_main_image()

    def render_gallery(self) -> None:
        for child in self.gallery_frame.winfo_children():
            child.destroy()
        self.thumbnail_photos = []

        vehicle = self.get_selected_vehicle()
        if vehicle is None:
            return
        crops = self.get_filtered_crops(vehicle)
        if not crops:
            ttk.Label(self.gallery_frame, text="没有匹配的截图。", style="Panel.TLabel").grid(row=0, column=0, padx=12, pady=12)
            return

        columns = 4
        for index, crop in enumerate(crops):
            row = index//columns
            column = index%columns
            frame = ttk.Frame(self.gallery_frame, style="Panel.TFrame")
            frame.grid(row=row, column=column, padx=6, pady=6, sticky="n")
            frame.configure(width=THUMBNAIL_CARD_SIZE[0], height=THUMBNAIL_CARD_SIZE[1])
            frame.grid_propagate(False)
            image = read_image_bgr(crop["path"])
            if image is None:
                thumb = tk.Canvas(
                    frame,
                    bg="#e2e8f0",
                    width=THUMBNAIL_SIZE[0],
                    height=THUMBNAIL_SIZE[1],
                    highlightthickness=0,
                )
                thumb.create_text(
                    THUMBNAIL_SIZE[0]//2,
                    THUMBNAIL_SIZE[1]//2,
                    text="读取失败",
                    fill="#64748b",
                    font=("Microsoft YaHei", 9),
                )
            else:
                photo = image_to_photo(image, THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1], background_bgr=(226, 232, 240))
                self.thumbnail_photos.append(photo)
                thumb = tk.Label(frame, image=photo, bg="#e2e8f0", cursor="hand2")
            thumb.pack(anchor=tk.N, pady=(0, 0))
            thumb.bind("<Button-1>", lambda event, item_index=index: self.select_crop(item_index))
            caption = ttk.Label(
                frame,
                text=f'F{crop["frame"]} · {crop["phase"]}',
                style="Panel.TLabel",
                font=("Microsoft YaHei", 8),
                anchor=tk.CENTER,
            )
            caption.pack(fill=tk.X, padx=4, pady=(4, 0))
            caption.bind("<Button-1>", lambda event, item_index=index: self.select_crop(item_index))

    def select_crop(self, filtered_index: int) -> None:
        self.selected_crop_index = filtered_index
        vehicle = self.get_selected_vehicle()
        if vehicle is None:
            return
        crops = self.get_filtered_crops(vehicle)
        if not crops:
            return
        crop = crops[self.selected_crop_index]
        self.info_labels["current"].configure(text=crop["name"])
        self.render_main_image(crop["path"])

    def show_video_frame(self, frame_index: int) -> None:
        if self.video_capture is None:
            return
        frame_index = max(0, min(frame_index, max(0, self.video_total_frames-1)))
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.video_capture.read()
        if not ret:
            return
        self.video_current_frame = frame_index
        self.render_video_frame(frame, frame_index)

    def render_video_frame(self, frame: np.ndarray, frame_index: int) -> None:
        self.video_current_frame = frame_index
        self.video_photo = image_to_photo(frame, VIDEO_SIZE[0], VIDEO_SIZE[1], fast=True)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(VIDEO_SIZE[0]//2, VIDEO_SIZE[1]//2, image=self.video_photo)
        self.update_video_status_label(frame_index)
        self.set_video_scale(frame_index)

    def play_next_frame(self) -> None:
        if not self.video_playing or self.video_capture is None:
            return
        elapsed = time.perf_counter()-self.video_play_start_time
        target_frame = self.video_play_start_frame + int(elapsed*max(self.video_fps, 1.0))
        next_frame = max(self.video_current_frame+1, target_frame)
        if next_frame>=self.video_total_frames:
            self.video_playing = False
            self.play_button.configure(text="播放")
            return
        skip_count = next_frame-self.video_current_frame-1
        for _ in range(max(0, skip_count)):
            if not self.video_capture.grab():
                self.video_playing = False
                self.play_button.configure(text="播放")
                return
        ret, frame = self.video_capture.read()
        if not ret:
            self.video_playing = False
            self.play_button.configure(text="播放")
            return
        self.render_video_frame(frame, next_frame)
        self.root.after(1, self.play_next_frame)

    def toggle_video(self) -> None:
        if self.video_capture is None or self.video_seek_in_progress:
            return
        self.video_playing = not self.video_playing
        self.play_button.configure(text="暂停" if self.video_playing else "播放")
        if self.video_playing:
            self.video_play_start_time = time.perf_counter()
            self.video_play_start_frame = self.video_current_frame
            self.play_next_frame()

    def restart_video(self) -> None:
        self.video_playing = False
        self.video_play_start_time = 0.0
        self.video_play_start_frame = 0
        self.play_button.configure(text="播放")
        self.show_video_frame(0)

    def on_video_seek(self, value: str) -> None:
        if self.video_playing or self.updating_video_scale:
            return
        try:
            frame_index = self.seconds_to_frame(float(value))
        except ValueError:
            return
        frame_index = max(0, min(frame_index, max(0, self.video_total_frames-1)))
        self.update_video_status_label(frame_index)

    def on_video_seek_press(self, event=None) -> None:
        self.video_user_seeking = True
        if self.video_seek_job is not None:
            self.root.after_cancel(self.video_seek_job)
            self.video_seek_job = None

    def on_video_seek_release(self, event=None) -> None:
        if self.video_playing:
            return
        if self.video_seek_job is not None:
            self.root.after_cancel(self.video_seek_job)
            self.video_seek_job = None
        self.video_user_seeking = False
        self.seek_video_frame(self.seconds_to_frame(float(self.video_scale.get())))

    def seek_video_frame(self, frame_index: int) -> None:
        self.video_seek_job = None
        frame_index = max(0, min(frame_index, max(0, self.video_total_frames-1)))
        if abs(frame_index-self.video_current_frame)<=1:
            self.update_video_status_label(frame_index)
            return
        self.video_seek_generation += 1
        generation = self.video_seek_generation
        self.video_seek_in_progress = True
        self.play_button.configure(state=tk.DISABLED)
        self.update_video_status_label(frame_index)
        self.video_frame_label.configure(text=f"{self.video_frame_label.cget('text')} · 定位中")
        threading.Thread(
            target=self.load_seek_frame_in_background,
            args=(frame_index, generation),
            daemon=True,
        ).start()

    def load_seek_frame_in_background(self, frame_index: int, generation: int) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            self.root.after(0, lambda: self.finish_seek_frame(generation, None, None, frame_index))
            return

        target_seconds = self.frame_to_seconds(frame_index)
        capture.set(cv2.CAP_PROP_POS_MSEC, target_seconds*1000.0)
        ret, frame = capture.read()
        actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or frame_index+1)-1
        if not ret:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = capture.read()
            actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or frame_index+1)-1
        if not ret:
            capture.release()
            self.root.after(0, lambda: self.finish_seek_frame(generation, None, None, frame_index))
            return

        self.root.after(0, lambda: self.finish_seek_frame(generation, capture, frame, actual_frame))

    def finish_seek_frame(self, generation: int, capture, frame, actual_frame: int) -> None:
        if generation!=self.video_seek_generation:
            if capture is not None:
                capture.release()
            return

        self.video_seek_in_progress = False
        self.play_button.configure(state=tk.NORMAL)
        if capture is None or frame is None:
            self.update_video_status_label(self.video_current_frame)
            messagebox.showwarning("提示", "视频定位失败，请换一个时间点重试。")
            return

        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = capture
        actual_frame = max(0, min(actual_frame, max(0, self.video_total_frames-1)))
        self.render_video_frame(frame, actual_frame)

    def set_video_scale(self, frame_index: int) -> None:
        self.updating_video_scale = True
        self.video_scale.set(self.frame_to_scale_value(frame_index))
        self.updating_video_scale = False

    def on_gallery_configure(self, event=None) -> None:
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))

    def on_gallery_canvas_configure(self, event) -> None:
        self.gallery_canvas.itemconfigure(self.gallery_window, width=event.width)

    def open_vehicle_folder(self) -> None:
        vehicle = self.get_selected_vehicle()
        if vehicle is None:
            return
        folder = PROJECT_ROOT/"outputs"/"demo"/"vehicles"/vehicle["vehicle_id"]
        if folder.exists():
            os.startfile(str(folder))
        else:
            messagebox.showwarning("提示", f"目录不存在：{folder}")

    def open_current_image(self) -> None:
        vehicle = self.get_selected_vehicle()
        if vehicle is None:
            return
        crops = self.get_filtered_crops(vehicle)
        if not crops:
            return
        image_path = crops[self.selected_crop_index]["path"]
        if image_path.exists():
            os.startfile(str(image_path))
        else:
            messagebox.showwarning("提示", f"图片不存在：{image_path}")

    def on_close(self) -> None:
        self.video_playing = False
        if self.video_capture is not None:
            self.video_capture.release()
        self.root.destroy()


def main() -> None:
    try:
        root = tk.Tk()
        DemoViewerApp(root)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("AeroTrack-ReID Demo Viewer", str(exc))
        raise


if __name__=="__main__":
    main()
