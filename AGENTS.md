# AeroTrack-ReID 项目说明

## 项目目标

本项目用于完成“无人机视频车辆识别任务”Demo：

- 识别辅路中静止停放车辆。
- 截取并识别静止车辆车牌。
- 将同一车辆不同视角图片归档到同一车牌目录。

原始视频为 4K 无人机视频，约 2 分 37 秒，约 4711 帧。无人机先沿辅路向前飞行，随后 180° 调头返程。

## 当前主要入口

- `run_demo.py`：最终 Demo 总入口。
- `demo_pipeline.py`：完整流水线实现。
- `configs/demo_config.yaml`：当前主要配置文件。
- `demo_viewer.py`：Windows桌面可视化展示窗口，读取 `outputs/demo/manifest.json` 与车辆截图，不重新执行模型。
- `roi_calibrator.py`：关键帧 ROI 手工标定工具。
- `manual_crop_calibrator.py`：检测器漏框时的原始帧补充截图框选工具；支持 GUI 两点框选，也支持 `--save-frame` 和 `--bbox` 在无图形界面的服务器上使用。

运行方式：

```powershell
python run_demo.py
```

现场展示窗口：

```powershell
python demo_viewer.py
```

## 当前流水线

1. 使用 SAHI + YOLO11 对 4K 视频做切片检测。
2. 使用关键帧 ROI 插值过滤辅路区域。
3. 使用 BoT-SORT 跟踪；若 BoT-SORT 在返程阶段无轨迹输出，则启用 ByteTrack 风格两阶段 IoU 兜底跟踪。
4. 使用相机运动补偿后的世界坐标位移做静止车辆判定。
5. 保存未画框原始帧中的车辆 crop，避免 `STATIC` 等标注文字污染 OCR。
6. 在车辆 crop 内先做车牌候选检测和图像预处理，再使用 PaddleOCR 对同一 track 的多张 crop 做车牌识别投票。
7. 使用后处理纠错、归并和过滤，将最终结果按车牌号输出到 `outputs/demo/vehicles/<车牌号>/`。

## 关键配置说明

`configs/demo_config.yaml` 中重点字段：

- `runtime.detect_interval`：检测间隔，当前为每 5 帧检测一次。
- `runtime.output_scale`：输出视频缩放比例，当前为 0.5，用于降低播放卡顿。
- `filter.flight_segments`：去程、调头、返程分段。
- `filter.flight_segments[].roi_keyframes`：各飞行段关键帧 ROI 多边形。
- `tracking.fallback_iou_tracker`：BoT-SORT 无轨迹时启用兜底跟踪。
- `ocr.max_crops_per_track`：每条 track 抽取多少张 crop 做 OCR 投票。
- `ocr.isolate_paddle_process`：当前默认 `true`，用子进程隔离 PaddleOCR 段错误，避免主流程被 C++ SIGSEGV 直接杀掉。
- `ocr.device`：PaddleOCR 设备，默认 `cpu`；若安装了匹配 CUDA 的 `paddlepaddle-gpu`，可测试 `gpu:0`。
- `ocr.ocr_version`：PaddleOCR 模型版本，当前默认 `PP-OCRv4`，用于避开已多次崩溃的 `PP-OCRv5_server_*` CPU 推理路径。
- `ocr.max_subprocess_failures`：OCR 子进程连续失败熔断阈值，超过后本次运行跳过后续 OCR。
- `ocr.plate_detector_enabled`：是否在车辆 crop 内启用车牌候选区域检测。
- `ocr.plate_preprocess_scale`：车牌候选图像进入 OCR 前的放大倍率。
- `ocr.plate_preprocess_modes`：OCR 前预处理版本，当前默认 `raw`、`clahe`，用于降低推理次数和段错误概率。
- `ocr.plate_candidate_min_center_y_ratio`、`ocr.plate_candidate_min_top_margin_ratio`：过滤车辆 crop 顶部边缘的车牌候选，减少相邻停放车辆车牌被误读为本车车牌。
- `ocr.min_ocr_crop_width`、`ocr.min_ocr_crop_height`、`ocr.max_ocr_crop_pixels`、`ocr.max_preprocess_long_side`：进入 PaddleOCR 前的尺寸硬过滤。
- `ocr.allow_partial_plate_text`：允许 `75168`、`F75168` 这类不完整 OCR 后缀进入后处理软匹配。
- `io.ocr_debug_csv`：保存 OCR 原始识别文本、归一化文本、置信度和候选图路径，默认 `outputs/demo/ocr_debug.csv`。
- `supplemental_crops.enabled` / `supplemental_crops.items`：检测器漏框但原始帧车牌清楚时的补充截图配置。每项包含 `frame`、`bbox`、`track_id`，可选 `plate_hint` 和 `padding_ratio`。
- `postprocess.soft_expected_plate_matching`：当前默认 `true`，把 OCR 近似结果软匹配到人工复核的 8 个有效车牌；不会删除额外合法车牌目录。
- `postprocess.merge_near_duplicate_plate_text`：合并同一车牌的单字符 OCR 小样本变体。
- `postprocess.assign_unobserved_crops_to_dominant_plate`：当前默认 `false`，避免冲突 track 把未 OCR 的整条截图混入错误车牌目录。
- `postprocess.observation_neighbor_crops`：冲突 track 中，每个明确 OCR 命中的 crop 可带入的相邻 crop 数量；当前为 1，并会避开两个不同车牌之间的歧义中间帧。
- `postprocess.manual_track_plate_overrides`：人工复核后的最终 Demo 补救表；支持 `"track_id": "车牌"`，也支持 `{plate, start_frame, end_frame}` 帧范围写法。当前用于将稳定漏 OCR 的 `track_10052` 归入 `湘CF75168`，并把 `track_10029` 中 4075-4255 帧的返程侧后方视角归入 `湘CA755L`。
- `postprocess.plate_aliases`：少量 OCR 误识别的显式纠错表。
- `postprocess.expected_plates`：人工确认车牌列表，仅用于可选交付修正模式。
- `postprocess.use_expected_plates`：默认 `false`，避免把人工答案作为通用算法默认逻辑。

## 当前已知车牌

人工复核目标车辆共 8 辆：

- `湘CAZ732`
- `湘C6DY29`
- `湘CC095C`
- `湘N1818D`
- `湘CF75168`
- `湘C726Z8`
- `湘CA755L`
- `湘CD935C`

注意：这些车牌默认不作为硬编码答案参与通用流程。若最终展示需要严格只输出 8 辆车，可在配置中设置：

```yaml
postprocess:
  use_expected_plates: true
```

## 已处理的重要问题

- 修复了 PaddleOCR 3.x 不支持 `show_log` 的初始化问题。
- 禁用了 PaddleOCR/PaddlePaddle 中可能导致错误的 oneDNN/PIR 执行路径。
- 对 PaddleOCR 增加了子进程隔离模式，用于规避 Paddle C++ `Segmentation fault` 直接杀死主进程的问题。
- 将 OCR 子进程改为批量处理单张车辆 crop 的多个候选图，减少重复初始化 PaddleOCR 的次数。
- 增加了近似车牌小样本合并和主车牌截图补全，减少单字符误识别导致的目录分裂。
- 收紧了冲突 track 的归档逻辑：同一 track 内出现多个车牌时，不再整条 track 补全，只归入 OCR 命中点附近的局部窗口，降低不同车辆混入同一目录的概率。
- 增加了有效车牌软匹配与顶部候选过滤，用于恢复 `湘CF75168` 这类清晰但 OCR 容易漏省字符的车牌，并减少相邻车辆车牌干扰。
- 增加了 OCR 调试 CSV 与人工复核 track 补救机制，解决 `track_10052` 明确为 `湘CF75168` 但 PaddleOCR 未形成有效文本时的最终归档问题。
- `demo_pipeline.py` 在视频处理结束后会输出 OCR、ReID、导出阶段进度，避免长时间无终端反馈。
- Python 3.10 环境仍复现 PaddleOCR 段错误，因此 Python 3.12 不是唯一原因；当前优先通过 PP-OCRv4、子进程隔离和输入尺寸过滤降低风险。
- 修复了 crop 中带有框线和 `STATIC` 文本导致 OCR 误识别的问题。
- 增加了车牌彩色候选检测、CLAHE 增强、锐化和二值化预处理，减少整车 crop 中无关文字对 OCR 的干扰。
- 修复了返程阶段 BoT-SORT 无轨迹输出导致无 crop 的问题。
- 修复了所有图片进入 `vehicle_0001` 的导出目录问题。
- 将输出目录从 `vehicle_000x` 改为优先使用车牌号。
- 对 `STATIC`、`DJIM4T`、纯数字等伪车牌做过滤。

## 上上次输出视频复核：`demo_annotated.mp4`

文件 `D:\ProjectFiles\ML\all_datasets\demo_annotated.mp4` 是上上次程序运行后的带标注输出视频，不是干净原始视频。

读取结果：

- 当前环境未安装 `ffprobe`；OpenCV 默认后端可通过 Windows `MSMF` 读取该 MP4。GStreamer 会提示缺少 Quicktime demuxer 插件，但会回退到 `MSMF`。
- 视频属性：1920x1080，约 29.970 FPS，4712 帧，约 157.224 秒，文件约 1073 MiB。
- 该输出分辨率符合 `runtime.output_scale: 0.5`，即原始 4K 画面缩放到 1080p 输出。
- 已抽样保存检查帧和预览图到 `outputs/video_review/`，包括 `demo_annotated_contact_sheet.jpg`，采样帧为 0、471、942、1649、2356、2827、3298、3769、4240、4710。

画面和标注观察：

- 输出视频已叠加绿色 ROI 多边形、蓝色/橙色检测框、track ID、`MOVING`/`STATIC` 文本，以及右下角 DJI M4T 时间、经纬度、高度水印。因此该视频只适合复核检测/跟踪可视化，不适合作为 OCR crop 输入；OCR 仍应使用未标注原始帧或未画框 crop。
- 天气为雨天，路面反光明显；画面中有机动车、三轮/电动车、行人和遮阳伞等干扰目标，辅路右侧路边停车车辆是主要目标。
- 去程前半段可以看到多辆已知目标车牌，例如 `湘C726Z8`、`湘CF75168`、`湘CA755L`、`湘CC095C`、`湘C6DY29`。
- 约 78.6 秒附近画面显示 `U-TURN TRACKING SUSPENDED`，调头阶段可视化中无检测框/跟踪框；这属于当前分段逻辑现象，不应直接当作检测失败。
- 约 94 秒后返程阶段恢复静止框输出，但 track ID 明显跳到 `100xx` 段，说明兜底跟踪/返程 track 仍可能与去程 track 断裂。
- 返程阶段部分车辆框存在过大、局部截断、重叠或只覆盖车辆局部的问题，尤其靠近画面边缘、车身被树/杆/其他车辆遮挡时更明显。
- 后半段仍能看到 `湘CAZ732`、`湘C6DY29`、`湘CC095C`、`湘CD935C` 等车牌，但有些车牌只在少数帧清晰，OCR 抽帧质量和 track 合并仍是优化重点。

## 当前仍需注意

- YOLO 对无人机俯视角车辆可能会检测到车辆局部，尤其返程阶段更明显。
- ByteTrack 风格兜底跟踪能提高召回，但可能产生碎片 track。
- 输出目录不会自动批量清空；复核结果时应优先看本轮 `summary.csv` / `manifest.json` 中列出的 `crop_paths`，避免把旧文件误认为本轮结果。
- 当前 OCR 抽样不只做等距抽帧，还会额外选取文件体积较大的清晰 crop，避免漏掉只在少数帧清晰的车牌。
- 若 OCR 识别到 DJI 水印、时间戳或经纬度信息，该车辆 crop 会被标记为 rejected，后处理不会把它随整条 track 归入车牌目录。
- 冲突 track 中即使某个弱车牌观察被主车牌压制，弱车牌所在 crop 仍会作为反例位置参与窗口过滤，防止相邻车辆混入主车牌目录。
- 2026-05-13 修复了 `reject_crop` 被最佳 OCR 文本覆盖后丢失的问题；`track_10000_frame_001970.jpg` 这类只读到 DJI 水印/时间戳的 crop 应在后处理时被排除。
- 2026-05-13 调整了车牌候选顺序：优先尝试车辆底部固定区域，再尝试颜色候选，用于恢复 `湘CF75168` 去程背面这类绿色车牌被车窗反光候选挤掉的问题。
- 2026-05-13 将 `manual_track_plate_overrides` 升级为可限制帧范围，避免人工补救时整条不确定 track 污染目标车牌目录。
- 2026-05-14 增加 `supplemental_crops` 和 `manual_crop_calibrator.py`。用于 `湘CA755L` 这类靠近 ROI 边界、检测框在关键帧缺失但原始帧车牌清楚的情况；补充 crop 使用未标注原始帧，并继续进入 OCR/归档流程。
- 结合 `demo_annotated.mp4` 复核结果，返程阶段 `100xx` track ID、局部框和边缘截断是后续优化时需要重点检查的问题。
- OCR 偶尔会把相似字符识别错误，例如：
  - `湘C0935C` 应纠正为 `湘CD935C`
  - `湘CCD95C` 应纠正为 `湘CC095C`
- 对这类少量稳定误识别，优先在 `postprocess.plate_aliases` 中显式配置，不建议写死在代码里。

## 文件操作约束

禁止批量删除文件或目录。

不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件。

如果需要批量删除文件，应停止操作并让用户手动删除。
