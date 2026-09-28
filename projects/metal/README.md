# 金属表面划痕与色斑候选

此项目把 NumPy/OpenCV 算子与 C++ 细线响应组合为候选检测流程。它输出划痕和色斑候选，不做 OK/NG 判定，也不能据颜色确认氧化成因。九张授权公开的实拍照片位于 `samples/`；对应人工 ROI 见 `sample_rois.json` 和 `new_sample_rois.json`。ROI 不是缺陷真值标注。

从仓库根目录运行：

```bash
python projects/metal/run_samples.py
python projects/metal/compare_versions.py
python projects/metal/demo_defect_ops.py
```

对自己的图像运行时，传入 `--images`、`--manifest` 和 `--output`。manifest 中每张图需要文件名、`[宽, 高]` 和金属区域的多边形；可以增加参数与参考区域。完整调用可查看 `run_samples.py`。照片拍摄条件与 ROI 变化时应重新设定 ROI，并用独立真值集测量误报和漏检。

九张照片没有 EXIF 标签。SHA-256 记录位于 `samples/`。本机测量和旧版本结果是历史记录，不代表跨平台性能或现场准确率。
