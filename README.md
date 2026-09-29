# Operator Lab / 机器视觉与机器人感知算子

可复用的 Python 接口与 C++17 内核，覆盖定位与模板匹配、几何测量、相机标定、双目深度、点云处理和机器人感知。`operators/` 是算子库；`projects/` 是示例、基准与金属表面检测应用。详细接口见 [算子索引](operators/README.md) 和 [项目索引](projects/README.md)。

## 快速开始

需要 Python 3.9+、NumPy、OpenCV 和 C++17 编译器。Windows 使用 MinGW-w64 `g++`，macOS 使用 Xcode Command Line Tools 的 `c++`，Linux 使用 `g++` 或 `c++`。

```bash
python -m pip install -r requirements.txt
python -B build.py
python -B verify.py --demos
```

`verify.py --demos` 构建动态库、运行 Python 测试与 C++ 断言，并执行 27 组演示及检查 PNG 输出。样本照片位于 `projects/metal/samples/`；若自行移除照片，验证会跳过 5 组实拍演示，仍运行算子和合成数据测试。输出写入 `projects/output/verified_current/`。跨电脑构建时不要使用 `--native`；该选项针对当前 CPU 生成代码。

```python
from operators.feature_ops import create_orb_template, locate_planar_template
from operators.measurement_ops import estimate_translation, measure_circle
from operators.perception_ops import pointcloud_to_depth, elevation_grid
```

可以从 [双目文件流程](projects/docs/STEREO_FILE_WORKFLOW.md) 接入自己的左右相机图像与标定 JSON；金属表面样例见 [说明](projects/metal/README.md)。

## 验证边界

2026-09-28 在 Windows 本机完成 248 项 Python 测试、C++ 断言、27 组演示和 177 张 PNG 解码。几何与标定精度主要来自已知真值的合成数据；金属照片有候选框与人工 ROI，没有缺陷真值。2026-09-29 在 macOS arm64（Python 3.12.13、NumPy 2.5.3、OpenCV 4.14.0）完成独立验证：248 项 Python 测试、C++ 断言与地址/未定义行为 sanitizer、27 组演示、177 张 PNG 解码。最近邻测试允许几个 float64 舍入步骤的误差，匹配索引仍要求完全一致。Linux、真实双目相机、机器人和生产现场尚未完成独立运行与精度验证。历史基准只适用于各自所记载的硬件、编译器和输入条件。

代码和仓库内样本以 [MIT 许可证](LICENSE) 发布。代码使用 OpenCV 和 NumPy，但不包含它们的源码。这里的接口不声称兼容 HALCON。
