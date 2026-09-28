# 平面棋盘相机标定

`detect_chessboard_corners(image, (内角点列数, 内角点行数))` 使用 OpenCV 的 `findChessboardCornersSB`，返回亚像素内角点和检测状态。棋盘必须完整可见；返回顺序是检测器确定的棋盘行列顺序，不包含印刷标记或物理朝向识别。方形、无标记棋盘旋转 180° 后可能得到另一种同样合法的坐标方向。

`calibrate_planar_camera(object_points, image_points, (宽, 高))` 使用同一块 z=0 棋盘的多视图对应点。`object_points` 是 N×3 米制棋盘点，`image_points` 是每张图 N×2 像素点列表，顺序必须逐点对应。默认至少 8 张图、法向跨度至少 8°、整体 RMS 不超过 1 px、逐视图 RMS 不超过 2 px。返回 `camera`、5 系数 `distortion`、每视图棋盘到相机位姿、重投影 RMS、视角跨度及参数标准差。`rational8` 可选，但需要覆盖更大像场和更多视角；否则额外参数可能严重耦合，虽然棋盘上的重投影残差很小，图像边缘的反畸变仍可能错误。返回的标准差可辅助判断稳定性。

实现调用 OpenCV C++ 标定核心，Python 层做输入校验和最终位姿的独立重投影检查。`success` 只表示输入视图通过门限，不代表真实相机的整个像场或机器人外参已经验证。真实标定应保留未参与拟合的不同位置、不同倾角图像，检查独立重投影误差，并确认打印棋盘的实际格距。混合棋盘方向、局部错序、运动模糊和角点被遮挡均可能破坏结果。示例：`python projects/vision_robot/demo_calibration.py`；性能与多随机种子检查：`python projects/benchmarks/benchmark_calibration.py`。
