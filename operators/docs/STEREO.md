# 标定双目三角化

`triangulate_stereo(pixels_left, pixels_right, left_camera, right_camera, T_right_from_left, ...)` 接受逐行对应的原始像素和 OpenCV 标准畸变，输出左相机光学坐标系中的三维点。外参的平移单位就是输出坐标的单位；调用者必须通过标定提供可信的相机内外参和已匹配的像素。算子不做特征匹配、遮挡判断或机器人外参估计。

默认使用批量双射线最短距离中点，逐点检查像素边界、畸变反解、最小视差角、正深度、最大射线距离和左右相机的原始像素重投影误差。失败点保持原位置，三维坐标为 NaN，`valid` 与 `reasons` 给出原因。`method='dlt'` 可切换 OpenCV C++ 齐次 DLT，便于在特定相机几何下对照。两个方法都是带噪声条件下的几何估计，不是严格最大似然双视图非线性最优解。

合成数据中位误差与性能见 `projects/output/continuous_batch11/stereo_*`。基线源代码在 `projects/benchmarks/baselines/stereo_dlt_before_midpoint.py`。真实双目应先核对时钟同步、左右图像匹配与外参稳定性，并以独立测距目标验证尺度；低视差远距离点的深度误差会迅速放大。运行 `python projects/vision_robot/demo_stereo.py` 和 `python projects/benchmarks/benchmark_stereo.py`。
