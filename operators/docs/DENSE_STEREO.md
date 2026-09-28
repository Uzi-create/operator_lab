# 已校正双目稠密深度

`dense_stereo_depth(left_image, right_image, camera, baseline)` 对**已经极线校正且同步**的左右图调用 OpenCV C++ SGBM。左右图使用同一已校正内参，右相机位于左相机正 X 方向，视差定义为 `u_left-u_right>0`。`baseline` 与输出 `depth` 使用相同长度单位。灰度输入可为 uint8 或 `[0,1]` float，BGR 输入为 uint8；输出无效位置的视差/深度为 NaN，保留 `raw_disparity_px` 供诊断。

默认进行左右双向 SGBM 匹配与 1 px 一致性检查，并剔除无效/负视差及深度范围外点。双向检查需要第二次 C++ 匹配，也会在图像边缘和视差搜索范围不足处减少有效覆盖。`left_right_check=False` 可用于性能/覆盖对照；它不提供同等双向几何核验。低纹理、重复纹理、强反光、运动模糊和曝光不同会造成空洞或误匹配，左右一致性也不能识别所有错深度。`num_disparities` 是 16 的倍数，需覆盖真实视差范围；本算子不自动估计相机基线、校正映射或真实尺度。

已知 12 px 平移、0.12 m 基线、500 px 焦距的合成纹理中，无遮挡清晰区域输出视差 12 px、深度 5.0 m；遮挡区域大部分被拒绝。见 `projects/vision_robot/demo_dense_stereo.py` 和 `projects/benchmarks/benchmark_dense_stereo.py`。这些随机纹理测试不能代表真实双目相机精度；实际使用时需要独立标定、极线残差测量和量块/已知距离核验。
