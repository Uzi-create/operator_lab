# 标定双目极线校正

`StereoRectifier(left_camera, right_camera, T_right_from_left, distortion_left=..., distortion_right=...)` 根据已知内参、畸变和右相机相对左相机的刚体外参，调用 OpenCV `stereoRectify` 建立并缓存双向映射。仅支持两相机同图像大小、右相机主要位于左相机正 X 方向的水平双目。构造时返回经过校正的 `rectified_camera`、正基线、左右有效 ROI、重投影矩阵 `Q` 和校正后的相机相对位姿。

`rectify_images(left,right)` 用缓存映射处理同步的 uint8 灰度/BGR 图，返回可送入 `dense_stereo_depth` 的图像、共同内参及基线。`rectify_points(left_xy,right_xy)` 将原始对应点变换到校正图像坐标，同时返回对应点的纵向偏差、畸变反解的正向重投影误差及逐点 `valid` 掩码；无效点的坐标为 NaN。强畸变时使用按需迭代而非 OpenCV 默认的少次反解，避免画面边缘出现明显极线误差。有效坐标可送入 `triangulate_stereo`，使用 `T_right_from_left_rectified`。校正后有些原图内的点可能落到新图像范围之外，应另外检查输出坐标。校正前后焦距和主点可能变化，必须使用新的 `rectified_camera`，不能复用原始内参。

数据必须来自真实双目标定与同步采集；本算子不估计外参或时间偏移。缓存映射由 OpenCV C++ 创建，图像重映射也在 C++ 中执行。合成位姿和点真值示例 `python projects/vision_robot/demo_stereo_rectify.py`，性能 `python projects/benchmarks/benchmark_stereo_rectify.py`。需以实拍对应点检查极线残差，并以独立距离真值验证深度尺度。

实际文件入口 `projects/vision_robot/run_stereo_pair.py` 把校正、稠密双目和三维反投影接在一起；标定 JSON 格式见 `projects/docs/STEREO_FILE_WORKFLOW.md`。
