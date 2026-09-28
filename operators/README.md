# 算子库

这里包含可复用实现，不依赖 `projects` 中的检测项目或示例。将整个 `operators` 文件夹放到其他 Python 工程的根目录，即可通过 `from operators...` 导入；在目标电脑运行 `python operators/build.py` 构建动态库。

| 文件 | 能力 |
|---|---|
| `core.py` | 阈值、均值、自适应、引导滤波、CLAHE、形态学、边缘与距离场 |
| `vision_ops.py` | 动态阈值、亮度校正、区域处理、模板匹配、卡尺、直线/圆拟合 |
| `measurement_ops.py` | 平移配准、条纹宽度、径向圆边测量 |
| `metrology_ops.py` | 批量卡尺直线、独立四边拟合的矩形精测；[说明](docs/METROLOGY.md) |
| `shape_ops.py` | 离散旋转/尺度轮廓匹配、全边缘验收；[说明](docs/SHAPE_MATCHING.md) |
| `registration_ops.py` | C++ 精确 KD 树最近邻、局部鲁棒 ICP；[说明](docs/REGISTRATION.md) |
| `icp_plane_ops.py` | 带目标法向的六自由度局部点到平面 ICP；[说明](docs/ICP_POINT_TO_PLANE.md) |
| `pose_ops.py` | 标定相机 PnP 位姿、最终残差及平面双解诊断；[说明](docs/POSE_ESTIMATION.md) |
| `calibration_ops.py` | 棋盘内角点检测、多视图平面相机标定及独立重投影诊断；[说明](docs/CALIBRATION.md) |
| `stereo_ops.py` | 畸变感知双目三角化、逐点视差/深度/重投影诊断；[说明](docs/STEREO.md) |
| `handeye_ops.py` | 机器人末端相机手眼标定、运动可观测性和固定靶一致性诊断；[说明](docs/HANDEYE.md) |
| `dense_stereo_ops.py` | 已校正双目图的 SGBM 视差、左右一致性和米制深度；[说明](docs/DENSE_STEREO.md) |
| `stereo_rectify_ops.py` | 标定双目图和对应点的缓存极线校正；[说明](docs/STEREO_RECTIFY.md) |
| `depth_components_ops.py` | 深度跳变连通区域，原生 C++ 与独立参考；[说明](docs/DEPTH_COMPONENTS.md) |
| `ray_plane_ops.py` | 标定像素射线、已知平面米制交点；[说明](docs/RAY_PLANE.md) |
| `ground_ops.py` | 重力/高度约束的地面候选和障碍高度；[说明](docs/GROUND_OBSTACLES.md) |
| `box_ops.py` | 重力对齐的三维定向包围盒、可见点极值；[说明](docs/BOXES.md) |
| `sphere_ops.py` | 带离群点的三维球心/半径鲁棒测量；[说明](docs/SPHERE.md) |
| `line3d_ops.py` | 带离群点的三维直线及已观测线段；[说明](docs/LINE3D.md) |
| `voxel_cluster_ops.py` | 无序点云体素图连通区域，C++ 与参考后端；[说明](docs/VOXEL_CLUSTERS.md) |
| `feature_ops.py` | 缓存 ORB 模板、旋转/尺度/透视平面定位 |
| `robot_ops.py` | 相机模型、反投影、刚体变换、体素、平面、配准与法线 |
| `perception_ops.py` | 点云深度图、高度网格；C++ 与 NumPy 后端 |
| `ridge_ops.py` | 可复用 ridge 响应；原先位于金属检测脚本 |
| `fast_stats.py` / `defect_ops.py` | 局部/环形统计、缺陷对比度与片段分组 |
| `competition.py` | 深度主簇估计和时序证据确认 |
| `robust_geometry.py` / `image_io.py` | 稳健几何拟合及支持中文路径的图像 IO |
| `*.hpp` / `operators.cpp` | C++ 内核，构建产物与 Python 包同目录 |
| `tests/` | 算子单元测试与 C++ 断言 |

在工作区根目录使用 `python build.py --native` 构建，`python verify.py` 验证。单独测试算子包可运行 `python -m unittest discover -s operators/tests -t . -v`。

基础接口使用 `array('f')`；多数视觉接口使用 `[0,1]` float32/float64 灰度图，点云遵循各接口明确给出的坐标系和单位。Python 模块不因目录整理自动改变数值约定。

算法说明见 [docs](docs/)，最新能力说明见 [VISION_ROBOT.md](docs/VISION_ROBOT.md)。历史说明中的命令请按根目录新指南执行。
