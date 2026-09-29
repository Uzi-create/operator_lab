# 项目与示例

所有命令从仓库根目录运行；输出默认写入 `projects/output/`。`python -B verify.py --demos` 会执行全部演示并检查图像输出。

| 用途 | 示例 |
|---|---|
| 基础图像处理 | `python projects/examples/demo.py` |
| USB 摄像头实时算子 | `python projects/camera/live_operators.py --camera 1`；[说明](camera/README.md) |
| 金属表面候选检测 | `python projects/metal/run_samples.py` |
| 视觉定位与测量 | `python projects/vision_robot/demo_vision_robot.py` |
| 相机标定 | `python projects/vision_robot/demo_calibration.py` |
| 双目文件到点云 | `python projects/vision_robot/demo_stereo_file_workflow.py` |
| 手眼标定 | `python projects/vision_robot/demo_handeye.py` |
| 点云配准 | `python projects/vision_robot/demo_icp_plane.py` |

其他演示位于 `projects/vision_robot/`，性能脚本位于 `projects/benchmarks/`。金属检测的照片和人工 ROI 已随仓库发布；候选结果没有缺陷真值，不能算作准确率。双目文件输入格式见 [说明](docs/STEREO_FILE_WORKFLOW.md)。
