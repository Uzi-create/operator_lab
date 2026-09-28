# 机器人末端相机手眼标定

`calibrate_eye_in_hand(T_base_from_gripper, T_camera_from_target)` 从多次机器人末端到基座位姿、固定标定靶到相机位姿求 `T_gripper_from_camera`。列表逐帧对应，必须使用同一时刻的机器人状态和相机图像，标定靶在采集期间固定。相机为眼在手上；眼在手外的坐标链不能直接套这个接口。

接口调用 OpenCV `calibrateHandEye`，默认 Park 方法，另可选 `tsai`、`horaud`、`andreff`、`daniilidis`。先检查机器人旋转幅度与旋转轴多样性，排除单一旋转轴或只平移的退化采集。求解后，对每帧独立计算 `T_base_from_gripper @ T_gripper_from_camera @ T_camera_from_target`，输出固定靶在基座下的平均位姿、逐帧平移/旋转误差及 RMS；不一致时 `success=False`。默认平移 RMS 门限 `.005` **使用输入平移相同单位**：若机器人位姿以毫米输入，表示 0.005 mm，通常应按设备误差显式调整。默认旋转 RMS 门限为 0.3°。

这个一致性检查衡量采集数据和模型的内部一致性，不能代替独立外部测量。机器人 TCP、相机靶位姿、时间同步或靶板格距若整体有系统偏差，仍可得到低内部残差。合成真值示例 `python projects/vision_robot/demo_handeye.py`，性能与多随机种子基准 `python projects/benchmarks/benchmark_handeye.py`。尚无真实机器人或相机现场验证。

当少量标定板位姿来自失败的 PnP 帧，可使用 `calibrate_eye_in_hand_robust(...)`。它从 6 帧子集生成假设，要求每个候选内点把固定靶变换到相同的基座位姿，之后仅用内点重算，并返回所有原始帧的 `inliers`、逐帧残差和实际试验次数。默认最多 128 次子集尝试，随机种子固定以便复现；找到足够一致的内点后，按采样成功概率达到 `confidence=.999999` 可提前停止，`adaptive=False` 则固定尝试 128 次以便比较。这个概率估计取决于当前内点比例及独立采样假设，不是现场成功率保证。`max_translation_error` 与输入平移单位相同；默认 `.005`。这只覆盖少数离群位姿，不修复系统性的相机/机器人时钟误差，也不保证在多数观测都坏时能找到真值。
