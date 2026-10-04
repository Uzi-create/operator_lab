# 画面运动测试界面验证记录

2026-10-04，Windows 本机验证，环境为 Python 3.11.4、NumPy 2.2.6、OpenCV 4.12.0。测试入口为 `live_motion.py`，算法沿用 `estimate_scene_motion`，本次新增会话管理、界面和录制，没有修改生产算子的判定条件。

## 自动验证

- 全库 329 项 Python 测试通过。此前同批修改已通过 C++ 原生断言、27 组演示及 177 张 PNG 解码；最后补全录制计时字段后，相关 29 项界面/录制测试再次通过。
- 新模块共 42 项专项测试，覆盖真实合成平移、ROI 原始像素单位、拒绝断轨、复用采集缓冲区、时间戳异常、尺寸改变、暂停恢复、鼠标坐标映射和箭头裁剪。
- 视频测试实际调用 MJPG 编码并解码，包括中文路径、奇数尺寸补边、输入尺寸变化分片、录像回放、提前 EOF、编码器静默失败、源打开/读取失败及报告写入重试。
- 实际创建 OpenCV 窗口运行 12 帧，每帧窗口可见状态均为 1；另检查接受、拒绝和恢复后的界面输出。

## 120 帧合成录制

默认 960 × 540 输入，最长分析边上限 480，正反向每帧 `(4,-3)` / `(-4,3)` 像素平移。索引 48～52 为均匀空白；第 53 帧刚恢复纹理，与前一空白帧仍无对应关系。

| 项目 | 结果 |
| --- | --- |
| 建立基线 | 1 帧 |
| 接受 / 拒绝 | 113 / 6 帧 |
| 连续有效轨迹段 | 2 段 |
| 接受帧真值误差中位 / 最大 | 0.347 / 0.363 px |
| 原始 / 叠加 AVI 实际解码 | 各 120 帧 |
| CSV 与 JSONL 一致性 | 120 行逐行核对通过 |
| 保存的原始 AVI 重新输入 | 120 帧处理完成 |

该次合成运行的 API 中位耗时约 21.48 ms；读取、会话处理和绘制合计约 30.85 ms；加两路视频编码约 45.92 ms。最后一项不含 JSON/CSV 刷盘和 GUI 等待，不能直接宣称真实相机帧率。运行中包含拒绝帧，没有单独稳定预热，不属于跨硬件性能对照。

录制模块的 `source_timeline_frame_rate_hz` 表示输入时间轴频率；合成/视频输入的时间轴为名义帧率。真实相机的时间戳记录读帧完成时间，不是传感器曝光时间。接受率不等于准确率，累计像素位移也不是三维机器人轨迹。

## 复现

```text
python -B -m unittest projects.tests.test_motion_session projects.tests.test_live_motion projects.tests.test_motion_recording -v
python -B projects/camera/live_motion.py --demo --headless --frames 120
python -B projects/camera/live_motion.py --demo
python -B verify.py --native --demos
```

在其他电脑初次构建时省略 `--native`。新界面尚未在真实 Windows 摄像头、macOS/Linux GUI 或机器人上完成现场验证。录制的视频与本机日志保存在 `projects/output/`，不随源码仓库和便携包发布。
