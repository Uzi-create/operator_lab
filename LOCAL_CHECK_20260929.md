# 本机替换与验证记录

- source: https://github.com/Uzi-create/operator_lab
- commit: 259a8aa
- backup: /Users/cathypark/Documents/创建/operator_lab_backup_20260929_091705
- backup_files_verified: 368
- python_tests: 248
- demo_runs: 27
- pngs_decoded: 177
- sanitizers: passed
- local_change: float64 oracle tolerance: 2e-16 -> 8*eps; algorithms unchanged
- environment: /Users/cathypark/Documents/创建/operator_lab_env

运行完整验证：

```bash
cd /Users/cathypark/Documents/创建/operator_lab
../operator_lab_env/bin/python -B verify.py --sanitizers --demos
```

GitHub 未推送本地改动；mode 设计按用户选择保留。验证不代表真实设备或生产现场精度。

## USB 摄像头接入

- 设备：HK 1080P RGB，AVFoundation 索引 1。
- 真实帧：640×480，自适应算子连续 30 帧；八种显示效果各运行 15 帧。
- 预览：已启动实时左右对照窗口，OpenCV 报告窗口可见。
- 处理记录：projects/output/camera/all_operators_rgb.json（仅元数据与耗时，没有保存原始照片）。
- 新摄像头测试及完整单元回归：252 项通过。

## 扩展相机算子验证

- 新调用程序：projects/camera/verify_camera_operators.py。
- 索引 1 彩色、索引 0 灰度桌面画面各测试 38 个操作项目，每项 6 帧，实际尺寸 640×480；全部无执行错误。
- 每个通道均有 16 项真实帧数值对比通过，整数/掩码严格相等，浮点 atol=rtol=2e-5。
- 圆/矩形未检出可靠目标；直线部分帧未检出。模板为同帧自匹配；不代表真实测量、跟踪或识别精度验证。
- 完整回归：256 项测试、C++ 原生断言、地址和未定义行为检查通过；最终报告输出修改后追加调用程序和入口检查，6 项通过。
- 结果：projects/output/camera/full_check_camera1/RESULTS.md、gallery.png、report.json；灰度索引 0 的报告在 full_check/。
- 本次 --save-previews 明确启用效果图保存；原先实时脚本的 --report 仍仅保存元数据。

## 38 项实时浏览

- projects/camera/live_all_operators.py：点击列表或 N/P 切换，空格轮播，V 切换统计字段；图像/坐标/测量/模型均有对应显示。
- 9 项浏览器、验证脚本和入口测试通过；真实相机 --headless --cycle --frames 38 覆盖 38 个不同操作，无执行错误，记录 projects/output/camera/live_all_38.log。
- 真实 GUI 单独运行 10 帧后退出，窗口状态记录 projects/output/camera/live_all_window.log。

## 主窗口原始像素预览

- 默认采集 1920×1080、max-width=0，不再默认缩到 640。左侧使用完整采集图，右侧使用算子结果。
- image_view 原始像素视口：100% 精确裁切复制、拖动、缩放、F 完整画面、1 恢复 100%；窗口尺寸可用时自适应扩大可见区域。
- 9 项相关测试通过，包含与原始像素逐点比较、放大像素块、边界拖动和适应窗口比例检查。
- GUI 实测 Captured 与 processing 均 1920×1080，窗口可见；记录 native_preview_window.log。
- 1080P 实时处理路径轮流覆盖 38 项，记录 native_preview_38.log；示例画布 native_preview_example.png。

## 清晰度对照窗口

- projects/camera/focus_meter.py：原始中心 ROI、Laplacian 方差和梯度能量的滑动中位数、亮度与近饱和比例；B 记录指标与原图，C 清空当前滑动历史，Q 退出。
- 5 项指标与入口测试通过；相同印刷图案加模糊后两个指标均下降，恒定图指标为零，读取失败释放相机。
- 相机 GUI 读取 1920×1080、窗口可见，20 帧后自动退出，记录 focus_window.log。
- 指标不是焦距或绝对对焦判据，无法排除噪声、纹理尺度与光照影响。

## iPhone 摄像头

- 系统枚举名称：“iPhone”的相机，model identifier iPhone17,3；当前 OpenCV 排序索引 3。
- 实际 5 帧 1920×1080 自适应算子链路成功，记录 iphone_probe.log；GUI 窗口另行记录 iphone_window.log。
- 当前 Mac 接口没有报告可控对焦/自定义曝光能力；没有验证手机镜头切换、手动对焦或标定参数。

## iPhone 均值算子实验

- 新增 experiment_mean.py 与 EXPERIMENT_01.md，直接调用 mean/mean_naive；固定同一帧交替顺序，每组各预热一次、7 次计时。
- 1920×1080 实拍：半径 1/3/7/15 的正常场景复测中位加速比分别为 1.45/5.85/34.38/203.10；所有采样整幅图最大误差与 RMSE 为 0。
- 6 项实验与入口测试通过：结果对比、输入不变、差异故障注入、报告/图片保存、近黑帧拒绝、读取失败释放相机以及入口清单检查。日志 experiment_mean_tests.log。
- GUI 12 帧实际运行成功，captured=(1080,1920,3)、visible=1.0；日志 experiment_mean_window.log。
- 此处是 API 耗时，包括检查、分配和 C++ 调用，不含采集、转换、验证与显示。没有改动原生算法，本次验证不是重新跑全套回归。

- 首次实验输出近黑帧，已判定不作有效场景报告；加入近黑帧拒绝检查。重新采集的有效报告与图像位于 experiment_mean_valid/，已检查对照图片。
