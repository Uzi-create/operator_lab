# USB 摄像头与实时算子

左侧显示原始彩色画面，右侧显示处理结果。输入由 OpenCV 读取，灰度转成 float32 `[0,1]` 后调用本项目的 C++ 算子；不用静默回退来掩盖动态库问题。

本机系统识别到 HK 1080P RGB 和 HK 1080P NIR。2026-09-29 本次 OpenCV 实测：索引 1 为彩色桌面画面，索引 0 为灰度桌面画面；分别保留验证结果。普通 AVFoundation 枚举次序不能直接当成 OpenCV 索引：OpenCV macOS 后端按设备 uniqueID 排序，见 [官方源码](https://github.com/opencv/opencv/blob/4.x/modules/videoio/src/cap_avfoundation_mac.mm)。索引可能随设备连接情况或换电脑变化。

2026-09-29 已在本机实测 RGB 通道：640×480 连续读取成功，八种显示效果各处理 15 帧；实时窗口成功创建并报告可见。测试没有验证缺陷检测准确率或相机标定精度。真实帧处理计时见 `projects/output/camera/all_operators_rgb.json`。

```bash
cd /Users/cathypark/Documents/创建/operator_lab
../operator_lab_env/bin/python projects/camera/live_operators.py --camera 1
```

窗口获得焦点后按数字键切换：0 灰度、1 自适应平滑、2 CLAHE、3 Canny、4 动态阈值、5 金属细线响应、6 开闭运算、7 均值。Q/Esc 退出。细线响应是候选响应，不能直接当作缺陷结论。

默认申请 640×480，实际尺寸以终端输出为准；处理时最多宽 640 像素，保持宽高比。窗口显示的处理时间包含灰度转换、Python/C++ 调用和输出转换，不是纯内核耗时。

后台实测（不保存原始照片）：

```bash
../operator_lab_env/bin/python projects/camera/live_operators.py --camera 1 --headless --frames 60 --report projects/output/camera/probe.json
```

如果日志显示未授权，请先处理 macOS 的摄像头访问提示，必要时在“系统设置 → 隐私与安全性 → 摄像头”允许实际运行程序的应用，然后重新运行。相机能枚举不代表已有权限或已经读到图像；`LIVE` 输出才表示至少成功读取并处理了一帧。

## 一次验证 38 个图像算子

### 使用已连接的 iPhone

2026-09-29 当前设备列表按 OpenCV macOS 后端的 uniqueID 顺序排列：NIR=0、USB RGB=1、内置相机=2、iPhone=3。重连或换机器应重新确认索引。iPhone 实际读取 5 帧，Captured 与 processing 均为 1920×1080，自适应算子调用成功；GUI 另行验证，记录 `iphone_probe.log` 与 `iphone_window.log`。设备格式和控制能力查询见 `iphone_device_info.json`。

```bash
cd /Users/cathypark/Documents/创建/operator_lab
../operator_lab_env/bin/python projects/camera/live_all_operators.py --camera 3
```

通道声明支持 640×480、1280×720、1920×1080 和 1920×1440，部分格式声明支持 60 FPS；本次未测量持续帧率。查询到的自动、连续自动、锁定对焦及自定义曝光控制均为 false，代表当前 Mac 采集接口暴露的能力，不可据此判断 iPhone 镜头硬件没有自动对焦。现有 OpenCV 程序能取图和处理，未实现手机镜头切换或手动对焦控制。

使用期间保持手机锁屏、稳定放置、后置镜头无遮挡；解锁或暂停可能中断连续互通相机，参见 [Apple 官方说明](https://support.apple.com/en-us/102546)。

### 用分数辅助比较清晰距离

```bash
cd /Users/cathypark/Documents/创建/operator_lab
../operator_lab_env/bin/python projects/camera/focus_meter.py --camera 1
```

默认 1080P 采集。左侧显示中心 640×480 区域的原始像素，右侧概览标出该区域；不应用锐化、降噪或自适应平滑。显示 Laplacian 方差、梯度能量、平均亮度和近黑/近白像素比例；最多 15 帧取中位数以减小跳动。

将同一张大号印刷文字放入中心框，保持光照与相机稳定，逐步改变距离；每次停稳约 1 秒后按 B 记录当前分数和完整采集 PNG。右侧列出最近 8 个样本；C 清空当前滑动评分历史（保留已有记录）；Q 退出。报告与样本保存到 `projects/output/camera/focus/`，终端打印路径。报告同时保存记录时的单帧指标，避免把滑动中位数误认为该 PNG 的单帧分数。

两个分数只是局部高频/纹理强度，并非焦距、真实对焦距离或绝对画质。纹理、物体在画面中的大小、噪声、曝光和运动同样会改变它们，因此不同距离的分数不能单独证明哪个位置光学对焦最好；应结合保存的文字图比较。没有通用“多少分算清楚”的阈值，也不能用分数证明传感器原生分辨率。

### 实时查看全部 38 项

```bash
cd /Users/cathypark/Documents/创建/operator_lab
../operator_lab_env/bin/python projects/camera/live_all_operators.py --camera 1
```

左侧点击任意算子名字，右侧对照原图和结果。N/P（或 ]/[）切换下一项/上一项；空格开启或关闭每 2 秒自动轮播；V 切换统计算子的分数、背景、方差、残差、有效范围和支持数量图；Q/Esc 退出。关闭窗口也释放相机。

主窗口默认申请 **1920×1080**，算子按实际采集分辨率处理，预览默认 **100% 局部细节**：一个图像像素对应一个画布像素，直接裁出可见区域，不先把整幅图缩小再放大。操作方式：

- 在原图或结果区域按住鼠标左键拖动，左右同步查看同一场景位置。
- 滚轮或 `+` / `-` 缩放；放大时使用最近邻显示，便于观察实际像素。
- `1` 恢复 100% 并回到图像中心。
- `F` 适应窗口查看完整画面（此模式会缩小显示）。
- 拖大主窗口时，支持窗口尺寸查询的后端会增加可见像素区域。

100% 表示图像与画布的像素比例；操作系统显示缩放仍可能影响物理屏幕上的大小。它不会恢复镜头失焦造成的细节。原图 pane 使用完整采集帧，和可选的算子降采样设置分开。

高清查看：

```bash
../operator_lab_env/bin/python projects/camera/live_all_operators.py --camera 1 --width 1920 --height 1080
```

R 打开/关闭独立采集原图窗口，可以拖大窗口；S 保存当前未缩小、未经过算子处理的采集帧到 `projects/output/camera/originals/`，终端打印完整路径。PNG 保留采集帧的像素，不是传感器 Bayer RAW。独立窗口仍可能按窗口大小缩放显示，要检查像素细节可打开保存的 PNG 放大查看。

默认 `--max-width 0`，按实际采集分辨率执行算子。需要更快的处理时可显式指定 `--max-width 640`；原图视口和保存仍使用完整采集帧，但结果只能显示该较低处理分辨率。主窗口显示 Captured 与 processing 两个尺寸。2026-09-29 实测 RGB 返回 1920×1080×3；这是输出图像尺寸，尚未核实传感器原生像素数。设备报告 30 FPS，不代表已实测持续帧率。

38 项均可实时选择：图像结果直接显示；区域/匹配返回框；测量显示搜索位置、观测点与通过检查的拟合；模板创建显示边缘或 ORB 特征点。黄色表示观测点、蓝色表示近似搜索位置、绿色表示接受的结果。未检出的测量只显示诊断点和原因，不画成成功拟合。没有足够纹理/边缘时显示拒绝原因，执行错误显示 EXECUTION ERROR。

模板仍为每帧创建的同帧示例，形状仍限制角度 0、缩放 1 与源模板附近 ROI；不是跨帧目标跟踪。菜单中的 `estimate_translation` 使用上一帧与当前帧，调用相机专用的 `estimate_scene_motion`。绿色箭头表示**画面内容从上一帧到当前帧的位移**；静态背景且近似纯平移时，相机平移方向与之相反。它不是机器人轨迹，旋转、景深视差和独立运动目标需要额外几何信息。首帧、纹理不足、重复纹理或运动不一致时不画箭头，并显示拒绝原因。算法内部最多按 480 像素宽分析，返回位移仍换算为处理画面的像素；若设置 `--max-width`，这里的像素单位就是缩小后的处理画面。API 耗时不含共享输入准备、匹配前的模板创建和显示；模板创建项本身单独计时。

2026-09-29：9 项浏览器/验证脚本/入口测试通过；真实 USB RGB 画面轮流执行全部 38 项，无执行错误。窗口显示另行检查，记录在 live_all_window.log。此前的 live_operators.py 保留 8 种简易效果。

2026-09-30 绿色箭头优化回归：含相机入口在内的全库 286 项测试、C++ 断言、27 组演示和 177 张 PNG 解码通过。相机入口在合成 `(+12,-7)` 像素画面位移上测得约 `(+12.01,-7.02)` 像素；循环别名、重复纹理和双运动层会拒绝箭头。本次 Windows 环境尝试打开相机索引 1 时设备未能打开，不能将合成验证当作当前现场相机验证。运行性能基准和适用边界见[算子说明](../../operators/docs/SCENE_MOTION.md)。

### 自动跑完并保存报告

先退出实时预览，再运行：

```bash
cd /Users/cathypark/Documents/创建/operator_lab
../operator_lab_env/bin/python projects/camera/verify_camera_operators.py --camera 1 --frames 6 --save-previews --output projects/output/camera/full_check_camera1
```

`verify_camera_operators.py` 是调用程序；算法仍在上一级项目的 `operators/` 包里。先采集同一批真实帧，然后逐个调用，第一帧预热，其余帧计时。默认不保存图像；`--save-previews` 保存最后一帧的处理效果和原图对照拼图。

覆盖范围：

- 10 个基础模式：threshold、mean、adaptive、mean_naive、guided、clahe、erode、dilate、open、close。
- Canny、距离变换、有符号距离、羽化、细线响应。
- 方框统计、环形统计、动态阈值、光照校正、局部缺陷对比度。
- 滞后阈值、填孔、区域筛选、区域特征、碎片分组。
- 边缘卡尺、条纹、圆、直线和矩形测量，以及二维直线/圆拟合。
- 帧间平移估计、灰度模板匹配、形状模板创建/匹配、ORB 模板创建/匹配。

并非全部都是自写 C++：报告的实现列区分 custom C++、OpenCV/NumPy 和组合调用。优化均值/朴素均值属于同一算法的两种实现；方框/环形统计也是同一接口的不同参数，因此“38”表示验证的操作项目数。

结果目录：

- `full_check_camera1/RESULTS.md`：彩色通道逐项结果和耗时，先看它。
- `full_check_camera1/gallery.png`：可输出图像的算子效果拼图。
- `full_check_camera1/report.json`：所有返回值摘要、未检出次数和错误详情。
- `full_check/`：索引 0 灰度画面的独立验证结果。
- `regression_full_check.log`：256 项回归测试与 C++ 地址/未定义行为检查结果。

两个索引均实测 640×480、每个项目 6 帧，没有执行错误；各自 16 项真实帧数值对比通过。圆与矩形测量均未检出可靠目标，直线测量也存在未检出。它们使用近似中心几何位置，不是已标注目标。模板使用同帧自匹配，形状匹配只验证角度 0、缩放 1 和源模板附近的 ROI；不能据此宣称跟踪、旋转匹配或检测精度达标。

耗时包括算子 API 调用，不含采集、灰度转换、共享掩码/模板裁剪准备、结果验证或写文件；模板创建单独计时。效果图中的距离与分数会归一化显示，数值以 JSON 为准。真实帧数值对比覆盖均值优化/朴素、细线 C++/NumPy、方框和环形统计 C++/OpenCV；整数/掩码严格相等，浮点 atol=rtol=2e-5。

双目、深度、点云、机器人、手眼与物理尺寸测量需要其他数据或标定，只完成项目回归测试，未用这台摄像头做硬件验证。

### iPhone 实验 1：相同均值算法的两种实现

从 [EXPERIMENT_01.md](EXPERIMENT_01.md) 开始，调用程序是 `experiment_mean.py`。当前 iPhone 索引 3。

```bash
cd /Users/cathypark/Documents/创建/operator_lab
../operator_lab_env/bin/python projects/camera/experiment_mean.py --camera 3
```

N/P 切换半径，Q 退出。加 `--benchmark` 使用固定同一帧、每组 7 次采样，保存报告与图片。
2026-09-29 iPhone 实测 1920×1080；正常场景复测：31×31 窗口 mean 4.586 ms、mean_naive 931.486 ms，中位 API 耗时约相差 203 倍，该帧最大误差为 0。有效场景报告在 projects/output/camera/experiment_mean_valid/RESULTS.md；首次 experiment_mean 输出近黑画面，不作有效场景报告。两个实现均为自定义 C++。时间不含采集与显示，不能视为端到端 FPS。
