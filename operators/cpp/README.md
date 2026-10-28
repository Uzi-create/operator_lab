# 纯 C++17 RGB-D 目标感知算子

这里包含与 KCF、检测网络和 ROS 解耦的基础算子、可回放的传统 RGB-D 目标跟踪原型，以及跟随闭环和激光避障组件。算法和回放程序没有 Python、OpenCV、第三方库或动态库加载依赖。它不包含人员检测网络或底盘驱动。接口位于 `include/operator_lab/`，均为 `inline` / 模板，可从多个翻译单元安全包含。激光、制动和局部规划接口详见 [避障与控制说明](NAVIGATION.md)。

## 构建与验证

在仓库根目录运行（macOS / Linux，需 C++17 编译器和 make）：

```sh
make -C operators/cpp test demo
make -C operators/cpp replay
make -C operators/cpp navigation
make -C operators/cpp BUILD_DIR=build-sanitized \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' test demo
```

也可以完全不使用 make：

```sh
c++ -std=c++17 -O2 -Ioperators/cpp/include \
  operators/cpp/examples/depth_target_demo.cpp -o /tmp/depth_target_demo
/tmp/depth_target_demo
```

已有的 `python verify.py --sanitizers` 开发验证入口也会编译并执行新 C++ 测试；这不引入 Python 运行时接口。所有实际算法可以只使用 C++。Windows 可以用支持 C++17 的编译器直接构建；本轮未在 Windows 或 Linux 上运行，不宣称跨平台实测完成。

## 编辑器头文件与 C++17 配置

`Makefile` 的 `-Iinclude` 和 `-std=c++17` 只对构建命令生效，编辑器也需要知道这些参数，否则可能把 `operator_lab/...` 报成找不到文件，并产生大量后续错误。

- clangd：本目录的 `compile_flags.txt` 提供 C++17、相对本目录的 `include/` 搜索路径与警告选项；不依赖本机绝对路径。已有 `compile_commands.json` 时，编辑器应使用其实际构建参数。
- VS Code Microsoft C/C++ 扩展：打开 **operator_lab 仓库根目录**时，使用仓库 `.vscode/c_cpp_properties.json`；直接打开 **operators/cpp 子目录**时，使用该子目录自己的 `.vscode/c_cpp_properties.json`。两种打开方式均配置 C++17，并使用各自工作区对应的头文件路径。
- 子目录配置中的 `Mac` 项明确使用 `/usr/bin/clang++` 与 `macos-clang-arm64`，匹配本轮验证的 Apple Silicon Mac；Intel Mac 应改用 `macos-clang-x64`。`Linux` / `Win32` 项由扩展自动发现编译器，未在对应系统验证。VS Code 不会自动继承工作区根目录之外的父目录配置，Microsoft C/C++ 扩展也不直接读取 clangd 的 `compile_flags.txt`。
- 修改配置后仍显示旧诊断时，重启语言服务器或重载编辑器窗口。这些配置只服务于本库，不会安装 ROS 或为机器人供应商工程补齐 catkin 依赖。

可在仓库根目录用 `clangd --check=operators/cpp/tests/test_navigation.cpp` 检查编辑器解析结果；实际编译和测试仍用上面的 `make` 命令。

## 1. 深度视图、反投影与坐标转换

```cpp
#include <operator_lab/rgbd_geometry.hpp>
using namespace operator_lab::rgbd;

// 假设缓冲区来自相机，单位为毫米；不拷贝、不持有所有权。
DepthView<std::uint16_t> image(buffer, accessible_elements, width, height, 0.001,
                               row_stride_elements);
Intrinsics k{width, height, fx, fy, cx, cy};
auto samples = depth_to_points(image, k, Rect{0, 0, width, height});
if (!samples.empty()) {
    Point3 base_point = transform_point(samples.front().point_m, camera_to_base);
}
```

`camera_to_base` 是调用方提供的 `RigidTransform`，包括行优先旋转矩阵和米制平移，语义是 `T_base_from_camera`。它不会估计外参或查询 TF。

- `DepthView<T>` 支持 `uint16_t`、`float`、`double`，显式指定原始数值到米的比例。步长单位为**元素数**而非字节；OpenCV `Mat::step` 若用于此处，应先除以 `sizeof(T)` 并确认能整除。调用方负责类型、缓冲区实际长度与生命周期。
- 尺寸、可访问长度、步长、ROI、参数和刚体旋转在 C++ 入口校验。无效参数抛 `std::invalid_argument`，越界读取抛 `std::out_of_range`，不可表示的几何结果抛 `std::overflow_error`。
- 输入像素应已去畸变，内参必须匹配当前深度图。相同分辨率不代表 RGB 与深度已经配准；彩色目标框必须先映射到深度图坐标。
- 光学坐标系为 X 向右、Y 向下、Z 向前；深度指光轴 Z，而不是沿射线的欧氏距离。三维输出为米。
- 0、负数、NaN、Inf、缩放溢出和范围外样本被跳过。深度范围为 `min_depth_m < z <= max_depth_m`。缺测不表示可通行。
- `depth_to_points()` 保留原始像素索引，可用 `sampling_step` 采样；它不会缩放图像或修改内参。空点云是有效结果。

## 2. 候选区域的稳健深度估计

```cpp
#include <operator_lab/depth_target.hpp>
DepthTargetConfig config;
// 所有默认参数只是可调起点，不是人体/相机的已验证配置。
auto estimate = estimate_depth_target(image, k, roi, config);
if (estimate.accepted()) {
    const Point3 point = *estimate.point_m;
    // point 可用于后续关联；不能仅凭 accepted 就授权机器人运动。
}
```

算法：检查 ROI 内的有效深度 → 可选的预测深度门限 → 按深度排序 → 滑动窗口找最大支持的固定宽度深度带 → 检查支持比例和次强带歧义 → 中位数及 MAD 检查 → 反投影代表点。

- `band_width_m` 是整条深度带的最大跨度；不会因为一连串小跳变就把整段斜坡连接起来。复杂度为 `O(N log N)`，空间为 `O(N)`，N 为 ROI 像素数。
- 有效比例的分母是 ROI 面积；支持比例的分母是 ROI 内全部有效样本，**不是经过先验过滤后的数量**。较小的残留目标不会被虚报为满支持。
- 歧义检测比较最佳带与其左右、不复用支持像素的次强带；它是深度分布诊断，不是物体数量或遮挡真值。等大小的分离深度带会被拒绝。
- 可传入 `DepthPrior{predicted_current_depth_m, tolerance_m}`。先验必须预测到**当前相机姿态、当前时刻的光学 Z**；旧帧深度在机器人移动后不能直接复用。先验是关联线索，不证明身份。
- `point_m` 是支持像素平均位置与中位深度形成的代表点，不声称是人体质心、脚点或物体中心。
- 仅 `accepted` 返回深度、MAD、代表点和支持像素。其他状态包括有效数据不足、没有先验支持、目标支持不足、多带歧义和深度离散过大。
- 该算子按深度分布分组，不做空间连通分割。同深度的不同人仍可能混合，错误的先验也可能选错人；需要上游的外观与身份关联。

## 3. 已关联目标的时序证据确认

```cpp
#include <operator_lab/target_evidence.hpp>
TargetEvidence evidence;
// 观测已变换到一个跨帧一致的坐标系，例如 odom；target_key 来自上游关联。
auto result = evidence.observe(TargetObservation{target_key, odom_point, stamp_s}, now_s);
// 没有新观测时，每个控制周期仍然调用：
result = evidence.poll(now_s);
```

`ready` 需要满足连续帧数、持续时间、帧新鲜度、相同关联标识和位置连续性。位置门限采用 `position_slack_m + max_speed_mps * dt`，适应不同帧率，而不是固定“每帧最多移动几厘米”。

- 所有时间必须转换为同一个单调秒时钟；不能混用 ROS 时间、设备时间和本机单调时间。时钟倒退会清空状态并抛异常；切换任务、坐标系或主动重选目标时显式 `reset()`。
- 重复/乱序、未来、过期、身份变化、位置突跳或主动报告缺测都会打断确认。缺测时可短暂保留上一个点作为诊断，但 `ready=false`。
- 传感器完全静默时，`poll()` 会撤销过期确认；超出 `hold_timeout_s` 后清除目标记录。不可缓存旧的 `ready=true`；每个控制周期都要更新或查询。该类不是线程安全的，应由调用方串行化访问。
- **不做身份识别，不估计相机运动，不预测丢失目标，不检查障碍。** 相同 key 不代表一定是同一个人，身份仍由上游保证。相机原始坐标不能直接跨帧用于速度门限，必须先补偿机器人运动。
- 恢复确认后，实际运动仍须经过独立障碍检查、速度约束与控制器。此接口不提供碰撞安全保证。

## 验证范围

测试使用独立的闭式投影与穷举深度带参考，覆盖米/毫米、带行填充的图像、非零 ROI、缺测/无效值、越界与参数拒绝、旋转反射拒绝、多深度带歧义、先验过滤、遮挡后重新确认、不同帧率、相机运动补偿、静默超时、时钟倒退及多翻译单元链接。测试不依赖 `assert`，`-DNDEBUG` 下仍会执行。

示例使用合成深度图和已知目标关联，不读取相机、不运行检测网络、不控制机器人。本轮的 macOS 编译/测试与 sanitizer 结果只说明这些离线条件下的验证，不能外推为真实人员跟踪精度、实车避障效果或目标平台实时性。

### 本轮实测（2026-10-26）

- macOS arm64：独立 `-O2 -Werror` 构建，563 项 C++ 检查和合成示例通过。
- `-O3 -DNDEBUG -Werror`：同样的 563 项检查与示例通过，证明检查未被 release 宏移除。
- AddressSanitizer + UndefinedBehaviorSanitizer：新 C++ 检查与示例通过。
- 仓库完整 `verify.py --sanitizers`：356 项现有测试、原生断言及新增 563 项 C++ 检查通过。
- 未做真实深度相机、Linux/Windows 目标机、模型推理、ROS 联调或实车运动验证。

## 4. 空间区域、外观关联与三维预测

新增头文件与输入输出：

| 文件 | 输入 → 输出 |
|---|---|
| `depth_regions.hpp` | 深度图、内参、ROI、可选掩膜及邻接判定 → 空间连通区域、实际支撑像素、代表点 |
| `target_association.hpp` | 候选三维位置/外观、预测位置及方差、参考外观 → 匹配、无匹配或歧义 |
| `motion3d.hpp` | 固定坐标系内带时间戳的三维观测 → 位置/速度估计、预测位置及方差 |
| `rgbd_tracker.hpp` | 已配准 RGB-D、内参、按帧位姿、时刻和初始选择像素 → 跟踪状态及观测/预测位置 |
| `replay_io.hpp` | PPM/PGM、标定和逐帧清单 → 经检查的数据；支持诊断 BMP 输出 |

`extract_depth_regions()` 依据像素邻接和深度跳变分区。相同深度但不连通的物体保持分离，`point_m` 使用区域内**真实像素及其实际深度**，不会将两个人的像素平均到中间空隙。深度中位数和 MAD 单独保存在区域统计中，不能把代表点理解为人体中心。

可传入全图 `MaskView` 排除区域外像素，也可提供 `can_connect(Pixel, Pixel)` 附加邻接约束。`RgbdTracker` 默认使用 RGB 邻域颜色差（8 位 RGB 的 L2 距离）阻止明显不同颜色的相接表面合并，阈值 `max_rgb_neighbor_distance=120` 需要实测调整。噪声、衣服纹理可能分裂同一个人；同颜色、相接且深度接近的不同物体仍可能合并。这些几何区域不是人体实例分割。

`associate_target()` 同时检查三维马氏距离和外观余弦距离，再比较最佳/次佳分数差。可疑的并列候选返回 `ambiguous`，不按数组顺序随意选择。颜色直方图仅统计区域支撑像素；同色衣服、光照变化和遮挡仍可能造成误关联。外部网络生成的非零有限特征向量也可直接传给这个关联算子；当前集成原型固定使用颜色直方图，不是深度学习版本。

`Motion3D` 使用位置/速度恒速模型和每轴 2×2 协方差，按实际 `dt` 加入分段恒加速度过程噪声，采用 Joseph 形式更新测量协方差。创新距离过大时拒绝更新并保留原状态。`predict()` 为只读调用，超出 `max_prediction_s` 返回空值；预测次数不会延长可恢复时间。输出方差来自所设模型及噪声参数，不是经实车标定的身份置信度。

## 5. 传统 RGB-D 跟踪原型

```cpp
#include <operator_lab/rgbd_tracker.hpp>
RgbdTracker tracker;
auto result = tracker.process(rgb_view, depth_view, intrinsics,
    fixed_from_camera, FrameTimes{rgb_s, depth_s, pose_s, now_s}, selection_pixel);
```

第一帧通过 `selection_pixel` 选择区域，此后传 `std::nullopt`。流程为：

```text
已配准 RGB-D → 空间/颜色边界分区 → 区域外观与三维点
                                             ↓
按帧相机位姿 → 转换至固定坐标系 → 预测位置 + 外观关联
                                             ↓
                            运动创新检查 → 连续证据确认
```

- `fixed_from_camera` 由调用方按图像时刻提供，例如 `T_odom_from_camera`；原型不会估计里程计、执行 SLAM 或自动做位姿插值。
- 每帧 RGB、深度和位姿时刻必须在同一个单调秒时钟中。过期、未来、乱序，以及 RGB/深度或位姿时差过大的输入返回 `rejected_frame`。时钟倒退会清空状态并抛异常。
- `observed_position_m` 仅在关联到新观测时提供。`prediction` 是单独的推测结果，遮挡时不会伪装为新观测，`confirmed=false`。
- 短时遮挡后需重新积累确认帧。超过预测时限后保持 `lost`，即使目标重新出现也要求显式重选；不会自动挑一个更近的区域重新开始。
- 参考外观在选择时固定，不自动学习新外观，减少学习遮挡物的风险，但也限制对光照和外观变化的适应能力。
- 完全没有新帧时，仍须每个控制周期调用 `poll(now_s)` 使确认到期。返回的确认结果仅代表跟踪证据，不代表可通行或允许发送速度。

## 6. 文件回放与诊断

在仓库根目录执行，输入与输出目录必须是新的或空目录：

```sh
make -C operators/cpp replay
operators/cpp/build/rgbd_replay --generate-demo /tmp/rgbd-demo-input
operators/cpp/build/rgbd_replay /tmp/rgbd-demo-input/calibration.txt \
  /tmp/rgbd-demo-input/frames.txt /tmp/rgbd-demo-output
```

打开输出的 `index.html` 可逐帧查看 RGB、深度、目标框、投影后的预测位置和状态原因；`results.csv` 包含观测/预测三维位置及算法处理耗时。绿框为已确认，橙框为确认中，青色十字为预测。`processing_ms` 不包含文件读取、图像输出、相机延迟或控制延迟，不能当作整机实时性。

生成器提供 **34 帧合成矩形物体**，包括不同颜色目标交叉、短时遮挡、长时丢失、相机平移以及第 30 帧重选。它不是人员视频，也不代表真实相机效果。程序保留空的观测字段表示没有可靠测量。

接入自己录制的数据需要：

1. RGB：二进制 P6 PPM，最大值 255；深度：二进制 P5 PGM，最大值 65535，16 位大端。深度 0 表示缺测。输入应已去畸变并完成 RGB-D 配准。
2. `calibration.txt` 一行：`width height fx fy cx cy depth_scale_to_m`。示例 `80 48 70 70 40 24 0.001`。
3. `frames.txt` 每帧一行，字段如下（路径相对清单目录，可使用双引号包含空格）：

```text
rgb_s depth_s pose_s now_s "rgb.ppm" "depth.pgm" r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz seed_u seed_v
```

旋转矩阵按行排列，平移单位米。`seed_u seed_v` 为该帧人工选择的目标像素，普通帧写 `-1 -1`。`now_s` 可来自录制的接收时刻，用于回放延迟和过期判断。未知机器人位姿时，只有相机确实固定才能使用单位变换；移动相机不能假装静止。

PNM 读取器只支持这里声明的格式；其他格式需先在数据导出环节转换。未接入真实相机录制器、ROS bag、检测网络、避障或底盘控制。

### 本轮补充验证（2026-10-27）

空间连通性使用独立穷举邻接参考；测试还覆盖同深度分离区域、同深度相接的不同颜色区域、掩膜/ROI、不同帧率、遮挡恢复、预测失效、静默超时、相机平移补偿、外观与距离冲突，以及 PNM 字节序、二进制空白像素、截断文件、清单/标定错误和 BMP 像素顺序。发布前另跑 release 与地址/未定义行为 sanitizer。所有这些仍属于离线/合成验证。

本轮在 macOS arm64 / Apple Clang 21 实测：356 项现有测试、4,863 项 C++ 检查（563 基础 + 4,282 空间/关联/运动/跟踪 + 18 回放 I/O）通过；`-O3 -DNDEBUG -Werror` 与 AddressSanitizer / UndefinedBehaviorSanitizer 构建通过。完整验证入口还检查了 34 帧端到端回放：短时遮挡不产生新观测，长时丢失不自动恢复，重选后重新确认，所有新观测的投影都落在生成输入中的红色目标支撑像素上。34 张 BMP 和 CSV/HTML 输出已生成。没有真实人员视频或实车验证。
