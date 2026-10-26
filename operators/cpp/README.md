# 纯 C++17 RGB-D 目标感知算子

这是一组与 KCF、检测网络和 ROS 解耦的基础算子，没有 Python、OpenCV、第三方库或动态库加载依赖。它不构成完整目标跟踪器，也不直接发送运动指令。接口位于 `include/operator_lab/`，均为 `inline` / 模板，可从多个翻译单元安全包含。

## 构建与验证

在仓库根目录运行（macOS / Linux，需 C++17 编译器和 make）：

```sh
make -C operators/cpp test demo
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
