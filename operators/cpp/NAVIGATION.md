# 激光避障与目标跟随控制（纯 C++17）

课题平台为用户任务书中的 **WHEELTEC R550 MEC PLUS、深度相机和激光雷达**。本目录新增的是不依赖 ROS、Python、OpenCV 的算法与离线验证程序。R550 的实际尺寸、雷达外参、延迟、制动能力和话题尚未实测；代码中的合成配置不能当作该车出厂参数。

## 算法结构

```text
RGB-D 跟踪结果 → 固定坐标中的真实观测与原始时间戳
                            ↓ 当前车体位姿
                     距离/方位 P 闭环 → 期望速度
                                                ↓
激光扫描 + 逐束位姿 → 当前车体坐标局部栅格 → 局部速度采样
                                                ↓
                         车体扫掠 + 延迟 + 完整制动轨迹检查
                                                ↓
                         有效性/急停/恢复确认 → 最终速度
```

此结构无需全局地图或 SLAM，但仍需可靠的里程计/TF、时间戳和传感器外参。当前程序没有发布 `/cmd_vel`，也未连接真实底盘。

| 文件 | 能力 |
|---|---|
| `planar_geometry.hpp` | 平面刚体变换、含横移的 SE(2) 积分、速度约束、加速/制动模型 |
| `laser_grid.hpp` | 激光有效性、逐束时间、运动去畸变接口、局部未知/空闲/占据栅格 |
| `collision_check.hpp` | 非对称矩形车体与栅格的 SAT 相交、转弯扫掠、带制动尾段的轨迹检查 |
| `follow_control.hpp` | 距离/方位误差、死区、先转向、速度上限、目标过期停车 |
| `local_avoidance.hpp` | 动态速度窗口采样、可选横移和局部绕障路点、无可行速度时停车 |
| `follow_navigation.hpp` | 末端指令仲裁、急停/传感器失效停车、独立新扫描恢复确认 |
| `tracking_navigation.hpp` | RGB-D 观测缓存、当前车体系目标转换、深度障碍高度筛选与正证据融合 |
| `laser_scan_io.hpp` | 保留有限 float 数值和 NaN/Inf 语义的原始扫描文件读写 |

## 输入约定

- 车体坐标：X 前、Y 左、Z 上，角速度绕 Z 轴，米/秒/弧度。不要把相机光学 Z 直接当成车体 X；使用已验证的外参转换。
- `RobotFootprint{front, back, half_width, margin}` 相对指定车体原点，四项单位米。零默认值无效，必须显式给出几何尺寸。车体的前后长度可以不同。
- `MotionLimits` 中平移速度上限约束 **XY 合速度**；`max_lateral` 单独约束横移。加速和制动上限应取实际车在相关方向、负载和地面下可支持的保守值。
- `LaserScanView` 使用原始量程数组、角起点/增量、最小/最大量程、第一束采集时间及逐束时间增量。支持负角增量和局部视场。
- `LaserObservation` 的局部栅格必须对应当前决策参考车体坐标，`reference_stamp_s` 应接近本次控制时刻。复用旧扫描时，应按当前里程计重新投影；不能仅更新这个时间字段假装完成了运动补偿。
- 目标、激光、里程计和当前时间必须处于同一时间域。不能混用设备时钟、ROS epoch 和本机单调时间。仲裁器发现时钟倒退会返回零速度并清空恢复状态。

## 激光与未知空间

`laser_to_grid()` 保留未知区域。NaN、零、负无穷、越界量程不会清出空闲空间；小于 `range_min` 的盲区也不会被自动清空。正无穷默认视为未知，只有核对驱动语义后才能显式开启 `positive_infinity_is_clear`。有限 `range_max` 默认仍视为一个保守的障碍返回。

射线只清出被观测到的栅格，障碍端点优先于空闲证据。车体扫掠进入未知区或地图外均会被拒绝。车体本身已经占据的**完整内部格子**可标为空闲，但不会清除障碍返回，也不会替车体周围安全边距制造空闲数据。部分视场可能因此禁止横移或旋转，这是预期行为。

较长扫描默认要求 `reference_from_beam_base[i]`，将第 i 束时刻车体系中的点变换到当前参考车体系。可由里程计按束时间插值得到。`constant_twist_deskew()` 提供恒定实测体速度近似；加速、打滑和急转情况下应使用实际逐束位姿，不能把该近似当成真值。

## 控制、避障与制动

`follow_command()` 采用距离/方位 P 反馈，带距离与方位死区。目标太近时保持停车，不自动倒车；超过可配置的可信目标距离上限或距离计算溢出时撤销目标有效性；方位偏差大时先转向。平滑速度通过加速度窗口实现，仲裁器会使用实际控制间隔（上限为配置值）限制一步可改变的速度。

默认局部策略为停车让行；`AvoidanceConfig.allow_detour=true` 开启局部绕行。在已观测的通道内，规划器围绕阻挡物选择局部路点，再采样可达的前向/横向/角速度。目标本身依然属于碰撞图，不能为继续跟随而从安全检查里删除。局部规划在死路、遮挡后未知空间或无可行通道时会停住，不保证像全局规划器一样找到所有绕行路线。

每个候选速度都检查：当前状态 → 反应/执行延迟 → 受限加速 → 一段运动 → **完整制动尾段**。不能只检查目标框、机器人中心或固定长度的前方扇区。时间步会按线速度、角速度和车体角点半径缩小，并为采样间运动增加扫掠边距。已选速度还会在末端再次检查。

`TrajectoryConfig.obstacle_speed_bound_mps` 将障碍可能的运动扩张到未来保护区。示例默认值 0.5 m/s 只是算法起点，**不是行人速度保证**；静态场景可显式设为零，横穿场景使用了覆盖所模拟 0.4 m/s 行人运动的 0.5 m/s 上界。实际速度更快、突然出现、雷达漏检或移动物体主动进入已停车车体时，单纯这个模型不能保证无碰撞。需要实际测试速度/延迟边界、持续更新扫描和硬件急停。

`FollowNavigator` 在急停、目标未确认/过期、雷达失效/过期、里程计速度过期、参考坐标时间不匹配、未知制动空间或无可行速度时请求零速度。紧急零指令不经过延迟停车的平滑器；**零指令仍需要物理制动距离**。`braking_path_clear=false` 表示模型下无法确认制动空间，不能理解为发零后已经安全停住。

恢复需要若干个独立的新扫描及持续时间，重复使用同一帧不能凑够恢复次数。中途目标丢失或障碍重新阻挡会重新开始确认。`moving_allowed` 是本次模型检查的结果，不是硬实时认证或实机安全证明。控制器应只保留一个最终 `/cmd_vel` 发布者，避免其它节点绕过仲裁器。

## 与 RGB-D 组件连接

```cpp
#include <operator_lab/tracking_navigation.hpp>
using namespace operator_lab;
navigation::TrackingTargetCache target_cache;
// 只在收到一个新的视觉处理结果时调用；保留原始图像时刻。
target_cache.observe(tracker_result, camera_capture_stamp_s);
auto target = target_cache.target(current_base_from_fixed_tracking_frame);
// 原始扫描与里程计 -> laser_to_grid(..., per_ray_poses)，随后：
navigation::FollowNavigator controller(explicit_footprint, measured_limits, config);
auto command = controller.step(target, laser_observation, measured_body_velocity, now_s);
```

实际使用应长期保存 `controller`，每个控制周期调用 `step()`。`TrackingTargetCache` 不会把预测位置伪装成新观测，也不会在查询时刷新观测时间；视觉明确丢失时会撤销目标确认。目标的固定坐标点由调用方按当前车体位姿转换到当前车体系。

`height_obstacles()` 可以从已转换到当前 +Z 向上车体系的深度点云提取指定高度带，补充激光扫描平面之外的障碍；高度相对显式的 `floor_height_m`，不会自动估计地面。`add_obstacles(laser, points, capture_stamp_s, reference_stamp_s)` 仅添加占据证据，不以缺测点清出空间，并同时检查原始采集时刻的新鲜度和参考时刻一致性。重新变换坐标不能让旧观测变新。过期/不同坐标系数据不应融合。

传感器元数据错误在底层算子中会抛异常；实际 ROS 适配层必须捕获并提交无效观测/零指令，同时设置底盘速度超时。当前尚未实现并验证 ROS 节点、时间同步器或 R550 通信适配器。

## 构建和合成闭环回放

```sh
make -C operators/cpp test navigation
operators/cpp/build/navigation_replay /tmp/navigation-new-output
```

输出目录必须新建或为空，程序不会覆盖已有数据。`index.html` 链接六个逐帧查看器；每个场景包含 `results.csv`、`parameters.txt`、`summary.txt`、SVG 画面和每帧原始 `.scan`。配置标明为合成参数。

| 场景 | 检查目标 |
|---|---|
| `clear_follow` | 距离闭环收敛到期望距离附近 |
| `crossing_person` | 有界速度行人横穿时让行，通道清楚后恢复 |
| `static_detour` | 麦轮侧移局部绕过箱体并继续接近目标 |
| `blocked_passage` | 通道完全堵住时停住，不虚报绕行成功 |
| `sensor_dropout` | 全部雷达样本失效时停车，恢复后重新确认 |
| `target_loss` | 目标未确认时停车，恢复后重新确认 |

模拟器使用已知目标位置而非人员检测网络，扫描为合成瞬时 2D 激光，速度执行使用离散延迟/加速/制动模型。独立于栅格的连续世界矩形碰撞/间距检查每 20 ms 采样一次；回放结果只是这些模型下的检查，不是实车无碰撞证明。

## 原始扫描交换格式

```text
OPERATOR_LASER_V1
count angle_min angle_increment range_min range_max first_stamp_s time_increment_s
range_0 range_1 ... range_N-1
```

`read_laser_scan()` / `write_laser_scan()` 保留有限 float 数值和 NaN/Inf 语义。文件不包含外参、里程计或目标位置；对应运行数据在场景 CSV/参数中。用户真实记录应保留这些信息并统一时钟；当前合成生成器没有读取硬件或伪造真实实验记录。

## 任务书覆盖边界

| 任务 | 当前状态 |
|---|---|
| 深度相机目标感知 | 已有传统 RGB-D 算法与回放；人员检测模型/真实人员稳定识别仍待开发验证 |
| 距离/方位闭环跟随 | C++ 算子与合成闭环已实现，实车参数/效果未验证 |
| 激光雷达避障防护 | C++ 投影、局部规划、末端保护与合成场景已实现；真实驱动/ROS 接入未验证 |
| 实机实验与论文数据 | 尚未进行；所有本轮结果明确标注为合成数据 |

参考接口：[ROS LaserScan 定义](https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/msg/LaserScan.msg)规定第一束时间及量程范围；[Nav2 Collision Monitor](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/collision_monitor/configuring_collision_monitor_node/)体现规划之外的末端速度保护思路。这里借鉴接口和分层原则，不要求将当前 ROS1 工程迁移到 ROS2。

## 本轮验证（2026-10-27）

macOS arm64 / Apple Clang 21：完整验证入口通过 356 项原有测试、5,457 项 C++ 检查（此前 4,863 + 本轮 594）；8 个导航头文件独立编译通过。`-O3 -DNDEBUG -Werror` 构建和 AddressSanitizer / UndefinedBehaviorSanitizer 检查通过。还完成 34 帧 RGB-D 回放、6 个控制闭环场景及 630 份原始扫描/CSV/SVG 记录检查。

以下结果来自声明的合成尺寸、速度、延迟和制动模型：

| 场景 | 最小世界几何间距（米） | 最终目标距离（米） |
|---|---:|---:|
| 空旷跟随 | 0.812466 | 1.21247 |
| 行人横穿 | 0.210000 | 1.47951 |
| 静态箱体绕行 | 0.244907 | 1.21504 |
| 堵塞通道 | 0.550000 | 2.80000 |
| 雷达失效后恢复 | 0.822466 | 1.22247 |
| 目标丢失后恢复 | 0.822466 | 1.22247 |

静态绕行的最大横向位移约 0.735 米。世界几何间距使用未膨胀的车体矩形和真实的合成障碍矩形计算，与规划栅格分开；它仍受离散执行模型和 20 ms 采样限制。未进行真实相机/雷达、Linux/Windows 目标机、ROS 或 R550 实机验证。程序中的控制器与缓存实例应由调用方串行访问。
