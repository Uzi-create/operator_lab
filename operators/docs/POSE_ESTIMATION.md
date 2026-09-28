# 已标定相机的鲁棒 PnP 位姿

入口：`operators.pose_ops.estimate_pose_pnp`。根据**已知对应关系**的 N×3 物体点和 N×2 像素点，估计 `T_camera_from_object`。内部使用 OpenCV C++ 的 EPNP/RANSAC 与 LM 精修；本模块负责坐标归一化、输入检查、最终质量验证和平面双解诊断。

它不负责识别特征、建立对应、标定相机或控制机器人。至少需要 **6 个对应点**，不作为 4 角点标记的最小求解器。

```python
from operators.robot_ops import Intrinsics
from operators.pose_ops import estimate_pose_pnp, project_object_points

camera = Intrinsics(1280, 960, 900., 920., 640., 480.)
result = estimate_pose_pnp(
    object_points, image_points, camera,
    distortion=None,
    reprojection_threshold=2.0,   # 最终单点最大重投影误差，像素
    max_rms=1.0,                  # 最终内点 RMS，像素
    min_inliers=12,
    min_inlier_ratio=0.6,
)
if result['success']:
    T = result['T_camera_from_object']
    projected = project_object_points(object_points, camera, T)
else:
    print(result['reason'])
```

## 坐标、单位、畸变

- `T_camera_from_object` 将物体坐标转换为相机光学坐标：`p_camera = R @ p_object + t`。相机 X 向右、Y 向下、Z 向前。
- 平移、`min_depth` 与 `object_points` 使用相同单位。输入米，输出米；输入毫米，输出毫米。没有隐式单位换算。
- `image_points` 为 `(u, v)`，必须与所提供内参对应，图像缩放/裁剪后必须同步调整内参。点不必位于图像边界内。
- `distortion=None` 为无畸变针孔模型，适用于已经去畸变、且使用相应新内参的像素。不要对已去畸变像素重复传入原畸变。
- 原始畸变像素可使用 OpenCV 标准模型的一维 4/5/8/12/14 系数向量：`k1,k2,p1,p2[,k3,k4,k5,k6,s1,s2,s3,s4,tauX,tauY]`。**不支持 fisheye 模型**。
- `project_object_points` 支持相同的位姿和畸变规则，背面/零深度点的像素为 NaN；不负责遮挡或图像范围裁剪。已有相机坐标的纯针孔投影继续使用 `robot_ops.project_points`。

## 返回和失败规则

成功返回：`success`、`reason`、`T_camera_from_object`、布尔 `inliers`、`inlier_count`、`rms`、逐点 `reprojection_errors`、`positive_depth_fraction`、`planar`、`ambiguous`、`candidates`。

`inliers` 和 `rms` 在归一化坐标还原为**最终返回位姿之后重新计算**：点必须深度严格大于 `min_depth`、重投影误差不超过 `reprojection_threshold` 且误差有限。成功需要至少 `max(min_inliers, ceil(N * min_inlier_ratio))` 个内点，最终内点 RMS 不超过 `max_rms`。最小内点数不得小于 6。

非法形状、非数值、NaN/Inf、畸变格式或参数抛出 `ValueError`/`TypeError`。正常求解失败返回 `success=False`、`T_camera_from_object=None`、全 False 的主 `inliers` 和无穷主 `rms`，并给出原因：

| reason | 含义 |
|---|---|
| `insufficient_correspondences` | 对应数量不足以达到请求支持数 |
| `degenerate_object_geometry` / `degenerate_image_geometry` | 输入重合或共线 |
| `ransac_failed` | RANSAC 无法找到位姿 |
| `insufficient_positive_depth_inliers` | 满足正深度与重投影门限的点不足 |
| `degenerate_inlier_geometry` / `degenerate_inlier_image_geometry` | 最终共识点几何退化 |
| `planar_ambiguity_check_failed` | IPPE 双候选检查失败或产生无效候选 |
| `final_pose_quality_failed` / `invalid_final_pose` | 还原后的最终位姿未通过质量检查 |
| `ambiguous_planar_pose` | 存在解释相同共识点的近似等价平面位姿 |

求解失败时，可能仍保留 `candidates` 供诊断。候选中的误差和内点也由相应候选的最终位姿计算，不应绕过 `success` 直接拿失败候选控制机器人。

## 平面目标为何会拒绝

最终物体内点的第三奇异值不超过第一奇异值的 `1e-6` 时，另外建立平面局部坐标，使用 IPPE 双候选并分别验证/精修。精修可能把两组种子收敛到相同极小值，因此保留有足够支持的原始 IPPE 候选做歧义检查。

当两个不同姿态都能解释最佳位姿的同一组内点、这些点都在相机前，且候选 RMS 与最佳 RMS 的差不超过 `ambiguity_rms_delta`（默认 0.25 px），默认返回歧义失败。不同姿态定义为相对旋转超过 0.1°，或平移相差超过物体点 RMS 尺度的 0.1%。候选 `reference_rms` 使用同一参考内点集，避免候选通过丢掉难点获得虚假的低误差。

只有调用者有额外约束并愿意自行处理歧义时，才可设置 `allow_ambiguous=True`。此时仍返回 `ambiguous=True`、`reason='ambiguous_planar_pose_allowed'`，并保留候选。**`ambiguous=False` 不证明位姿唯一，也不表示已得到位姿协方差。** 正对、远距离、小目标、重复纹理、标定偏差等问题，单凭小重投影残差不能解决。

## 2026-09-27 本机验证

专项命令：`python -B -m unittest operators.tests.test_pose_ops -v`，13 项通过。测试使用独立 `cv2.projectPoints` 按返回矩阵重新计算误差和掩码，覆盖非平面、任意物体坐标中的平面、噪声离群、全部 5 类畸变长度、单位/大坐标偏移、负深度、退化、支持不足及 IPPE 失败。

下表均为 120 个合成对应点，NumPy 随机种子 123；内参为示例内参。平移真值使用米，因此误差换算为毫米。无噪声浮点级误差不代表真实相机能达到这种精度。

| 场景 | 保留真内点 / 误接纳离群 | RMS (px) | 旋转误差 | 平移误差 |
|---|---:|---:|---:|---:|
| 非平面，无噪声 | 120 / 0 | 5.98e-14 | <1e-6° | <1e-6 mm |
| 非平面，0.3 px 噪声 + 30% 随机离群 | 84 / 0 | 0.43275 | 0.01770° | 0.42059 mm |
| 倾斜平面，0.2 px 噪声 + 20% 离群 + 5 系数畸变 | 96 / 0 | 0.28530 | 0.01293° | 0.27601 mm |
| 远处近正对平面，0.05 px 噪声 | 明确拒绝歧义 | 候选 0.0732–0.0793 | 不承诺唯一解 | 不承诺唯一解 |

短时性能采样（3 次预热、21 次、含检查/归一化/分配/精修/最终验证）：上述前三例中位数分别 **0.876 / 4.105 / 3.335 ms**，P95 为 1.194 / 4.598 / 4.443 ms。平台为本机 Windows、i5-13400F、Python 3.11.4、OpenCV 4.12.0、NumPy 2.2.6；这是短时参考数值，并非隔离负载的完整性能基准。离群比例、迭代预算和几何条件会影响耗时。

当前仅验证合成数据。实际精度依赖标定、点定位精度、对应关系和目标几何；未验证真实相机、手眼标定、机器人运动、macOS 或 Linux。不估计协方差，不保证全局最优。不会设置 OpenCV 的全局随机种子；不同 OpenCV 版本的 RANSAC 行为可能不同。
