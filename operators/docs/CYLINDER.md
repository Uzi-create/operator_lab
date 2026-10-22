# 已知轴向的圆柱表面拟合

`operators.cylinder_ops.fit_cylinder_known_axis` 用于已知轴方向的管件、
圆柱工件和立柱点云。它估计半径、轴线位置和可见点的轴向跨度。
轴方向必须来自标定、夹具方向、重力方向或其他可信上游；算子不会搜索未知轴向。
输入轴偏了，输出也会受到影响，`success=True` 不能证明轴方向正确。

```python
from operators.cylinder_ops import fit_cylinder_known_axis

result = fit_cylinder_known_axis(
    points_xyz,                 # N×3，例：单位为米的已分割管件表面点
    axis_direction=[0, 0, 1],   # 自动归一化，保留传入方向的正负号
    threshold=0.002,            # 径向误差阈值，单位同 points_xyz
    min_radius=0.02,
    max_radius=0.30,
    min_inliers=40,
    min_inlier_ratio=0.65,
    min_coverage_deg=120,
    min_axial_span=0.03,
    max_iterations=512,
    seed=0,
)
if result['success']:
    print('radius:', result['radius'])
    print('axis point:', result['axis_point'])
    print('observed support:', result['segment_xyz'])
else:
    print('rejected:', result['reason'])
```

## 输出与单位

| 字段 | 含义 |
| --- | --- |
| `success`, `reason` | 是否满足配置的支持度、半径、角覆盖及轴向跨度要求；成功原因是 `ok` |
| `axis_point` | 轴线上的一个点，靠近可见支持段中部；不代表物体质心 |
| `axis_direction` | 保持输入符号的单位轴向；任意轴点 `axis_point + t*axis_direction` 均位于拟合轴线上 |
| `radius` | 圆柱表面半径，单位与输入坐标相同 |
| `segment_xyz` | 形状 `(2,3)`，径向内点沿轴方向的投影最小/最大端点；顺序跟随输入轴向 |
| `length_observed` | 两投影端点之间的可见支持跨度；单位与输入坐标相同 |
| `coverage_deg` | 内点角度排序后，`360° - 最大相邻角度间隙`；衡量包围角跨度，不代表角度采样均匀 |
| `inliers` | 长度 N 的布尔掩码，与输入点一一对应 |
| `residuals` | 长度 N 的非负径向误差，`abs(distance_to_axis - radius)`；单位同输入 |
| `inlier_count`, `inlier_ratio` | 最终径向内点数、占全部输入点比例 |
| `rms` | 最终内点径向误差的均方根；单位同输入，不是预测置信度 |
| `candidate` | 被拒绝时保留的候选几何字典；成功时为 `None` |

失败时，顶层 `axis_point`、`axis_direction`、`radius`、`segment_xyz`、
`length_observed`、`coverage_deg` 均为 `None`，防止把被拒绝的几何直接当作测量结果。
如果已经拟合出候选，上述几何放在 `candidate` 中，顶层残差、掩码及统计仍描述这个
最终候选：`inliers == (residuals <= threshold)`，`rms` 使用同一个掩码计算。
没有有效候选时，`candidate/residuals/rms` 为 `None`，掩码全为假。

**支持段不是物理管长，端点不是端盖位置。** 有限圆柱的两个端面不参与此模型。
离轴误差小但沿轴方向很远的点仍然属于无限圆柱表面，它们会扩大 `length_observed`。
需要真实工件长度时，先通过 ROI、连通域、工作空间范围或单独的端面测量获得正确
支持点；单凭这个算子的径向 RANSAC 无法排除轴向离群点。遮挡也可能使可见跨度小于
物理长度。不要把这项输出直接标成“管长测量值”。

## 算法与拒绝条件

先在 float64 中平移并缩放点云，把已知轴向垂直平面的投影转成二维圆拟合。
有界 RANSAC 以三点圆生成候选，使用径向残差统计支持度；随后用几何最小二乘
优化中心和半径，并更新内点。最后从实际返回的三维轴点重新计算所有点的
残差、掩码、均方根和支持跨度。
半径范围同时约束候选搜索和最终结果，避免较大但半径不合要求的表面遮住另一片
符合半径要求的支持点；没有合法半径候选时保留越界候选用于诊断。

常见拒绝原因：

- `insufficient_points`：输入点数达不到 `min_inliers`。
- `degenerate_cross_section`：重复点、轴线上点或投影共线等，无法构成有效圆。
- `insufficient_support`：最终内点数或比例不够。
- `radius_out_of_bounds`：候选半径不在指定范围内。
- `insufficient_angular_coverage`：可见角度太窄，例如只看到一小段弧；小残差不足以保证稳定半径。
- `insufficient_axial_span`：例如一个平面圆环，无法提供要求的轴向支持跨度。
- `numerical_range`：候选几何超出可表示数值范围。

角覆盖不会检查每一个角度区间的采样密度。三簇分离的点可能有较大的角覆盖，但
它们并不能证明整个圆柱表面被观测。非圆截面、错误轴向、多个目标混合也可能在
宽松阈值下通过；阈值应按传感器噪声、标定精度和实际工件尺寸设置。

## 参数边界与性能

- 输入必须为有限实数 `N×3` 数组；最多 200 万点，坐标绝对值不超过 `1e150`。
  大平移量不能恢复输入浮点数中已经丢失的小尺寸精度。
- 轴方向必须为有限非零三维实数向量；输入点和方向都不会被修改。
- `threshold > 0`；`0 <= min_radius < max_radius`；`max_radius` 可为正无穷。
- `min_inliers >= 3`，`0 < min_inlier_ratio <= 1`。
- `0 <= min_coverage_deg <= 360`；实际有限采样的完整圆覆盖通常也略小于 360°。
  `min_coverage_deg=0` 关闭覆盖要求，会放过不稳定短弧，需自行承担测量约束。
- `min_axial_span >= 0`，设为 0 可接受平面圆环，但不能据此推断物理圆柱存在。
- `max_iterations` 为 1 至 10000 的整数，`seed` 为非负整数。默认迭代上限 512，
  按支持比例自适应提前停止；固定输入及 seed 可重复。有限次数 RANSAC 不保证全局最优。
- 参数或数组格式错误抛出 `ValueError`；正常几何失败使用 `success=False` 返回。

投影、距离计算、掩码和最小二乘都由 NumPy 原生数组运算执行，没有逐点 Python 循环；
候选迭代及小规模迭代控制在 Python 层，未新增自定义 C++ DLL 接口。
若 N 为点数、K 为实际 RANSAC 次数，主要计算为 `O(K*N)`，角排序为 `O(N log N)`，
内存为 `O(N)`。200 万是输入保护上限，不是实时处理承诺。
较低内点比例、较紧阈值会增加迭代次数；建议先做工作空间裁剪和合理体素降采样。
降采样会改变权重、角覆盖及可见跨度，因此需要根据测量要求检查偏差。

运行回归：

```powershell
python -m unittest operators.tests.test_cylinder_ops -v
```

回归包含任意轴、30% 离群、噪声、`1e8` 量级坐标平移、短弧与平面环拒绝、轴向
离群语义、反轴等价性、输入不变和独立三维残差核对。合成测试不替代实际传感器验证。
