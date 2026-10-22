# 有错配的 3D 对应点刚体定位

`operators.correspondence_ops.estimate_rigid_transform_robust` 从**已经建立对应关系**的两组 3D 点估计刚体位姿，可用于 RGB-D / 双目关键点对应后的初始定位，或作为 ICP 的初值。实现使用 NumPy 的底层矩阵运算、SVD 和三点 RANSAC，不依赖新增库。

它不负责发现对应点，也不代表全局唯一物体识别。多个物体、重复结构或一致的错误匹配可能产生另一个有充分支持的刚体解；调用者仍需结合身份、工作空间或时序约束判断。

```python
from operators.correspondence_ops import estimate_rigid_transform_robust

result = estimate_rigid_transform_robust(
    source_points, target_points,  # 同形状 Nx3，source[i] 对应 target[i]
    threshold=0.005,              # 若输入单位为米，则此处为 5 mm
    min_inliers=12,
    min_inlier_ratio=0.4,
    max_iterations=512,
    confidence=0.999,
    seed=0,
)
if result['success']:
    T_target_from_source = result['transform']
    aligned = source_points @ T_target_from_source[:3, :3].T + T_target_from_source[:3, 3]
else:
    print(result['reason'])
```

## 输入和结果约定

- 坐标必须是有限实数，同形状 Nx3，每组至多 2,000,000 点；支持列表、非连续数组和只读数组，不修改输入。双方必须使用相同长度单位。
- 返回矩阵满足 `p_target = R @ p_source + t`。旋转为正交矩阵，`det(R)=+1`；不允许缩放或反射变换。
- `threshold` 是三维欧氏距离阈值，必须有限且大于零。`min_inliers` 为至少 3 的整数，默认 6；`min_inlier_ratio` 在 `(0,1]`，默认 0.5，二者必须同时满足。
- `max_iterations` 为 1–10,000 的整数，默认 512；`confidence` 在 `(0,1)`，默认 0.999；`seed` 为非负整数。相同环境、输入和 seed 可复现。输入或参数无效抛 `ValueError`；合法数据不足则返回失败。
- `success` 为真时 `transform` 才可使用；失败时为 `None`。`candidate_transform` 可用于诊断，不能直接驱动机器人。它与成功时的 `transform` 是同一估计矩阵。
- `residuals` 是从最终候选矩阵和原始坐标重算的全部 N 个距离；`inliers = residuals <= threshold`，`inlier_count` / `inlier_ratio` 是该掩码的统计，`rms` 是内点距离均方根。无候选时 `residuals` 为 N 个正无穷，内点全假，`rms=None`；如需 JSON，调用者应将非有限数转为 `null`。
- `iterations` 只计随机三点试验次数；干净数据可直接精修返回 0。初始全点拟合和精修不计入随机次数。

| reason | 意义 |
| --- | --- |
| `ok` | 当前候选满足配置的几何与支持要求 |
| `insufficient_points` | 点数不足以达到支持要求 |
| `degenerate_geometry` | 无有效非共线候选，可能为重复、共线点或随机预算不足 |
| `insufficient_inliers` | 最终候选的内点数或比例不足 |
| `degenerate_inliers` | 最终内点几何无法可靠约束旋转 |
| `numeric_range` | 浮点动态范围不支持此次坐标/阈值组合 |

## 精度和性能取舍

先尝试全点 Kabsch 拟合，干净数据省去随机搜索；带错配时采用不放回三点采样，根据当前支持率自适应缩短随机预算。初始全点精修得到的支持率也会应用相同的预算公式，避免已有好解仍盲跑全部迭代。候选和精修的距离判定以矩阵运算覆盖所有点，不逐点运行 Python 循环。每次候选的局部精修最多 8 次；不分配点数平方级距离矩阵，临时内存随点数线性增长。随机阶段最坏处理量随 `max_iterations × N` 增长；上述输入上限不是实时延迟保证，大点集加最大预算仍可能很慢。

样本和最终内点的第二奇异值必须大于第一奇异值的 `1e-6`；平面上的非共线点可接受，共线、重复或极细长支持会被保守拒绝。重复行不等于独立观测，调用前应避免把同一对应关系复制多次凑内点数。阈值应根据传感器三维误差设置，而非只看图像像素误差。

局部坐标中心化和尺度归一化减少大世界坐标偏移及单位差异的数值问题。返回矩阵后仍用原始输入重算全部诊断，确保掩码与实际返回矩阵一致。若世界原点极大而目标精度接近 float64 的坐标分辨率，建议上游先转换到邻近局部坐标系；本函数无法恢复输入中已丢失的精度。

`confidence` 是标准 RANSAC 采样预算参数，**不是位姿正确概率**。自适应公式假设支持集有效；退化样本、结构化错配或相关观测会削弱该解释。达到次数上限仍可能漏掉真实解。较低支持比例需要更大预算；60% 错配应同时把 `min_inlier_ratio` 配置在 0.4 以下并检查实际内点分布。

自动测试使用已知位姿、噪声、30–60% 随机错配、平面、共线/重复、反射、尺度变化、单位变化和大坐标偏移。它们验证数值行为，不等价于真实机器人场景的识别率或安全性。
