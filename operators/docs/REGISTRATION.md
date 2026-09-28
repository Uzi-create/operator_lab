# 精确最近邻和局部刚体 ICP

`NearestNeighborIndex(target, backend='auto')` 保存 3D 目标点快照；原生后端为平衡 KD 树，`numpy` 后端为分块穷举。两个后端都算精确浮点平方距离，等距选原始目标中较小的索引。`query(points, max_distance=...)` 返回 `indices`、`squared_distances`、`distances` 和 `valid`；门限是包含等号的。无匹配索引为 `-1`，距离为无穷。建议用 `with` 在大量查询后及时释放索引。原生查询支持并发读；查询执行时不得调用 `close()`。

```python
from operators.registration_ops import NearestNeighborIndex, icp_point_to_point

with NearestNeighborIndex(target, backend='native') as index:
    neighbors = index.query(query, max_distance=0.05)
    aligned = icp_point_to_point(source, index, max_distance=0.08,
                                 trim_fraction=0.9, min_overlap=0.7)
    if aligned['converged']:
        T_target_from_source = aligned['transform']
```

`icp_point_to_point` 对源点应用 `T_target_from_source`，通过最近邻重复拟合刚体更新；支持距离门控和按残差保留的 `trim_fraction`。`overlap` 是修剪前通过距离门限的源点比例；`inliers` 是修剪后用于拟合的源点。达到迭代上限返回 `converged=False`，但仍返回最后位姿。输出的目标索引、距离、RMS 和内点在**返回位姿**上重新求值，不能用上一轮残差冒充最终结果。坐标及全部距离参数由用户保持同一单位。

需要有用的初始位姿、足够重叠和非共线支持点。重复结构、局部极小值、错误单位或错误初始位姿可能给出低 RMS 的错误配准；`converged=True` 只代表更新幅度很小，不能证明全局真值。许多源点可对应同一个目标点，这是 point-to-point ICP 的当前语义。无 SciPy 依赖。

20 项专项测试覆盖精确 NN、等距索引、门限、大坐标偏置、缓存不可变、句柄释放/并发、原生 ABI 错误、ICP 的噪声离群、退化和最终残差一致性。联合合成示例中 620 源点、780 目标点（70 个源离群点）、真值旋转误差约 0.0067°，平移误差约 0.051 mm；这不代表实际扫描仪精度。短基准 2,000 对 8,000 点缓存查询，原生 1.05–2.10 ms，NumPy 穷举 276–431 ms；ICP 620 对 780 点原生（含构建）约 3.94–7.66 ms，NumPy 约 62–96 ms。不同轮次系统负载明显改变绝对时间，完整数据在 `projects/output/continuous_batch1/advanced_perception_final/benchmark.json` 及 `advanced_perception_preliminary/benchmark.json`。
