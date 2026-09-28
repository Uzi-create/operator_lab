# 无序点云的体素图连通聚类

`voxel_clusters(points, voxel_size, connectivity=26, min_voxels=1, min_points=3, backend='auto')` 将每个点映射到 `floor(point / voxel_size)` 的整数网格，先合并同格点，再沿占用体素的 6/18/26 邻接图找连通区域。`labels` 与输入点逐一对应；过滤的区域标签为 0。`voxel_keys` 是按 x/y/z 字典序排列的唯一占用体素，`voxel_labels` 是其区域标签。区域元数据包含点数、体素数、点坐标中心和实测轴对齐范围。

```python
from operators.voxel_cluster_ops import voxel_clusters

result = voxel_clusters(cloud_m, .02, connectivity=26,
                        min_voxels=4, min_points=50, backend='native')
for cluster in result['clusters']:
    object_points = cloud_m[result['labels'] == cluster['label']]
```

坐标和体素边长同单位。区域标签由每个区域字典序最小的占用体素决定，原生 C++ 与独立 Python 参考给出相同点/体素标签。`min_voxels` 和 `min_points` 均为明确的过滤条件；不会偷偷抽样点。`auto` 在 DLL 不可用时退回参考，`native` 会明确报错。原生连接使用哈希查找加并查集，Python 使用广度遍历；共同的 NumPy `unique` 做体素量化映射。上限 200 万输入点与 10 万输出区域，以限制内存消耗。

**这不是欧氏半径聚类。** 同一或相邻体素里的点可能相距超过 `voxel_size`；体素链还可能跨越很长距离。6 邻接只接面，18 邻接加边，26 邻接再加角。负坐标使用 `floor` 而不是向零截断。点云遮挡、跨物体接触、噪声桥和体素边界精度可能改变连通关系；`min_points` 也不能证明某个区域是物体。

6 项专项测试覆盖三种邻接、传递链、负坐标、区域过滤、随机非连续输入、原生/参考精确标签等价和无效 ABI。三个合成高斯目标、500 个均匀噪声点：目标召回分别 99.983% / 100% / 99.983%，31 个噪声点因靠近目标而被接纳；质心误差分别 3.62 / 2.77 / 1.21 mm。11 次短基准完整 API：18,500 点原生中位 **13.06 ms**；2,180 点原生 **1.82 ms**、Python 参考 **9.38 ms**。四种子真值检查及小场景原生/参考对照通过。数据在 `projects/output/continuous_batch8/voxel_benchmark/benchmark.json`；这些数字不代表实际激光雷达或深度相机的物理精度。
