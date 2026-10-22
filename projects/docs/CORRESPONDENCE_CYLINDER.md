# 三维定位与圆柱测量

这组示例把两个新算子串联：从可能含有错误匹配的三维对应点恢复刚体位姿，将夹具已知的轴方向变换到目标坐标系，再测量目标点云中的圆柱半径与轴线。

```text
python -B projects/vision_robot/demo_correspondence_cylinder.py
python -B projects/benchmarks/benchmark_correspondence_cylinder.py --repeats 21
```

示例生成 1500 个点，其中包含 20% 几何杂点，再将 30% 的定位对应关系故意配错。输出包括两种投影的 PNG、HTML 预览和 JSON 真值对照。基准还覆盖 10000 个点，先检查多组已知真值，再对包含输入检查、RANSAC、精修和最终残差的完整 API 计时。

| 算子 | 输入和用途 | 必须注意 |
| --- | --- | --- |
| `estimate_rigid_transform_robust` | 两组已有对应关系的 Nx3 点，求 `T_target_from_source` | 不负责找到对应点；一致的错误匹配可能形成另一个几何可行解 |
| `fit_cylinder_known_axis` | Nx3 点和已知轴方向，测半径、轴线与支持跨度 | 轴方向由外部先验提供；跨度来自径向内点的轴向极值，不是端盖距离 |

详见[刚体定位接口](../../operators/docs/CORRESPONDENCE_POSE.md)和[圆柱测量接口](../../operators/docs/CYLINDER.md)。两者都必须先检查 `success`，拒绝时不能把候选几何作为有效测量使用。它们使用 NumPy 的底层矩阵运算和有界的 Python RANSAC 调度，不是新增的自写 C++ 动态库内核。

绿色点表示满足当前圆柱径向误差阈值，红色点表示被拒绝，青色线段表示支持点覆盖的轴段。杂点如果恰好落在同一个无限圆柱上，单靠径向距离就无法排除；它可能扩大角度覆盖和轴向跨度。示例因此核对最终内点的投影跨度，不把生成时设置的管长作为测得的端盖长度。

所有坐标使用相同长度单位；示例以米为单位。合成测试中极小的位姿或半径误差，只表示受控数据下的数值结果，不能当作真实相机、扫描仪或机器人的测量精度。实际应用还需要对应点质量、坐标标定、轴先验精度及传感器误差评估。

本批完整测试、独立审计和性能对照见[验证记录](GEOMETRY_VALIDATION.md)。
