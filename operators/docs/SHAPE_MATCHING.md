# 旋转与尺度形状匹配

入口 `create_shape_template(image, angles=..., scales=...)` 建立可复用模型，`match_shape(scene, model, ...)` 在场景中定位一个或多个实例。输入是 `[0,1]` 的有限 `float32/float64` 单通道图像。正角遵循 OpenCV：屏幕视觉上的逆时针。`center_xy`、`corners_xy` 与 `matrix` 均为原图像像素坐标，矩阵从模板坐标映射到场景坐标。

```python
from operators.shape_ops import create_shape_template, match_shape

model = create_shape_template(template, angles=(-20, 0, 25), scales=(0.9, 1, 1.1))
result = match_shape(scene, model, max_matches=2)
if result['success']:
    for match in result['matches']:
        print(match['center_xy'], match['angle_degrees'], match['scale'])
```

模板缓存全部 Canny 边缘点和每个离散姿态的相关核，不抽样边缘点。场景建立精确欧氏距离场；相关核给出整数平移候选。提案阶段先截断距离再插值，最终验收先插值再截断，二者在截断边界有差异；候选分数只是筛选代理。每一个候选最终都用原分辨率、**全部**模板边缘点核算支持比例、均值和 RMS，再做 0.5、0.25、0.125 像素平移精化。亚像素批量计算与逐点参考实现有相同结果。保守的距离场梯度上界仅跳过即使精化也不可能满足均值门限的候选；它不放宽通过条件。

默认 `coarse_step=1` 搜索每个姿态的整数平移图。`coarse_step>1` 明确用速度换召回。`candidates_per_pose` 限制每个姿态复核的局部极小值个数；`candidate_budget_hit=True` 表示候选被截断，第二件目标可能因此漏检。扩大预算/ROI 或姿态网格会增加计算量。角度和尺度只取传入的离散值，**仅平移**做亚像素精化。所有模板边缘必须完整在场景内，遮挡只通过 `min_support` 容忍；透视变形、对称轮廓和密集干扰可能产生歧义。`support_fraction`、`mean_distance_px` 等分数不是概率，也不证明唯一匹配。

输入无纹理目标可用此法，纹理丰富且允许透视时优先考虑 `feature_ops` 的 ORB 平面匹配。联合示例：`python projects/vision_robot/demo_advanced_perception.py`。19 项专项测试覆盖 ±90° 角度、双目标、候选预算漏检、截断语义、退化轮廓、完整边缘复核与标量参考结果一致性。合成双目标示例中心误差约 0.40/0.56 px；它不代表实际工件精度。9 姿态、360×260 两目标的完整 API 在本机负载不同的两次短采样约 59–131 ms（每姿态 8 候选），默认 24 候选约 71–157 ms，不能称为实时保证。数据见 `projects/output/continuous_batch1/advanced_perception_final/benchmark.json`，保留先前短样本作对照。

第 6 批进一步把同一姿态的初始候选和亚像素邻居分块批量核验；分块只限制临时数组大小，不减少边缘、候选或搜索邻居。保留的旧实现在 `projects/benchmarks/baselines/shape_before_batching.py`，16 个固定种子/参数组合的**所有输出字段逐项完全一致**。同一进程交替计时、21 次采样，默认 24 候选旧实现中位 50.94 ms，新实现 41.32 ms，约 **1.23 倍**；原始报告见 `projects/output/continuous_batch6/shape_comparison_durable/benchmark.json`。机器负载不同于上述旧测，勿拿不同轮次的绝对值直接对比。
