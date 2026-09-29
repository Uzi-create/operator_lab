# 相机画面位移估计

`operators.measurement_ops.estimate_scene_motion(reference, moving, *, max_analysis_width=480, min_response=.15, max_shift=None)` 接受同尺寸的 `[0,1]` 浮点灰度图，返回上一帧到当前帧的**画面像素位移**。`shift_xy=(dx,dy)` 中正 `dx` 表示画面内容向右移动；`alignment_matrix` 把当前帧平移回上一帧。返回的位移和矩阵以原始输入图的像素为单位，内部缩小分析图不改变单位。输入不会被修改。

```python
from operators.measurement_ops import estimate_scene_motion

result = estimate_scene_motion(previous_gray, current_gray, max_shift=(30, 30))
if result['success']:
    dx, dy = result['shift_xy']
    # 只有当前应用确实需要“画面位移”时，才使用这个方向。
else:
    print(result['reason'])
```

算法先将最长边压到 `max_analysis_width`，用零填充相位相关估计**非循环平移**，再检查重叠量、非循环重叠相关、重复纹理、循环别名和局部运动一致性。局部检查包含独立外缘区域和仿射残差趋势，可拒绝强运动前景或小角度旋转造成的部分误报。旧的 `estimate_translation` 保留周期边界约定，已有调用无需改动。`response` 是 OpenCV 的峰值响应，不是概率，也不能单独证明方向正确。

`success=False` 时不应使用候选 `shift_xy` 或画绿色方向箭头；`alignment_matrix` 为 `None`。常见拒绝原因包括 `insufficient_texture`、`low_response`、`shift_limit`、`insufficient_overlap`、`low_overlap_correlation`、`ambiguous_texture`、`ambiguous_wrap` 和 `inconsistent_motion`。`max_shift` 用于已知相邻帧最大速度的场合，单位仍是输入图像像素。极小图像可能因为证据不足而拒绝真实位移，这是有意的保守行为。

此算子估计的是二维图像内容变化，不是三维相机或机器人运动轨迹。静止背景、近似纯平移的情况下，相机横向移动和背景的画面位移方向相反；相机旋转、透视、景深视差、主体独立运动以及重复纹理，都可能使单一平移无意义。需要机器人轨迹时应结合相机标定、深度或多视图几何，并明确选择静态背景区域。实时程序在拒绝或缺少上一帧时不画箭头。

合成图专项验证运行 `python -B -m unittest operators.tests.test_scene_motion -v`；同机成对性能基准运行 `python -B projects/tools/benchmark_scene_motion.py`。这些测试不能替代实际相机现场验证。

2026-09-30 的 Windows/OpenCV 4.12.0 单线程成对基准中，1920×1080 合成平移 `(12,-7)` 的旧接口中位耗时为 47.53 ms，新接口为 25.04 ms，约快 1.90 倍；新接口该样本误差约 0.14 像素。240×320 小图因歧义和局部运动检查比旧接口更慢。基准只计算子调用，不计相机采集、灰度转换和绘图；详情及五种受控场景见 `projects/output/scene_motion_benchmark_20260930.json`（本地验证输出，通常不随源码发布）。另有 103 组独立合成对抗场景和 39 组额外探针；这些场景未出现错误方向的成功返回，但真实相机现场精度仍未验证。
