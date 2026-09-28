# 组织化深度图连通区域

`depth_components(depth, depth_scale=..., min_depth=..., max_depth=..., absolute_jump=..., relative_jump=..., connectivity=4, min_area=..., backend='auto')` 按相邻像素的**米制或调用者指定单位**深度跳变建立图。当前像素 `za` 与相邻像素 `zb` 连接，当且仅当：

```text
abs(za-zb) <= absolute_jump + relative_jump * min(za, zb)
```

其中 `za = raw_depth * depth_scale`。`uint16` 毫米图输入时，可用 `depth_scale=.001` 转为米，全部深度范围及绝对阈值也应以米给出。相对阈值无量纲。`min_depth` 是严格下界，`max_depth` 是包含上界。NaN、Inf、零及范围外值为孔洞。可选 4 或 8 邻接，`min_area` 删除小连通区域。返回的 `labels` 与原图同尺寸，0 为无效或已过滤，正标签按首个像素的行优先顺序赋值；每一区域包含面积、`(x,y,width,height)` 外接框和深度最小/最大/均值。

```python
from operators.depth_components_ops import depth_components

result = depth_components(depth_mm, depth_scale=.001, max_depth=3.,
                          absolute_jump=.015, min_area=100, backend='native')
labels = result['labels']
regions = result['regions']
```

原生后端为 C++ 广度遍历；NumPy/Python 后端为独立参考，二者标签及统计经过多场景对照。无需 SciPy。`auto` 在本地 DLL 可用时选原生，否则退回参考；`native` 无法加载时明确报错。最多 1,600 万像素，输入为 2D `uint`、`int` 或浮点 NumPy 数组，不会改写输入。Python 参考在大图上较慢；生产批量处理建议构建原生库。

**传递连通**意味着一串逐步变化的深度可以把总深度差很大的两端连起来。这不是 3D 欧氏聚类、目标识别、表面法线估计或语义地面分割。相邻像素在深度上接近，也不保证它们的 3D 距离接近；阈值需按深度噪声和实际相机视场定标。背景可能环绕多个目标形成一个连通区域。

6 项专项测试包含边界等号、4/8 对角连接、孔洞、小区域过滤、毫米/米换算、平滑坡传递、随机非连续视图与原生/参考严格标签一致。独立合成三深度场景在 640×480、1% 孔洞和孤立噪声下，303,938 个保留像素标签与生成真值全部一致；这只验证该合成模型。7 次交替短基准，640×480 原生全 API 中位 **6.43 ms**；160×120 原生 **0.397 ms**、参考 **38.44 ms**。原始采样见 `projects/output/continuous_batch2/depth_benchmark/benchmark.json`；真实性能随硬件、区域数量和噪声变化。
