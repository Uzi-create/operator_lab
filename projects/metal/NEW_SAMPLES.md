> 2026-09-26 目录已整理。本文保留原算法说明与历史测量；旧命令/导入方式请按[当前目录指南](../../README.md)使用，历史清单不代表现行布局。

# 新增五张照片：固定参数对照

接收到五张旋转姿态、木纹背景、照明不同的新照片后，使用现有 v1/v2 参数运行。
仅重新人工圈定金属 ROI；没有画缺陷真值框，没有逐图调整检测阈值，没有训练模型。
ROI 不是自动分割结果。照片副本及 SHA-256 记录保存在 samples/ 中，原图未修改。

| 样本序号 | v1 内部候选 | v2 内部候选 | 边缘待复核 | 色斑候选 |
|---|---:|---:|---:|---:|
| 1 | 2 | 2 | 9 | 0 |
| 2 | 4 | 4 | 5 | 2 |
| 3 | 5 | 5 | 9 | 2 |
| 4 | 8 | 6 | 13 | 0 |
| 5 | 2 | 2 | 9 | 2 |

候选数不等于缺陷数。可视检查显示主要划伤附近有响应，但不能在没有真值时统计召回率。
边缘待复核结果仍包含轮廓反光；色斑可能来自照明，不代表氧化确认。
第 4 张有两个内部候选被过滤，其余内部数量未变，因此没有证据表明 v2 全面优于 v1。
这批看起来仍是同一或相似工件的多个角度，不能作为五个独立产品的泛化验证。

v2 本轮约 51–57 ms/张，长边 960 工作分辨率，预热后交替顺序 5 次中位数，
不含读图/写文件。具体参数、原图坐标框和计时保存在每张 JSON。

## 重跑

在项目根目录：

```bash
bash operator_lab/metal/run-metal.command --compare \
  --manifest operator_lab/metal/new_sample_rois.json \
  --output operator_lab/output/metal_new_samples

bash operator_lab/metal/run-metal.command --diagnostics \
  --manifest operator_lab/metal/new_sample_rois.json \
  --output operator_lab/output/metal_new_diagnostics
```

新旧版本对照：../output/metal_new_samples/index.html。
局部对比度/片段分组：../output/metal_new_diagnostics/index.html。
人工 ROI：new_sample_rois.json，序号与收到照片的文件顺序一致。

此次只新增样本、ROI 与结果记录，未改算法代码；此前 59 项代码测试结果不等同于这批照片的检测准确率。
