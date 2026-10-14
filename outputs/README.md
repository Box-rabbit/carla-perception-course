# Outputs

这里保存算法运行产物和可视化结果，不作为运行时配置或输入来源。

在线数据链路的推荐输出目录是：

```text
outputs/data_chain/<episode_id>/
├── perception.jsonl
└── truth.jsonl
```

现有 `lane_detection/`、`traffic_light/` 和 `stereo_3d/` 子目录是历史离线实验结果，保留用于复核。
