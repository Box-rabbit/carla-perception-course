# 场景侧感知评估目标修改要求

## 目的

为 `front_rgb + YOLOv5s` 提供可重复、可评估的车辆和行人目标。场景真值只写入 `truth.jsonl`，不得作为 YOLO 输入。

## 必须修改

### 1. 增加专用评估目标配置

在场景配置中增加独立的 `evaluation_targets` 段，不与正式交通参与者混用：

```yaml
evaluation_targets:
  enabled: true
  actors:
    - id: eval_vehicle_01
      type: vehicle
      model: vehicle.tesla.model3
      distance_ahead_m: 18.0
      lateral_offset_m: 0.0
      speed_kmh: 0.0
      visible_from_frame: 0
      visible_until_frame: 199
    - id: eval_pedestrian_01
      type: pedestrian
      model: walker.pedestrian.0001
      distance_ahead_m: 20.0
      lateral_offset_m: 2.0
      speed_mps: 1.0
      visible_from_frame: 40
      visible_until_frame: 199
```

字段要求：

- `id` 唯一且稳定；
- `type` 仅使用 `vehicle` 或 `pedestrian`；
- `model` 不可用时提供同类 fallback blueprint；
- 位置使用相对 ego 的前向距离和横向偏移；
- 出现帧范围明确；
- 速度、朝向和触发条件可重复。

### 2. 保证目标可见

- 目标必须位于 `front_rgb` 光轴前方和有效视场内；
- 初始距离建议 15～25 m，保证目标在图像中有足够像素；
- 不得被建筑、路牌、其他车辆或地形完全遮挡；
- 车辆与行人应分别提供无遮挡样本；
- 目标不能与 ego 重叠或生成位置冲突；
- 场景启动后先完成一次 `world.tick()`，再依据 ego 实际位姿计算目标位置。

### 3. 保证真值完整

每个评估目标在 `truth.jsonl` 对应帧中必须包含：

- 稳定 `actor_id`；
- `type_id`；
- 世界坐标位置和姿态；
- 世界坐标包围盒 8 个角点；
- 当前帧号和仿真时间戳。

ego 车辆必须从检测指标真值中排除。

### 4. 提供固定测试用例

至少实现以下 4 个可重复用例：

1. 18 m 正前方静止车辆；
2. 30 m 正前方远距离车辆；
3. 20 m 侧前方行人；
4. 车辆和行人同时出现。

每个用例至少运行 100 帧，并保持目标位置、类别和出现时机稳定。

### 5. 提供遮挡用例

在基础无遮挡用例通过后，再增加：

- 部分遮挡车辆；
- 部分遮挡行人；
- 目标进入视野和离开视野的过渡帧。

遮挡用例不能替代无遮挡基线用例。

## 交付验收

场景侧交付以下内容：

- 修改后的场景 YAML；
- 评估目标生成和清理代码；
- 每个目标的稳定配置 ID；
- 一次运行产生的 `truth.jsonl`；
- 目标在 `front_rgb` 原图中的人工可见截图；
- 目标帧范围和预期类别说明。

## 禁止事项

- 不把 actor 真值、类别或包围盒传给 YOLO；
- 不修改 `perception.jsonl` 来伪造检测结果；
- 不只生成目标而不验证其在相机图像中可见；
- 不用随机位置替代固定评估配置。
