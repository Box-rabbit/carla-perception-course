# GitHub 上传内容说明

本项目包含课程设计中的 CARLA 场景侧配置/实现、感知运行时、传感器契约、`front_rgb` 标定验证、YOLOv5 接口、独立 GPU 推理服务和阶段验收文档。

不上传以下本地内容：

- CARLA 安装包、压缩包和本地安装路径；
- `outputs/data_chain`、`outputs/midterm_media` 等运行输出；
- Python 缓存、虚拟环境和日志；
- 真值日志中的大批量运行数据。

YOLOv5 和 UFLD 保留为当前工作区的上游依赖目录；根项目代码通过 `PYTHONPATH` 使用它们。新项目首次部署时，应按照各自目录 README 和 `requirements` 安装依赖。
