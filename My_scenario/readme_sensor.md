> 本文是场景侧传感器接入参考。当前仓库的正式感知入口是根目录的
> `perception_runtime`，公共时钟、传感器安装和帧契约分别由 `config/system.yaml`、
> `config/sensors.yaml` 和 `common/contracts.py` 管理。文中的 Windows 路径和旧版
> ScenarioRunner 命令仅作概念参考，不覆盖当前 Linux 运行方式，也不改变真值与感知输入隔离。

感知模块接入 README

本说明面向负责传感器挂载、YOLO 检测（红绿灯 / 行人 / 车辆）和真值对比评估的同学。

当前项目使用 CARLA ScenarioRunner 的 --route 模式运行场景，场景定义在 My_scenario/routes/my_scenario_test.xml，自车由 route_follower_agent.py 控制。感知同学需要在这个 Agent 里接入传感器、读取真值、跑 YOLO。

一、你需要在哪个文件夹工作

E:\CARLA\scenario_runner\srunner\autoagents\route_follower_agent.py
你的主要工作文件，在这里挂传感器、读数据、跑 YOLO。

E:\CARLA\My_scenario\routes\my_scenario_test.xml
场景定义（路径 + 触发点），一般不需要改。

E:\CARLA\scenario_runner\srunner\autoagents\sensor_interface.py
框架代码，不用改，直接用。

你只需要改 route_follower_agent.py。

二、运行方式

cd E:\CARLA\scenario_runner
python scenario_runner.py --route E:\CARLA\My_scenario\routes\my_scenario_test.xml --reloadWorld --agent E:\CARLA\scenario_runner\srunner\autoagents\route_follower_agent.py --sync --frameRate 20

注意：CARLA 服务器建议用 -benchmark -fps=20 启动，保证画面速度正常。

三、传感器挂载

3.1 挂载位置

在 route_follower_agent.py 的 sensors() 方法里填传感器配置，ScenarioRunner 会自动挂载。

3.2 当前接口

def sensors(self):
"""传感器接口：感知同学在这里填传感器配置"""
return []

3.3 填传感器配置

def sensors(self):
return [
{
'type': 'sensor.camera.rgb',
'x': 0.7, 'y': 0.0, 'z': 1.60,
'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
'width': 800, 'height': 600, 'fov': 100,
'id': 'Center'
},
{
'type': 'sensor.lidar.ray_cast',
'x': 0.0, 'y': 0.0, 'z': 2.5,
'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
'channels': 32, 'range': 100,
'points_per_second': 56000,
'rotation_frequency': 20,
'upper_fov': 10.0, 'lower_fov': -30.0,
'id': 'LIDAR'
}
]

3.4 参数说明

type：sensor.camera.rgb 或 sensor.lidar.ray_cast
x, y, z：相对 ego 的位置（米）
roll, pitch, yaw：相对 ego 的朝向（度）
width, height：摄像头分辨率
fov：摄像头视场角
id：唯一标识，读数据时用
channels：LiDAR 通道数
range：LiDAR 探测距离
points_per_second：LiDAR 每秒点数

四、在哪里读传感器数据

在 run_step() 里通过 self.sensor_data 字典读取。键是 sensors() 里的 id。

4.1 数据格式

SensorInterface（框架代码）已经把数据解析成 numpy 数组：

摄像头：numpy array，形状 (H, W, 4)，BGRA
LiDAR：numpy array，形状 (N, 4)，x/y/z/intensity
Radar：numpy array，形状 (N, 4)
GNSS：numpy array，形状 (3,)
IMU：numpy array，形状 (7,)

4.2 读摄像头数据

def run_step(self):
if 'Center' in self.sensor_data:
frame, image_array = self.sensor_data['Center']

image_array 已经是 (H, W, 4) 的 BGRA numpy 数组
bgr = image_array[:, :, :3] # 去掉 alpha 通道

喂给 YOLO
detections = yolo_model(bgr)
...

4.3 读 LiDAR 数据

if 'LIDAR' in self.sensor_data:
frame, points = self.sensor_data['LIDAR']

points 是 (N, 4) 的 numpy 数组
五、在哪里读真值

真值通过 CARLA API 直接获取，不需要场景模块提供。

5.1 获取 ego 车

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
ego = CarlaDataProvider.get_hero_actor()

5.2 获取场景 NPC 车辆

world = self._world
all_vehicles = world.get_actors().filter('vehicle.*')

排除 ego 和背景车，只留场景 NPC
scenario_vehicles = [
v for v in all_vehicles
if v.attributes.get('role_name', '') not in ('hero', 'background')
]

5.3 获取行人

pedestrians = world.get_actors().filter('walker.pedestrian.*')

5.4 获取红绿灯

traffic_lights = world.get_actors().filter('traffic.traffic_light')

for tl in traffic_lights:
state = tl.get_state() # Red / Yellow / Green / Off
location = tl.get_location()

5.5 actor 通用接口

get_location()：carla.Location，位置
get_velocity()：carla.Vector3D，速度
get_transform()：carla.Transform，朝向
attributes['role_name']：角色名（hero / background / 其他）

5.6 完整真值读取示例

def run_step(self):
world = self._world
ego = CarlaDataProvider.get_hero_actor()

场景 NPC 车辆（排除 ego 和背景车）
all_vehicles = world.get_actors().filter('vehicle.*')
scenario_vehicles = [
v for v in all_vehicles
if v.attributes.get('role_name', '') not in ('hero', 'background')
]

行人
pedestrians = world.get_actors().filter('walker.pedestrian.*')

红绿灯
traffic_lights = world.get_actors().filter('traffic.traffic_light')

打印真值
for v in scenario_vehicles:
dist = ego.get_location().distance(v.get_location())
print(f"[GT] 场景车辆 id={v.id} dist={dist:.1f}m")

for p in pedestrians:
dist = ego.get_location().distance(p.get_location())
print(f"[GT] 行人 id={p.id} dist={dist:.1f}m")

for tl in traffic_lights:
dist = ego.get_location().distance(tl.get_location())
print(f"[GT] 红绿灯 id={tl.id} state={tl.get_state()} dist={dist:.1f}m")
...

六、YOLO 检测接入

6.1 检测目标

红绿灯：输入 RGB 图像，输出位置 + 状态，真值来源 traffic.traffic_light actor 的 get_state()
行人：输入 RGB 图像，输出边界框，真值来源 walker.pedestrian.* actor
车辆：输入 RGB 图像，输出边界框，真值来源 vehicle.* actor（排除 hero 和 background）

6.2 对接示例

def run_step(self):

1. 拿传感器数据
if 'Center' in self.sensor_data:
frame, image_array = self.sensor_data['Center']
bgr = image_array[:, :, :3]

2. YOLO 检测
detections = yolo_model(bgr)

3. 拿真值
world = self._world
ego = CarlaDataProvider.get_hero_actor()
gt_vehicles = [
v for v in world.get_actors().filter('vehicle.')
if v.attributes.get('role_name', '') not in ('hero', 'background')
]
gt_pedestrians = world.get_actors().filter('walker.pedestrian.')
gt_lights = world.get_actors().filter('traffic.traffic_light')

4. 对比评估
evaluate(detections, gt_vehicles, gt_pedestrians, gt_lights)

5. 控制 ego（沿路径走）
...

6.3 建议的评估指标

检测率：真值目标中被检测到的比例
误检率：检测结果中假阳性的比例
红绿灯状态准确率：红绿灯状态识别正确的比例
距离误差：检测目标距离与真值的偏差

七、数据流（框架已处理，不用自己实现）

sensors() 定义传感器
↓
ScenarioRunner 自动挂载，并为每个传感器注册 CallBack
↓
传感器每帧回调 → CallBack 解析成 numpy
↓
SensorInterface 存入队列
↓
agent_wrapper.py 每帧调用 get_data()
↓
数据通过 self.sensor_data 传给 run_step()

sensor_interface.py 是框架代码，不用改。

八、注意事项

只改 route_follower_agent.py，不要改 XML 和框架代码。

图像格式是 BGRA，用 image_array[:, :, :3] 去掉 alpha 通道。

排除背景车：role_name 为 background 的是背景交通车，评估时要过滤掉。

同步模式：用 --sync --frameRate 20，保证传感器数据和场景状态同步。

红绿灯真值：用 traffic.traffic_light actor 的 get_state()。

场景 NPC 识别：通过 role_name 区分场景 NPC 和背景车。

CARLA 服务器：建议用 -benchmark -fps=20 启动，避免画面过快。
