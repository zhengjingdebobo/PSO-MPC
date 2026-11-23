# 基于 PSO-MPC 的自动泊车算法设计与仿真：第一阶段

本阶段仅实现五状态自行车模型、水平泊车环境、矩形碰撞检测、固定输入仿真与可视化。**没有实现 PSO、MPC、代价函数、自动泊车策略、强化学习或路径规划。** 倒车转弯演示不会驶入车位，也不代表已完成自动泊车。

## 环境与运行

Windows / Python 3.12.10。使用根目录已有 `.venv`，不要重新创建环境，也不要安装到全局 Python。在项目根目录的 PowerShell 执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m pytest -v --junitxml=outputs/pytest_results.xml
```

可选运行方式：

```powershell
# 显示图窗并播放动画；关闭图窗退出
.\.venv\Scripts\python.exe main.py --show
# 无图窗运行，只保存静态图片和数值数据
.\.venv\Scripts\python.exe -m pso_mpc_parking.main --no-animation
# 欧拉积分对照，独立输出目录
.\.venv\Scripts\python.exe main.py --method euler --output outputs_euler
```

默认使用 Agg 后端保存文件，适合 VS Code 终端及无图窗测试。GIF 由 Pillow 编码，无需 FFmpeg；Pillow 是轻量图像依赖。支持 GIF 保存与 Matplotlib 动画播放，本阶段不提供 MP4 编码。图中的英文标签避免依赖额外中文字体；所有内部角度使用弧度，图表以度显示并标注单位。

## 实际文件结构与模块职责

```text
code/
├── .venv/                        # 原有虚拟环境
├── parking.py                    # 保留的原有空文件
├── main.py                       # 主程序入口
├── requirements.txt
├── requirements-lock.txt         # 当前实际验证过的依赖版本
├── pytest.ini
├── .gitignore
├── README.md
├── pso_mpc_parking/
│   ├── __init__.py
│   ├── main.py                   # 固定输入演示与导出
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # 冻结 dataclass 参数及校验
│   ├── models/
│   │   ├── __init__.py
│   │   └── vehicle_model.py      # 纯预测与可变真实车辆状态
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── geometry.py           # 旋转矩形、SAT、欧氏间距
│   │   ├── parking_env.py        # 可行驶矩形、车位、静态车辆、引导接口
│   │   └── collision.py          # 单状态/插值轨迹碰撞报告
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── simulator.py          # 推进时间、执行外部输入、记录历史
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py              # 只读绘图和 GIF 导出
│   ├── controllers/__init__.py   # 仅扩展占位
│   └── planning/__init__.py      # 仅扩展占位
├── tests/
│   ├── test_vehicle.py
│   ├── test_collision.py
│   ├── test_environment.py
│   ├── test_simulation.py
│   ├── test_visualization.py
│   └── conftest.py               # 每次运行独立的项目内临时目录
└── outputs/
    ├── parking_environment.png   # 环境、最终车身、初始轮廓和轨迹
    ├── reverse_turn.gif          # 6 秒固定输入倒车转弯
    ├── state_curves.png          # 状态及输入曲线
    ├── simulation_data.npz       # times / states / controls
    ├── simulation_summary.json   # 碰撞、间距及最终状态
    └── pytest_results.xml        # pytest 实际执行报告
```

依赖层次由基础到上层为 `config → models → environment → simulation → visualization`；每层只导入它左侧的基础模块。模型不导入环境、仿真或绘图。几何和碰撞不导入 Matplotlib。主程序负责组装。`controllers/`、`planning/` 只含占位说明。

## 坐标、状态与参数

全局 x 沿道路正方向，y 垂直道路向上。`theta=0` 时车头朝 +x；航向角逆时针为正，前轮向车辆左侧转动时 `delta>0`。`v<0` 表示倒车，因此同一个正转角在前进时使航向角增加，在倒车时使航向角减少。

状态 `X=[x,y,theta,v,delta]`，控制 `u=[a,omega]`。x/y 始终是**后轴中心**，单位分别为 m、rad、m/s、rad、m/s²、rad/s。严格使用：

```text
dx/dt     = v*cos(theta)
dy/dt     = v*sin(theta)
dtheta/dt = v/L*tan(delta)
dv/dt     = a
ddelta/dt = omega
```

`config/settings.py` 集中管理物理及环境参数：车长 4.5 m、车宽 1.8 m、轴距 2.7 m，另假设前后悬均为 0.9 m（可改，需满足总长关系）；最大转角 35°，最大转角速度 30°/s，速度范围 [-1.5,2.0] m/s，加速度范围 [-1,1] m/s²，步长 0.05 s。这些都是初始仿真假设，并非声称课程指定值。

车身局部纵向范围为 `[-rear_overhang, wheelbase+front_overhang]`，横向为 `[-width/2,width/2]`。旋转和平移后得到实际四角，几何中心相对后轴向前偏移 1.35 m。目标位中心 `(0,-1.25)`，范围 x=[-3.5,3.5]、y=[-2.5,0]；目标后轴参考状态为 `[-1.35,-1.25,0,0,0]`。可行驶区域 x=[-12,18]、y=[-3.2,6]，包含道路及停车带；y=0 的虚线是道路与停车带的视觉分界，车位入口允许穿越，不是碰撞边界。前后障碍车中心 x=±5.85、y=-1.25。

默认初态 `[6,2,0,-0.8,deg2rad(20)]`，持续 6 s 输入 `[0,0]`。车辆沿恒曲率圆弧倒车；初态和整个演示均由碰撞检查验证。环境参数 dataclass 可传入构造器覆盖，车辆初态和仿真步长在 SimulationConfig 中配置。

## 积分与约束

默认经典四阶 Runge–Kutta（RK4），可选择显式欧拉。每步控制零阶保持。先限幅输入；初态速度/转角限幅、航向角归一化到 [-pi,pi)。碰到速度或转角边界时禁止继续向外增长，反向输入可以立即离开边界。积分区间会在达到限幅的准确时刻分段，避免整步仅裁剪末态导致位置或转角积分错误。每段 v/delta 线性更新，位姿由选定积分法计算。

位置不被强行裁剪到道路内，越界由碰撞模块报告。记录的 controls 是限幅后的**请求输入**；速度/转角达到边界后，实际状态导数可能为 0，不能将请求输入误解为饱和后仍实际施加的状态变化率。

## 碰撞检测与安全距离

`geometry.py` 用 SAT 检查两个旋转凸矩形：将所有顶点投影到两矩形各条边的法向，任一轴分离即不碰撞；接触、角点接触和重叠均视为真实碰撞（数值容差 1e-9 m）。可行驶边界为闭合矩形，所有车身角点必须严格在内部；触边也视为碰撞。

安全距离单独用凸多边形之间的最短**欧氏距离**及车身至边界的最小间距计算，默认 0.2 m。未碰撞但间距小于阈值时 `collision=False, safety_violation=True`。间距恰等于阈值不违规。碰撞时 clearance 为 0；安全距离设为 0 可禁用间距判据，但不会取消真实碰撞。结果 `safe` 要求两项都不发生。

`check_trajectory()` 在相邻状态间按“平移距离 + 后轴至最远角点半径 × 航向变化绝对值”估算角点运动量，默认每 0.05 m 插值检查，航向取最短角差。返回所有采样的最小间距、首次碰撞所在区间及区间比例。不会只检查两端点。

**处理边界与局限：**这仍是离散插值，不能证明连续扫掠体绝对无碰撞；极薄障碍、极小擦碰可能漏检。长步长两端之间的线性插值也不等于真实曲线运动。应保持细时间步长，并按障碍尺度降低 collision_resolution。相邻真实航向变化必须小于 pi，否则仅凭归一化端点无法恢复多圈转动。当前只有矩形可行驶区域和静态凸矩形障碍，没有非凸区域、动态障碍或真实车辆动力学。安全阈值不是车身膨胀后的真实碰撞。

仿真默认发现真实碰撞后停止；检测到跨步碰撞会记录该步末态并终止，不回滚到精确接触时刻。安全距离不足只报告，不自动改变输入。该行为不是避障或停车控制策略。

## 后续接入接口

```python
import numpy as np
from pso_mpc_parking.models import VehicleModel
from pso_mpc_parking.environment import CollisionChecker, ParkingEnvironment

model = VehicleModel()
x0 = np.array([6., 2., 0., -0.8, np.deg2rad(20.)])
controls = np.zeros((40, 2))
next_state = model.predict_step(x0, controls[0], dt=0.05)
trajectory = model.predict_trajectory(x0, controls, dt=0.05)  # (41,5)
checker = CollisionChecker(ParkingEnvironment(), model.config)
state_report = checker.check_state(next_state)
trajectory_report = checker.check_trajectory(trajectory)
```

- `predict_step(initial_state, control, dt, method='rk4') -> (5,)`：无副作用单步预测。
- `predict_trajectory(initial_state, controls, dt, method='rk4') -> (N+1,5)`：无副作用完整预测，任意初态，包含初始状态，不更改输入数组或真实车辆状态。
- `state`：读取真实状态副本；`reset(state)` 和 `step(control,dt,method)` 明确改变真实状态。
- `vehicle_vertices(state, vehicle_config) -> (4,2)`：纯几何车身轮廓。
- `check_state(...) -> CollisionResult`、`check_trajectory(...) -> TrajectoryCollisionResult`：与绘图分离的几何查询。
- `ParkingEnvironment.set_guide_pose([rear_x,rear_y,theta] | None)`：仅存储外部引导姿态，不计算引导或路径。
- `Simulator.run(controls, initial_state=None, stop_on_collision=True)`：外部输入序列仿真与记录。
- `plot_environment`、`plot_state_curves`、`animate_simulation`：只读结果，输出图像或动画。

预测循环不执行绘图、碰撞查询或文件操作。接口统一使用 NumPy，可由未来优化模块反复调用；本阶段是单条轨迹标量循环，尚未实现批量粒子预测、向量化内核或性能承诺。后续可在保持 `(N+1,5)` 接口行为的基础上增加 `(B,N+1,5)` 批量接口，不必耦合绘图。

## 测试范围

Windows 权限兼容：测试默认在项目根目录下创建本次运行专用的 `.pytest-tmp-*` 临时目录，运行结束后自动清理；不再访问系统临时目录下可能权限不兼容的 `pytest-of-Bobo`。同时禁用 pytest 缓存插件，避免旧 `.pytest_cache` 的访问权限影响运行。这不影响测试数量、断言或 JUnit 报告，也不需要管理员权限、删除旧目录或修改系统 ACL。显式传入 `--basetemp` 时保留用户指定路径。

pytest 覆盖静止、正负速度直线运动、正负转角前进/倒车方向和航向变化率、原地转方向盘、状态/输入限幅、触及速度上限时的位移、恒速恒转角解析圆弧、RK4/欧拉小步长比较、无副作用与状态复制、参数错误、几何车身参考点、SAT 分离/接触/重叠/旋转、欧氏安全距离、旋转越界、跨步碰撞与角度插值、场景可容纳性、仿真记录/停止逻辑、绘图只读及 GIF 导出。测试的真实通过/失败情况以 `outputs/pytest_results.xml` 和最终交付记录为准。

2026-11-19 使用现有 `.venv` 实测：**47 passed，0 failed，0 skipped**。Python 3.12.10，NumPy 2.5.5，Matplotlib 3.11.2，pytest 9.2.0，Pillow 12.3.0；`pip check` 通过。版本快照见 `requirements-lock.txt`（可用 `-m pip install -r requirements-lock.txt` 复现）。

默认 RK4 演示实际执行 120 步 / 6.0 s，未发生真实碰撞或安全距离违规；插值检查得到最小间距约 0.7044 m。最终后轴状态约为 `[1.5280,3.4995,-0.64706,-0.8,0.34907]`，其中航向角约 -37.07°。静态俯视图、状态曲线及 GIF 代表帧已进行视觉检查，车身按实际旋转轮廓绘制，坐标等比例。
