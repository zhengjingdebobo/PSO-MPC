# 开发记录
## 由ai生成基本仿真代码流程 
具体可见readme 需要关注的有：
### 根目录main.py为入口函数 实际流程位于 /pso_mpc_parking/main.py
#### 核心流程
1. 读取车辆、场景和仿真配置
vehicle, scene = VehicleConfig(), EnvironmentConfig()
config = SimulationConfig(integration_method=args.method)

2. 创建车辆及环境
environment = ParkingEnvironment(scene, vehicle)
model = VehicleModel(vehicle, config.initial_state)
checker = CollisionChecker(environment, vehicle)

3. 设置输入序列
controls = np.zeros((count, 2))

4. 执行仿真
result = Simulator(model, checker, config).run(controls)

### 车辆接口三个函数，位于/pso_mpc_parking/models/vehicle_model.py

| 接口 | 输入 | 返回 | 是否改变真实车辆 |
|---|---|---|---|
| `predict_step()` | 初始状态、一个控制、步长 | 下一状态 `(5,)` | 否 |
| `predict_trajectory()` | 初始状态、控制序列、步长 | 完整轨迹 `(N+1,5)` | 否 |
| `step()` | 一个控制、步长 | 更新后的状态 `(5,)` | **是** |

### 需要自建以及攥写的文件代码结构框架
pso_mpc_parking/
├── config/
│   └── settings.py          # 在已有文件中增加算法参数配置
├── controllers/
│   ├── objective.py         # 报告中的代价函数
│   ├── evaluator.py         # 轨迹预测、约束检查及候选解评价
│   ├── pso.py               # 粒子群搜索
│   └── mpc.py               # 调用 PSO，返回当前应执行的输入
├── simulation/
│   ├── simulator.py         # 保留已有的预设输入仿真
│   └── closed_loop.py       # 闭环推进、记录及终止判断
└── main_pso_mpc.py           # 自动泊车实验入口

## 11.23
新建controllers/objective.py与tests/test_objective.py 主要负责实现代价函数与测试