# 开发记录
## 由ai生成基本仿真代码流程 我负责攥写核心控制代码 即controllers下所有文件
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
│   └── settings.py          # 在已有文件中增加算法参数配置 已完善√
├── controllers/
│   ├── objective.py         # 报告中的代价函数 已完成√
│   ├── evaluator.py         # 轨迹预测、约束检查及候选解评价 已完成√
│   ├── pso.py               # 粒子群搜索   已完成√
│   └── mpc.py               # 调用 PSO，返回当前应执行的输入   已完成√
├── simulation/
│   ├── simulator.py         # 保留已有的预设输入仿真
│   └── closed_loop.py       # 闭环推进、记录及终止判断
└── main_pso_mpc.py           # 自动泊车实验入口

## 11.23
上传至Github
新建controllers/objective.py与tests/test_objective.py 主要负责实现代价函数与测试
完成代价函数模块的编写
### learning
学习到了pytest，python测试工具，可以通过这个来测试所编写的模块函数是否正确

## 11.24
完成评价候选序列模块的编写 evaluater.py 主要复制对生成的序列进行评价
发现对python还是有很多不熟练的地方 python相比与c确实在测试上有着很多不一样的小功能
编写代码习惯还有待改进 工程化模块化还不够 而且有很多防御性编程 我写的都是直接乘在一起就完事了，管他这维度那维度的
现在写的是爽，但是到时候调试确实错在哪里都不知道 好习惯要慢慢养成
工作日一天一个模块 周末来调参，应该就可以了

## 11.25
写pso相关代码  满课and出去玩了一下，只写了配置文件 明天写吧

## 11.26
完成pso相关代码 发现还是有很多想不到的地方，知道算法但是不知道代码的变量应该怎么添加
而且在写代码的时候总是会粗心，变量名搞错
python和c的差别还是有的，在引用函数的时候一直漏掉最后一个
模块化模块化 感觉现在只是把框架写出来了，到时候调参估计也是个大工作

开始写mpc相关代码了，搞了好久Gemini的会员，结果还是没搞定，账号一直是中国香港，明天GPT就过期了！！！我也想用30的ai！！
速写，快写

已经完成各个模块的编写：objective实现代价函数计算，evaluate评价生成序列，pso进行随机粒子群生成，mpc调用最优序列并进行第一步，接下来就是整合循环执行接入仿真了 以为工作量很大的，但是每天写一点其实还好
ai确实比我厉害多了，我还会有拼写错误，它直接写出来了 我30分钟的工作量它30s

## 11.28
昨天出去玩了 我了个豆
今天开始写完了整个闭环的系统，能够运行了，但是一直没有办法将车子停进去
不知道是因为参数的问题还是什么，我感觉通过调参应该可以完成整个系统的，因为理论上是能够实现的
但是他只有后退，没有前进，这会不会是停不进去的原因 这个明天看看能不能改掉
还有就是gpt建议我加一个中间态引导，其实我没有很理解
emmmmmmmm明天再看看把，吃火锅去了