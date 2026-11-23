import numpy as np
from pso_mpc_parking.models import VehicleModel

x0 = np.array([0., 0., 0., -0.8, np.deg2rad(20.)])
model = VehicleModel(initial_state=x0)

# 预测未来 0.05 秒，但不移动车辆
predicted = model.predict_step(x0, [1, 1], dt=0.05)

print("预测的下一状态：", predicted)
print("车辆当前状态：", model.state)  # 仍然是 x0

# 真正推进车辆 0.05 秒
actual = model.step([0, 0], dt=0.05)

print("执行后的状态：", actual)
print("车辆当前状态：", model.state)  # 已经更新