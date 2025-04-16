import numpy as np

adev_deg_s = 0.01  # 角速度のアラン偏差 [deg/s]
adev_ms2 = 0.001   # 加速度のアラン偏差 [m/s²]

# deg/s から deg/h への変換
adev_deg_h = adev_deg_s * 3600  # 3600秒 = 1時間

# m/s² から μg への変換
adev_ug = adev_ms2 * 1e6 / 9.81  # 1g = 9.81 m/s²

print(f"角速度ノイズ: {adev_deg_s:.6f} deg/s = {adev_deg_h:.6f} deg/h")
print(f"加速度ノイズ: {adev_ms2:.6f} m/s² = {adev_ug:.6f} μg")
