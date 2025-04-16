# MPU6050を使用した基本的なデータ取得例
from mpu6050 import mpu6050
from time import sleep

# センサーの初期化
sensor = mpu6050(address=0x68)

# データ読み取り
accel_data = sensor.get_accel_data()
gyro_data = sensor.get_gyro_data()

print(f"加速度: {accel_data}")
print(f"角速度: {gyro_data}")
