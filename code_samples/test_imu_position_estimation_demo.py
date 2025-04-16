import numpy as np
import time
from imupositionestimator import IMUPositionEstimator
from dataclasses import dataclass

@dataclass
class IMUData:
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

class VirtualIMU:
    def __init__(self, noise_std: float = 0.01, dt: float = 0.01):
        """仮想IMUセンサーの初期化
        
        Args:
            noise_std: センサーノイズの標準偏差
            dt: シミュレーションの時間間隔
        """
        self.orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.noise_std = noise_std
        self.dt = dt
        self.gravity = np.array([0.0, 0.0, 9.81])
    
    def get_data(self) -> IMUData:
        """センサーデータの取得
        
        Returns:
            IMUData: 加速度と角速度のデータ
        """
        t = time.time()
        gyro = np.array([
            0.01 * np.sin(t),
            0.01 * np.cos(t),
            0.01 * np.sin(2 * t)
        ]) + np.random.normal(0, self.noise_std, size=3)
        
        self.orientation += gyro * self.dt
        
        acc = self.gravity + np.random.normal(0, self.noise_std, size=3)
        
        return IMUData(
            acc_x=acc[0], acc_y=acc[1], acc_z=acc[2],
            gyro_x=gyro[0], gyro_y=gyro[1], gyro_z=gyro[2]
        )
def imu_position_estimation_demo():
    # 位置推定器の初期化
    estimator = IMUPositionEstimator()
    imu = VirtualIMU(noise_std=0.001)  # 高精度IMUを想定

    # メインループ
    try:
        while True:
            # IMUデータの取得
            data = imu.get_data()
            acc = np.array([data.acc_x, data.acc_y, data.acc_z])
            gyro = np.array([data.gyro_x, data.gyro_y, data.gyro_z])

            # 位置の推定
            position = estimator.update(acc, gyro, dt=0.01)

            # 結果の表示
            print(f"Position: X={position[0]:.3f}m, "
                  f"Y={position[1]:.3f}m, "
                  f"Z={position[2]:.3f}m")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n処理を終了します")


if __name__ == '__main__':
    imu_position_estimation_demo()