import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple
from eulerattitudeestimator import EulerAttitudeEstimator
from quaternionattitudeestimator import QuaternionAttitudeEstimator
from estimate_attitude_from_accel import estimate_attitude_from_accel

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
        # 角速度（ゆっくりとした回転を仮定）
        t = time.time()
        gyro = np.array([
            0.01 * np.sin(t),
            0.01 * np.cos(t),
            0.01 * np.sin(2 * t)
        ]) + np.random.normal(0, self.noise_std, size=3)

        # 姿勢を更新
        self.orientation += gyro * self.dt

        # 加速度（重力+ノイズ）
        acc = self.gravity + np.random.normal(0, self.noise_std, size=3)

        return IMUData(
            acc_x=acc[0], acc_y=acc[1], acc_z=acc[2],
            gyro_x=gyro[0], gyro_y=gyro[1], gyro_z=gyro[2]
        )

# メイン処理
def main_virtual(duration: float = 10.0, dt: float = 0.1):
    """仮想IMUによる姿勢推定のデモ

    Args:
        duration: シミュレーション時間 [秒]
        dt: サンプリング間隔 [秒]
    """
    # センサーと推定器の初期化
    imu = VirtualIMU(noise_std=0.01, dt=dt)
    euler_estimator = EulerAttitudeEstimator()
    quat_estimator = QuaternionAttitudeEstimator()

    # シミュレーションの開始時刻
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            # センサーデータの取得
            imu_data = imu.get_data()

            # ジャイロデータによる姿勢推定
            euler_estimator.update(
                imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z, dt
            )
            gyro_vector = np.array([imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z])
            quat_estimator.update(gyro_vector, dt)

            # 加速度データによる姿勢推定
            roll, pitch = estimate_attitude_from_accel(
                imu_data.acc_x, imu_data.acc_y, imu_data.acc_z
            )

            # 結果の表示
            print("\n=== 仮想IMUによる姿勢推定 ===")
            print(f"Elapsed Time: {time.time() - start_time:.1f}s")
            print("オイラー角による推定:")
            print(f"Roll: {np.degrees(euler_estimator.roll):.1f}°")
            print(f"Pitch: {np.degrees(euler_estimator.pitch):.1f}°")
            print(f"Yaw: {np.degrees(euler_estimator.yaw):.1f}°")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n処理を終了します")
