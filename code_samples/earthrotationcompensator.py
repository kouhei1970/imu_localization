import numpy as np
class EarthRotationCompensator:
    def __init__(self, latitude: float):
        """
        Args:
            latitude: 緯度（度単位）
        """
        self.earth_rotation_rate = 7.2921150e-5  # rad/s
        self.latitude_rad = np.radians(latitude)

    def get_earth_rotation_rates(self) -> np.ndarray:
        """各軸における地球の自転角速度を計算

        Returns:
            np.ndarray: [x, y, z]軸方向の地球自転角速度 [rad/s]
        """
        # 北方向（X軸）の成分
        omega_x = self.earth_rotation_rate * np.cos(self.latitude_rad)

        # 東方向（Y軸）の成分
        omega_y = 0.0

        # 上方向（Z軸）の成分
        omega_z = self.earth_rotation_rate * np.sin(self.latitude_rad)

        return np.array([omega_x, omega_y, omega_z])

    def compensate_gyro_data(
        self,
        gyro_data: np.ndarray,
        orientation: np.ndarray = None
    ) -> np.ndarray:
        """ジャイロデータから地球の自転の影響を除去

        Args:
            gyro_data: ジャイロデータ（Nx3行列）
            orientation: 現在の姿勢（オイラー角）。Noneの場合、
                        センサーが水平に設置されていると仮定。

        Returns:
            np.ndarray: 補正済みのジャイロデータ
        """
        earth_rotation = self.get_earth_rotation_rates()

        if orientation is not None:
            # 姿勢から回転行列を計算
            roll, pitch, yaw = orientation
            R = self.euler_to_rotation_matrix(roll, pitch, yaw)

            # 地球の自転をセンサー座標系に変換
            earth_rotation = R @ earth_rotation

        # 地球の自転の影響を除去
        compensated_data = gyro_data - earth_rotation

        return compensated_data

    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """オイラー角から回転行列を計算"""
        # 各軸の回転行列
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # ZYX順の回転行列の合成
        R = Rz @ Ry @ Rx
        return R

# 使用例
def earth_rotation_example():
    # 東京の緯度（35.6762° N）での補正例
    compensator = EarthRotationCompensator(latitude=35.6762)

    # 地球の自転角速度の計算
    earth_rates = compensator.get_earth_rotation_rates()
    print("地球の自転角速度（rad/s）:")
    print(f"X軸（北方向）: {earth_rates[0]:.9f}")
    print(f"Y軸（東方向）: {earth_rates[1]:.9f}")
    print(f"Z軸（上方向）: {earth_rates[2]:.9f}")

    # ジャイロデータの補正例
    # ダミーデータを使用
    gyro_data = np.array([
        [0.0001, 0.0, 0.0001],  # rad/s
        [0.0001, 0.0, 0.0001],
        [0.0001, 0.0, 0.0001]
    ])

    # 水平設置の場合
    compensated_data = compensator.compensate_gyro_data(gyro_data)

    # 任意の姿勢の場合
    orientation = np.array([np.pi/6, np.pi/4, np.pi/3])  # オイラー角
    compensated_data_with_orientation = compensator.compensate_gyro_data(
        gyro_data, orientation
    )

    print("\n補正前のジャイロデータ:")
    print(gyro_data[0])
    print("\n補正後のジャイロデータ（水平設置）:")
    print(compensated_data[0])
