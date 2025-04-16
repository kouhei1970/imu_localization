import numpy as np
class SimpleGyroCalibrator:
    def __init__(self):
        self.sampling_rate = 100  # Hz
        self.target_angle = np.pi/2  # 90度

    def calibrate_with_90deg_rotation(
        self,
        gyro_data: np.ndarray,
        axis: int
    ) -> float:
        """ジャイロデータの90度回転によるスケール係数の推定

        Args:
            gyro_data: ジャイロデータ（Nx3行列）
            axis: 回転軸（0=X, 1=Y, 2=Z）

        Returns:
            scale_factor: スケール係数
        """
        # 角速度の積分
        dt = 1.0 / self.sampling_rate
        integrated_angle = np.sum(gyro_data[:, axis]) * dt

        # スケール係数の計算
        scale_factor = self.target_angle / integrated_angle
        return scale_factor

    def perform_calibration(self):
        """キャリブレーション手順の例"""
        print("ジャイロセンサーの90度回転キャリブレーション")
        print("手順:")
        print("1. センサーを水平な面に設置")
        print("2. X軸回りに90度回転し、データを記録")
        print("3. Y軸回りに90度回転し、データを記録")
        print("4. Z軸回りに90度回転し、データを記録")

        # 実装例
        scale_factors = []
        for axis in range(3):
            input(f"{['X', 'Y', 'Z'][axis]}軸の90度回転を完了したらEnterを押してください")

            # ここで実際のデータ取得を行う
            # 例としてダミーデータを使用
            dummy_data = np.array([[0.1, 0.0, 0.0]] * 100)
            scale_factor = self.calibrate_with_90deg_rotation(dummy_data, axis)
            scale_factors.append(scale_factor)

        return np.diag(scale_factors)
