import numpy as np

class QuaternionAttitudeEstimator:
    def __init__(self):
        # クォータニオン初期化 [q0, q1, q2, q3]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro: np.ndarray, dt: float):
        """
        角速度データからクォータニオンを更新

        Args:
            gyro: 角速度ベクトル [rad/s] (x, y, z)
            dt: 時間間隔 [s]
        """
        # 角速度ベクトルの大きさ
        w_norm = np.linalg.norm(gyro)

        if w_norm > 1e-10:  # ゼロ除算を防ぐ
            # 回転軸の単位ベクトル
            rotation_axis = gyro / w_norm

            # 回転角
            rotation_angle = w_norm * dt

            # 回転のクォータニオン
            dq = np.array([
                np.cos(rotation_angle/2),
                rotation_axis[0] * np.sin(rotation_angle/2),
                rotation_axis[1] * np.sin(rotation_angle/2),
                rotation_axis[2] * np.sin(rotation_angle/2)
            ])

            # クォータニオンの更新（クォータニオン積）
            self.q = self._quaternion_multiply(self.q, dq)

            # 正規化
            self.q = self.q / np.linalg.norm(self.q)

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """クォータニオンの積を計算

        Args:
            q1: 1番目のクォータニオン [q0, q1, q2, q3]
            q2: 2番目のクォータニオン [q0, q1, q2, q3]

        Returns:
            np.ndarray: クォータニオンの積 [q0, q1, q2, q3]
        """
        """クォータニオンの積を計算"""
        q10, q11, q12, q13 = q1
        q20, q21, q22, q23 = q2

        return np.array([
            q10*q20 - q11*q21 - q12*q22 - q13*q23,
            q10*q21 + q11*q20 + q12*q23 - q13*q22,
            q10*q22 - q11*q23 + q12*q20 + q13*q21,
            q10*q23 + q11*q22 - q12*q21 + q13*q20
        ])

    def to_euler(self):
        """クォータニオンからオイラー角への変換"""
        q0, q1, q2, q3 = self.q

        # ロール（X軸周り）
        roll = atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))

        # ピッチ（Y軸周り）
        pitch = asin(2*(q0*q2 - q3*q1))

        # ヨー（Z軸周り）
        yaw = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))

        return roll, pitch, yaw
