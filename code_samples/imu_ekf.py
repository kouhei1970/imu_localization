import numpy as np
class IMU_EKF:
    def __init__(self):
        # 状態ベクトル: [q0, q1, q2, q3, bx, by, bz]
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # 共分散行列
        self.P = np.eye(7) * 0.1
        # プロセスノイズ
        self.Q = np.eye(7) * 0.001
        # 観測ノイズ
        self.R = np.eye(6) * 0.1
        # 重力加速度
        self.gravity = np.array([0, 0, 9.81])

    def predict(self, gyro: np.ndarray, dt: float):
        """予測ステップ"""
        # ジャイロバイアスを考慮した角速度
        wx = gyro[0] - self.x[4]
        wy = gyro[1] - self.x[5]
        wz = gyro[2] - self.x[6]

        # 状態遷移行列の計算
        F = self._compute_state_transition(wx, wy, wz, dt)

        # 状態予測
        self.x[:4] = self._quaternion_update(self.x[:4], wx, wy, wz, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, acc: np.ndarray, mag: np.ndarray = None):
        """更新ステップ"""
        # 観測行列とイノベーションの計算
        H = self._compute_observation_matrix()
        z = self._compute_measurement(acc, mag)
        y = z - self._compute_expected_measurement()

        # カルマンゲインの計算
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状態と共分散の更新
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P

        # クォータニオンの正規化
        q_norm = np.linalg.norm(self.x[:4])
        self.x[:4] /= q_norm

    def _compute_state_transition(self, wx: float, wy: float, wz: float, dt: float) -> np.ndarray:
        """状態遷移行列の計算"""
        q0, q1, q2, q3 = self.x[:4]
        F = np.eye(7)

        # クォータニオンの状態遷移
        F[0:4, 0:4] = np.array([
            [1, -wx*dt/2, -wy*dt/2, -wz*dt/2],
            [wx*dt/2, 1, wz*dt/2, -wy*dt/2],
            [wy*dt/2, -wz*dt/2, 1, wx*dt/2],
            [wz*dt/2, wy*dt/2, -wx*dt/2, 1]
        ])

        # バイアスの状態遷移
        F[0:4, 4:7] = np.array([
            [q1*dt/2, q2*dt/2, q3*dt/2],
            [-q0*dt/2, q3*dt/2, -q2*dt/2],
            [-q3*dt/2, -q0*dt/2, q1*dt/2],
            [q2*dt/2, -q1*dt/2, -q0*dt/2]
        ])

        return F

    def _quaternion_update(self, q: np.ndarray, wx: float, wy: float, wz: float, dt: float) -> np.ndarray:
        """クォータニオンの更新"""
        omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return q + 0.5 * omega @ q * dt

    def _compute_observation_matrix(self) -> np.ndarray:
        """観測行列の計算"""
        q0, q1, q2, q3 = self.x[:4]
        H = np.zeros((6, 7))

        # 加速度に関するヤコビアン
        H[0:3, 0:4] = np.array([
            [2*(q0*q2 - q1*q3), 2*(q1*q2 + q0*q3), 1-2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 + q0*q2), 1-2*(q0**2 + q3**2), 2*(q2*q3 - q0*q1), 2*(q1*q2 - q0*q3)],
            [1-2*(q1**2 + q2**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2), 2*(q2*q3 + q0*q1)]
        ]) * self.gravity[2]

        return H

    def _compute_measurement(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """観測値の計算"""
        z = np.zeros(6)
        z[:3] = acc
        if mag is not None:
            z[3:] = mag
        return z

    def _compute_expected_measurement(self) -> np.ndarray:
        """期待観測値の計算"""
        q0, q1, q2, q3 = self.x[:4]
        h = np.zeros(6)

        # 加速度の期待値
        R = self._quaternion_to_rotation_matrix(q0, q1, q2, q3)
        h[:3] = R @ self.gravity

        return h

    @staticmethod
    def _quaternion_to_rotation_matrix(q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
        """クォータニオンから回転行列への変換"""
        return np.array([
            [1-2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1-2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1-2*(q1**2 + q2**2)]
        ])
