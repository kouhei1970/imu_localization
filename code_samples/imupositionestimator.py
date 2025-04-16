import numpy as np
class IMUPositionEstimator:
    def __init__(self):
        """Madgwickフィルタ + EKFによる位置推定器の初期化"""
        # Madgwickフィルタ（姿勢推定用）
        self.attitude = MadgwickFilter(beta=0.1)

        # EKFの状態変数 [x, y, z, vx, vy, vz, ax_bias, ay_bias, az_bias]
        self.x = np.zeros(9)
        self.P = np.eye(9)  # 共分散行列

        # システムノイズ
        self.Q = np.diag([
            0.01, 0.01, 0.01,  # 位置ノイズ
            0.1, 0.1, 0.1,      # 速度ノイズ
            0.001, 0.001, 0.001  # バイアスノイズ
        ])

        # ZUPT用パラメータ
        self.R_zupt = np.eye(3) * 0.01  # ZUPT観測ノイズ
        self.acc_buffer = collections.deque(maxlen=10)
        self.zupt_threshold = 0.1  # [m/s^2]

    def reset(self):
        """状態をリセット"""
        self.x = np.zeros(9)
        self.P = np.eye(9)
        self.attitude.reset()
        self.acc_buffer.clear()

    def update(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """IMUデータから位置を推定

        Args:
            acc: 加速度[m/s^2] [ax, ay, az]
            gyro: 角速度[rad/s] [wx, wy, wz]
            dt: 時間間隔[s]

        Returns:
            np.ndarray: 推定位置[m] [x, y, z]
        """
        # 1. Madgwickフィルタによる姿勢推定
        self.attitude.update(gyro, acc, dt)
        R = self.attitude.get_rotation_matrix()

        # 2. 加速度の座標変換とバイアス補正
        acc_global = R @ acc - np.array([0, 0, 9.81])
        acc_global -= self.x[6:9]  # バイアス補正

        # 3. EKFの状態予測ステップ
        # システム行列
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -np.eye(3) * dt

        # 状態予測
        self.x[0:3] += self.x[3:6] * dt + 0.5 * acc_global * dt**2
        self.x[3:6] += acc_global * dt

        # 共分散予測
        self.P = F @ self.P @ F.T + self.Q * dt

        # 4. ZUPT更新
        self.acc_buffer.append(np.linalg.norm(acc))
        if len(self.acc_buffer) == self.acc_buffer.maxlen:
            if self._detect_zupt():
                self._apply_zupt()

        return self.x[0:3]

    def _detect_zupt(self) -> bool:
        """静止状態の検出"""
        acc_std = np.std(list(self.acc_buffer))
        return acc_std < self.zupt_threshold

    def _apply_zupt(self):
        """静止状態での速度補正"""
        H = np.zeros((3, 9))
        H[:, 3:6] = np.eye(3)  # 速度の観測

        # カルマンゲイン
        S = H @ self.P @ H.T + self.R_zupt
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状態と共分散の更新
        self.x = self.x - K @ self.x[3:6]  # 速度をゼロに補正
        self.P = (np.eye(9) - K @ H) @ self.P