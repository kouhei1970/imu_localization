import numpy as np
class MadgwickFilter:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # q = [q0, q1, q2, q3]

    def update(self, gyro: np.ndarray, acc: np.ndarray, dt: float):
        """Madgwickフィルタの更新"""
        if np.linalg.norm(acc) == 0:
            print("警告: accのノルムがゼロです。更新をスキップします。")
            return

        acc = acc / np.linalg.norm(acc)

        q0, q1, q2, q3 = self.q

        # 目的関数 f とそのヤコビアン J による動的な調整
        f = np.array([
            2*(q1*q3 - q0*q2) - acc[0],
            2*(q0*q1 + q2*q3) - acc[1],
            2*(0.5 - q1*q1 - q2*q2) - acc[2]
        ])
        J = np.array([
            [-2*q2,  2*q3, -2*q0, 2*q1],
            [ 2*q1,  2*q0,  2*q3, 2*q2],
            [   0, -4*q1, -4*q2,    0]
        ])
        grad = J.T @ f
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm

        # クォータニオンの微分（ジャイロスコープ + 動的な動きの調整）
        qDot = 0.5 * np.array([
            -q1 * gyro[0] - q2 * gyro[1] - q3 * gyro[2],
             q0 * gyro[0] + q2 * gyro[2] - q3 * gyro[1],
             q0 * gyro[1] - q1 * gyro[2] + q3 * gyro[0],
             q0 * gyro[2] + q1 * gyro[1] - q2 * gyro[0]
        ]) - self.beta * grad

        # クォータニオンの積分と正規化
        self.q += qDot * dt
        self.q /= np.linalg.norm(self.q)

    def get_euler_angles(self) -> np.ndarray:
        """クォータニオンからオイラー角（roll, pitch, yaw）を取得"""
        q0, q1, q2, q3 = self.q

        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
        pitch = np.arcsin(2*(q0*q2 - q3*q1))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))

        return np.array([roll, pitch, yaw])
