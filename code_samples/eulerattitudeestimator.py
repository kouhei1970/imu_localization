import numpy as np
from math import sin, cos

class EulerAttitudeEstimator:
    def __init__(self):
        self.roll = 0.0   # ロール角 [rad]
        self.pitch = 0.0  # ピッチ角 [rad]
        self.yaw = 0.0    # ヨー角 [rad]

    def update(self, wx, wy, wz, dt):
        """
        角速度データからオイラー角を更新

        Parameters:
        wx, wy, wz: 各軸の角速度 [rad/s]
        dt: 時間間隔 [s]
        """
        # オイラー角の変化率
        roll_dot = wx + sin(self.roll) * tan(self.pitch) * wy + \
                   cos(self.roll) * tan(self.pitch) * wz
        pitch_dot = cos(self.roll) * wy - sin(self.roll) * wz
        yaw_dot = sin(self.roll) / cos(self.pitch) * wy + \
                  cos(self.roll) / cos(self.pitch) * wz

        # オイラー角の更新（積分）
        self.roll += roll_dot * dt
        self.pitch += pitch_dot * dt
        self.yaw += yaw_dot * dt
