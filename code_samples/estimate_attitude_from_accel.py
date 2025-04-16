import numpy as np
from math import atan2, sqrt

def estimate_attitude_from_accel(ax, ay, az):
    """
    加速度データからオイラー角を推定

    Parameters:
    ax, ay, az: 各軸の加速度 [m/s^2]

    Returns:
    roll, pitch: ロール角、ピッチ角 [rad]
    """
    # ロール角の計算
    roll = atan2(ay, sqrt(ax*ax + az*az))

    # ピッチ角の計算
    pitch = atan2(-ax, sqrt(ay*ay + az*az))

    return roll, pitch

# ヨー角は加速度センサーのみでは求められない
