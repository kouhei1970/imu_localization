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

def test_estimate_attitude():
    """テスト用の関数 - 水平姿勢と45度傾いた姿勢をテスト"""
    ax, ay, az = 0.0, 0.0, -9.81
    roll, pitch = estimate_attitude_from_accel(ax, ay, az)
    print(f"水平姿勢: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")
    
    ax, ay, az = 0.0, 9.81 * np.sin(np.pi/4), -9.81 * np.cos(np.pi/4)
    roll, pitch = estimate_attitude_from_accel(ax, ay, az)
    print(f"X軸45度回転: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")

if __name__ == "__main__":
    test_estimate_attitude()

# ヨー角は加速度センサーのみでは求められない
