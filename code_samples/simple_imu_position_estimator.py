import numpy as np
import math

class SimpleIMUPositionEstimator:
    def __init__(self):
        """単純な積分による位置推定器の初期化"""
        self.position = np.zeros(3)  # [x, y, z]
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.angles = np.zeros(3)    # [roll, pitch, yaw]

    def reset(self):
        """状態をリセット"""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.angles = np.zeros(3)

    def update(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """IMUデータから位置を推定

        Args:
            acc: 加速度[m/s^2] [ax, ay, az]
            gyro: 角速度[rad/s] [wx, wy, wz]
            dt: 時間間隔[s]

        Returns:
            np.ndarray: 推定位置[m] [x, y, z]
        """
        self.angles += gyro * dt

        roll, pitch, yaw = self.angles
        
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        
        Ry = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        
        acc_global = R @ acc - np.array([0, 0, 9.81])
        
        self.velocity += acc_global * dt
        
        self.position += self.velocity * dt
        
        return self.position

def test_simple_position_estimator():
    """SimpleIMUPositionEstimatorのテスト"""
    estimator = SimpleIMUPositionEstimator()
    
    acc = np.array([0.0, 0.0, 9.81])  # 重力加速度のみ
    gyro = np.array([0.0, 0.0, 0.0])  # 回転なし
    dt = 0.01  # 10ms
    
    print("静止状態のテスト:")
    for i in range(100):  # 1秒間
        position = estimator.update(acc, gyro, dt)
        if i % 20 == 0:  # 200msごとに表示
            print(f"t={i*dt:.2f}s, Position: X={position[0]:.6f}m, Y={position[1]:.6f}m, Z={position[2]:.6f}m")
    
    estimator.reset()
    
    acc = np.array([1.0, 0.0, 9.81])  # x方向に1m/s²の加速度
    gyro = np.array([0.0, 0.0, 0.0])  # 回転なし
    
    print("\n等加速度運動のテスト:")
    for i in range(100):  # 1秒間
        position = estimator.update(acc, gyro, dt)
        if i % 20 == 0:  # 200msごとに表示
            print(f"t={i*dt:.2f}s, Position: X={position[0]:.6f}m, Y={position[1]:.6f}m, Z={position[2]:.6f}m")
    
    estimator.reset()
    
    acc = np.array([0.0, 0.0, 9.81])  # 重力加速度のみ
    gyro = np.array([0.0, 0.0, 0.1])  # z軸周りに0.1rad/sで回転
    
    print("\n回転運動のテスト:")
    for i in range(100):  # 1秒間
        position = estimator.update(acc, gyro, dt)
        if i % 20 == 0:  # 200msごとに表示
            print(f"t={i*dt:.2f}s, Position: X={position[0]:.6f}m, Y={position[1]:.6f}m, Z={position[2]:.6f}m")
            print(f"Angles: Roll={math.degrees(estimator.angles[0]):.2f}°, Pitch={math.degrees(estimator.angles[1]):.2f}°, Yaw={math.degrees(estimator.angles[2]):.2f}°")

if __name__ == "__main__":
    test_simple_position_estimator()
