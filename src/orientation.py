"""
Orientation Module

This module provides functions and classes for estimating orientation
from IMU data using various algorithms.
"""

import numpy as np
from typing import List, Tuple, Optional
from .imu_data import IMUReading


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Result of quaternion multiplication [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Calculate the conjugate of a quaternion.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    w, x, y, z = q
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        Quaternion [w, x, y, z]
    """
    roll, pitch, yaw = euler
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w
    
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (yy + zz)
    R[0, 1] = 2 * (xy - zw)
    R[0, 2] = 2 * (xz + yw)
    R[1, 0] = 2 * (xy + zw)
    R[1, 1] = 1 - 2 * (xx + zz)
    R[1, 2] = 2 * (yz - xw)
    R[2, 0] = 2 * (xz - yw)
    R[2, 1] = 2 * (yz + xw)
    R[2, 2] = 1 - 2 * (xx + yy)
    
    return R


class MadgwickFilter:
    """Madgwick filter for orientation estimation from IMU data."""
    
    def __init__(self, beta: float = 0.1, sample_freq: float = 100.0):
        """Initialize the Madgwick filter.
        
        Args:
            beta: Filter gain (higher values converge faster but are less stable)
            sample_freq: Expected sample frequency in Hz
        """
        self.beta = beta
        self.sample_freq = sample_freq
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation [w, x, y, z]
    
    def update(self, accel: np.ndarray, gyro: np.ndarray, 
               mag: Optional[np.ndarray] = None, dt: Optional[float] = None) -> np.ndarray:
        """Update orientation estimate using IMU data.
        
        Args:
            accel: Acceleration vector [x, y, z] in m/s^2
            gyro: Angular velocity vector [x, y, z] in rad/s
            mag: Optional magnetometer data [x, y, z] in μT
            dt: Time step in seconds (if None, uses 1/sample_freq)
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        if dt is None:
            dt = 1.0 / self.sample_freq
        
        if np.linalg.norm(accel) > 0:
            accel = accel / np.linalg.norm(accel)
        
        q = self.quaternion
        q1, q2, q3, q4 = q[0], q[1], q[2], q[3]  # Note: Madgwick uses q1=w, q2=x, q3=y, q4=z
        
        F = np.array([
            2.0 * (q2 * q4 - q1 * q3) - accel[0],
            2.0 * (q1 * q2 + q3 * q4) - accel[1],
            2.0 * (0.5 - q2**2 - q3**2) - accel[2]
        ])
        
        J = np.array([
            [-2.0 * q3, 2.0 * q4, -2.0 * q1, 2.0 * q2],
            [2.0 * q2, 2.0 * q1, 2.0 * q4, 2.0 * q3],
            [0.0, -4.0 * q2, -4.0 * q3, 0.0]
        ])
        
        step = J.T @ F
        step = step / np.linalg.norm(step) if np.linalg.norm(step) > 0 else step
        
        qDot = 0.5 * np.array([
            -q2 * gyro[0] - q3 * gyro[1] - q4 * gyro[2],
            q1 * gyro[0] + q3 * gyro[2] - q4 * gyro[1],
            q1 * gyro[1] - q2 * gyro[2] + q4 * gyro[0],
            q1 * gyro[2] + q2 * gyro[1] - q3 * gyro[0]
        ])
        
        qDot = qDot - self.beta * step
        
        q = q + qDot * dt
        q = q / np.linalg.norm(q)  # Normalize quaternion
        
        self.quaternion = q
        return q
    
    def update_batch(self, readings: List[IMUReading]) -> List[np.ndarray]:
        """Process a batch of IMU readings.
        
        Args:
            readings: List of IMU readings
            
        Returns:
            List of quaternions corresponding to each reading
        """
        quaternions = []
        
        for i, reading in enumerate(readings):
            if i > 0:
                dt = reading.timestamp - readings[i-1].timestamp
            else:
                dt = 1.0 / self.sample_freq
            
            q = self.update(reading.accel, reading.gyro, reading.mag, dt)
            quaternions.append(q.copy())
        
        return quaternions
    
    def reset(self) -> None:
        """Reset the filter to initial state."""
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])


class ComplementaryFilter:
    """Complementary filter for orientation estimation from IMU data."""
    
    def __init__(self, alpha: float = 0.98, sample_freq: float = 100.0):
        """Initialize the complementary filter.
        
        Args:
            alpha: Weight for gyroscope data (0-1)
            sample_freq: Expected sample frequency in Hz
        """
        self.alpha = alpha
        self.sample_freq = sample_freq
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation [w, x, y, z]
    
    def update(self, accel: np.ndarray, gyro: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """Update orientation estimate using IMU data.
        
        Args:
            accel: Acceleration vector [x, y, z] in m/s^2
            gyro: Angular velocity vector [x, y, z] in rad/s
            dt: Time step in seconds (if None, uses 1/sample_freq)
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        if dt is None:
            dt = 1.0 / self.sample_freq
        
        if np.linalg.norm(accel) > 0:
            accel = accel / np.linalg.norm(accel)
        
        roll_accel = np.arctan2(accel[1], accel[2])
        pitch_accel = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        
        roll_gyro, pitch_gyro, yaw_gyro = quaternion_to_euler(self.quaternion)
        
        roll = self.alpha * (roll_gyro + gyro[0] * dt) + (1 - self.alpha) * roll_accel
        pitch = self.alpha * (pitch_gyro + gyro[1] * dt) + (1 - self.alpha) * pitch_accel
        yaw = yaw_gyro + gyro[2] * dt  # Yaw can only be estimated from gyro
        
        self.quaternion = euler_to_quaternion(np.array([roll, pitch, yaw]))
        return self.quaternion
    
    def update_batch(self, readings: List[IMUReading]) -> List[np.ndarray]:
        """Process a batch of IMU readings.
        
        Args:
            readings: List of IMU readings
            
        Returns:
            List of quaternions corresponding to each reading
        """
        quaternions = []
        
        for i, reading in enumerate(readings):
            if i > 0:
                dt = reading.timestamp - readings[i-1].timestamp
            else:
                dt = 1.0 / self.sample_freq
            
            q = self.update(reading.accel, reading.gyro, dt)
            quaternions.append(q.copy())
        
        return quaternions
    
    def reset(self) -> None:
        """Reset the filter to initial state."""
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
