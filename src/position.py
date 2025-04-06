"""
Position Module

This module provides functions and classes for estimating position
from IMU data and orientation estimates.
"""

import numpy as np
from typing import List, Tuple, Optional
from .imu_data import IMUReading
from .orientation import quaternion_to_rotation_matrix


class PositionEstimator:
    """Class for estimating position from IMU data and orientation."""
    
    def __init__(self):
        """Initialize the position estimator."""
        self.position = np.zeros(3)  # [x, y, z] in meters
        self.velocity = np.zeros(3)  # [vx, vy, vz] in m/s
        self.gravity = np.array([0, 0, 9.81])  # Gravity vector in m/s^2
    
    def update(self, accel: np.ndarray, quaternion: np.ndarray, 
               dt: float, correct_gravity: bool = True) -> np.ndarray:
        """Update position estimate using IMU data and orientation.
        
        Args:
            accel: Acceleration vector in sensor frame [x, y, z] in m/s^2
            quaternion: Orientation quaternion [w, x, y, z]
            dt: Time step in seconds
            correct_gravity: Whether to remove gravity from acceleration
            
        Returns:
            Updated position [x, y, z] in meters
        """
        R = quaternion_to_rotation_matrix(quaternion)
        
        accel_world = R @ accel
        
        if correct_gravity:
            accel_world = accel_world - self.gravity
        
        self.velocity += accel_world * dt
        self.position += self.velocity * dt
        
        return self.position
    
    def update_batch(self, readings: List[IMUReading], quaternions: List[np.ndarray], 
                     correct_gravity: bool = True) -> List[np.ndarray]:
        """Process a batch of IMU readings with corresponding orientations.
        
        Args:
            readings: List of IMU readings
            quaternions: List of orientation quaternions corresponding to readings
            correct_gravity: Whether to remove gravity from acceleration
            
        Returns:
            List of positions corresponding to each reading
        """
        positions = []
        
        for i, (reading, quat) in enumerate(zip(readings, quaternions)):
            if i > 0:
                dt = reading.timestamp - readings[i-1].timestamp
            else:
                dt = 0.01  # Assume 100Hz for first reading
            
            pos = self.update(reading.accel, quat, dt, correct_gravity)
            positions.append(pos.copy())
        
        return positions
    
    def reset(self) -> None:
        """Reset the estimator to initial state."""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)


class KalmanFilter:
    """Kalman filter for position and velocity estimation."""
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """Initialize the Kalman filter.
        
        Args:
            process_noise: Process noise covariance scalar
            measurement_noise: Measurement noise covariance scalar
        """
        self.state = np.zeros(6)
        
        self.F = np.eye(6)
        
        self.Q = np.eye(6) * process_noise
        
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # z
        
        self.R = np.eye(3) * measurement_noise
        
        self.P = np.eye(6)
    
    def predict(self, dt: float, acceleration: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict the next state.
        
        Args:
            dt: Time step in seconds
            acceleration: Optional acceleration vector [ax, ay, az] in m/s^2
            
        Returns:
            Predicted state vector [x, y, z, vx, vy, vz]
        """
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt
        
        B = np.zeros((6, 3))
        B[3, 0] = dt  # vx += ax * dt
        B[4, 1] = dt  # vy += ay * dt
        B[5, 2] = dt  # vz += az * dt
        
        if acceleration is not None:
            self.state = self.F @ self.state + B @ acceleration
        else:
            self.state = self.F @ self.state
        
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update the state with a measurement.
        
        Args:
            measurement: Measurement vector [x, y, z] in meters
            
        Returns:
            Updated state vector [x, y, z, vx, vy, vz]
        """
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state
    
    def reset(self) -> None:
        """Reset the filter to initial state."""
        self.state = np.zeros(6)
        self.P = np.eye(6)
