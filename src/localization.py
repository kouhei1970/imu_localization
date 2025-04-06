"""
Localization Module

This module provides a high-level interface for IMU-based localization,
combining orientation estimation and position tracking.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time

from .imu_data import IMUReading, IMUDataBuffer
from .orientation import MadgwickFilter, ComplementaryFilter
from .position import PositionEstimator, KalmanFilter


class IMULocalizer:
    """High-level class for IMU-based localization."""
    
    def __init__(self, 
                 orientation_filter: str = "none", 
                 use_kalman: bool = False,
                 buffer_size: int = 1000,
                 **kwargs):
        """Initialize the IMU localizer.
        
        Args:
            orientation_filter: Type of orientation filter ("none", "madgwick", or "complementary")
            use_kalman: Whether to use Kalman filtering for position
            buffer_size: Maximum size of the IMU data buffer
            **kwargs: Additional parameters for the filters
        """
        self.data_buffer = IMUDataBuffer(max_size=buffer_size)
        
        self.use_orientation_filter = orientation_filter.lower() != "none"
        
        if orientation_filter.lower() == "madgwick":
            beta = kwargs.get("beta", 0.1)
            sample_freq = kwargs.get("sample_freq", 100.0)
            self.orientation_filter = MadgwickFilter(beta=beta, sample_freq=sample_freq)
        elif orientation_filter.lower() == "complementary":
            alpha = kwargs.get("alpha", 0.98)
            sample_freq = kwargs.get("sample_freq", 100.0)
            self.orientation_filter = ComplementaryFilter(alpha=alpha, sample_freq=sample_freq)
        elif orientation_filter.lower() == "none":
            pass
        else:
            raise ValueError(f"Unknown orientation filter: {orientation_filter}")
        
        self.position_estimator = PositionEstimator()
        
        self.use_kalman = use_kalman
        if use_kalman:
            process_noise = kwargs.get("process_noise", 0.01)
            measurement_noise = kwargs.get("measurement_noise", 0.1)
            self.kalman_filter = KalmanFilter(
                process_noise=process_noise,
                measurement_noise=measurement_noise
            )
        
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion [w, x, y, z]
        self.current_position = np.zeros(3)  # [x, y, z] in meters
        self.current_velocity = np.zeros(3)  # [vx, vy, vz] in m/s
        self.last_update_time = None
    
    def process_reading(self, reading: IMUReading) -> Dict[str, np.ndarray]:
        """Process a single IMU reading and update the state.
        
        Args:
            reading: IMU reading to process
            
        Returns:
            Dictionary with current orientation, position, and velocity
        """
        self.data_buffer.add_reading(reading)
        
        current_time = reading.timestamp
        if self.last_update_time is None:
            dt = 0.01  # Assume 100Hz for first reading
        else:
            dt = current_time - self.last_update_time
        
        if self.use_orientation_filter:
            self.current_orientation = self.orientation_filter.update(
                reading.accel, reading.gyro, reading.mag, dt
            )
        else:
            gyro = reading.gyro
            q = self.current_orientation
            q_dot = 0.5 * np.array([
                -q[1] * gyro[0] - q[2] * gyro[1] - q[3] * gyro[2],
                q[0] * gyro[0] + q[2] * gyro[2] - q[3] * gyro[1],
                q[0] * gyro[1] - q[1] * gyro[2] + q[3] * gyro[0],
                q[0] * gyro[2] + q[1] * gyro[1] - q[2] * gyro[0]
            ])
            q = q + q_dot * dt
            q = q / np.linalg.norm(q)
            self.current_orientation = q
        
        raw_position = self.position_estimator.update(
            reading.accel, self.current_orientation, dt, correct_gravity=True
        )
        
        if self.use_kalman:
            accel_world = np.zeros(3)  # This is already accounted for in position_estimator
            self.kalman_filter.predict(dt, accel_world)
            
            kalman_state = self.kalman_filter.update(raw_position)
            
            self.current_position = kalman_state[:3]
            self.current_velocity = kalman_state[3:6]
        else:
            self.current_position = raw_position
            self.current_velocity = self.position_estimator.velocity.copy()
        
        self.last_update_time = current_time
        
        return {
            "orientation": self.current_orientation,
            "position": self.current_position,
            "velocity": self.current_velocity
        }
    
    def process_batch(self, readings: List[IMUReading]) -> List[Dict[str, np.ndarray]]:
        """Process a batch of IMU readings.
        
        Args:
            readings: List of IMU readings to process
            
        Returns:
            List of dictionaries with orientation, position, and velocity for each reading
        """
        results = []
        
        for reading in readings:
            state = self.process_reading(reading)
            results.append(state.copy())
        
        return results
    
    def reset(self) -> None:
        """Reset the localizer to initial state."""
        self.data_buffer.clear()
        if self.use_orientation_filter:
            self.orientation_filter.reset()
        self.position_estimator.reset()
        if self.use_kalman:
            self.kalman_filter.reset()
        
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.last_update_time = None
