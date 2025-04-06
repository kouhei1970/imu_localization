"""
IMU Data Module

This module provides classes for handling IMU sensor data including
accelerometer and gyroscope readings.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class IMUReading:
    """Class representing a single IMU reading."""
    timestamp: float  # Time in seconds
    accel: np.ndarray  # Acceleration in m/s^2 (x, y, z)
    gyro: np.ndarray   # Angular velocity in rad/s (x, y, z)
    mag: Optional[np.ndarray] = None  # Optional magnetometer data in μT (x, y, z)

    def __post_init__(self):
        """Ensure data is in numpy array format."""
        if not isinstance(self.accel, np.ndarray):
            self.accel = np.array(self.accel, dtype=float)
        if not isinstance(self.gyro, np.ndarray):
            self.gyro = np.array(self.gyro, dtype=float)
        if self.mag is not None and not isinstance(self.mag, np.ndarray):
            self.mag = np.array(self.mag, dtype=float)


class IMUDataBuffer:
    """Class for storing and managing a buffer of IMU readings."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize an empty buffer with a maximum size.
        
        Args:
            max_size: Maximum number of readings to store in the buffer
        """
        self.max_size = max_size
        self.buffer: List[IMUReading] = []
    
    def add_reading(self, reading: IMUReading) -> None:
        """Add a new IMU reading to the buffer.
        
        Args:
            reading: The IMU reading to add
        """
        self.buffer.append(reading)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # Remove oldest reading if buffer is full
    
    def get_readings(self, start_time: Optional[float] = None, 
                    end_time: Optional[float] = None) -> List[IMUReading]:
        """Get readings within a specified time range.
        
        Args:
            start_time: Start time in seconds (None for beginning of buffer)
            end_time: End time in seconds (None for end of buffer)
            
        Returns:
            List of IMU readings within the specified time range
        """
        if start_time is None and end_time is None:
            return self.buffer.copy()
        
        filtered_readings = []
        for reading in self.buffer:
            if (start_time is None or reading.timestamp >= start_time) and \
               (end_time is None or reading.timestamp <= end_time):
                filtered_readings.append(reading)
        
        return filtered_readings
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range covered by the buffer.
        
        Returns:
            Tuple of (start_time, end_time) in seconds
        """
        if not self.buffer:
            return (0.0, 0.0)
        
        start_time = self.buffer[0].timestamp
        end_time = self.buffer[-1].timestamp
        return (start_time, end_time)
    
    def clear(self) -> None:
        """Clear all readings from the buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Get the number of readings in the buffer."""
        return len(self.buffer)
