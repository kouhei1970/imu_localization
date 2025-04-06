"""
Tests for the imu_data module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.imu_data import IMUReading, IMUDataBuffer


class TestIMUReading(unittest.TestCase):
    """Test the IMUReading class."""
    
    def test_initialization(self):
        """Test reading initialization."""
        reading = IMUReading(
            timestamp=0.01,
            accel=np.array([1.0, 2.0, 3.0]),
            gyro=np.array([0.1, 0.2, 0.3])
        )
        
        self.assertAlmostEqual(reading.timestamp, 0.01)
        np.testing.assert_array_equal(reading.accel, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(reading.gyro, np.array([0.1, 0.2, 0.3]))
        self.assertIsNone(reading.mag)
        
        reading = IMUReading(
            timestamp=0.02,
            accel=[4.0, 5.0, 6.0],
            gyro=[0.4, 0.5, 0.6],
            mag=[10.0, 20.0, 30.0]
        )
        
        self.assertAlmostEqual(reading.timestamp, 0.02)
        np.testing.assert_array_equal(reading.accel, np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(reading.gyro, np.array([0.4, 0.5, 0.6]))
        np.testing.assert_array_equal(reading.mag, np.array([10.0, 20.0, 30.0]))


class TestIMUDataBuffer(unittest.TestCase):
    """Test the IMUDataBuffer class."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = IMUDataBuffer(max_size=100)
        
        self.assertEqual(buffer.max_size, 100)
        self.assertEqual(len(buffer.buffer), 0)
    
    def test_add_reading(self):
        """Test adding readings to the buffer."""
        buffer = IMUDataBuffer(max_size=3)
        
        reading1 = IMUReading(timestamp=0.01, accel=[1.0, 0.0, 0.0], gyro=[0.1, 0.0, 0.0])
        reading2 = IMUReading(timestamp=0.02, accel=[2.0, 0.0, 0.0], gyro=[0.2, 0.0, 0.0])
        
        buffer.add_reading(reading1)
        buffer.add_reading(reading2)
        
        self.assertEqual(len(buffer), 2)
        self.assertAlmostEqual(buffer.buffer[0].timestamp, 0.01)
        self.assertAlmostEqual(buffer.buffer[1].timestamp, 0.02)
        
        reading3 = IMUReading(timestamp=0.03, accel=[3.0, 0.0, 0.0], gyro=[0.3, 0.0, 0.0])
        reading4 = IMUReading(timestamp=0.04, accel=[4.0, 0.0, 0.0], gyro=[0.4, 0.0, 0.0])
        
        buffer.add_reading(reading3)
        buffer.add_reading(reading4)
        
        self.assertEqual(len(buffer), 3)
        self.assertAlmostEqual(buffer.buffer[0].timestamp, 0.02)
        self.assertAlmostEqual(buffer.buffer[1].timestamp, 0.03)
        self.assertAlmostEqual(buffer.buffer[2].timestamp, 0.04)
    
    def test_get_readings(self):
        """Test getting readings from the buffer."""
        buffer = IMUDataBuffer()
        
        for i in range(5):
            reading = IMUReading(
                timestamp=i * 0.1,
                accel=[float(i), 0.0, 0.0],
                gyro=[0.1 * i, 0.0, 0.0]
            )
            buffer.add_reading(reading)
        
        all_readings = buffer.get_readings()
        self.assertEqual(len(all_readings), 5)
        
        filtered_readings = buffer.get_readings(start_time=0.15, end_time=0.35)
        self.assertEqual(len(filtered_readings), 2)
        self.assertAlmostEqual(filtered_readings[0].timestamp, 0.2)
        self.assertAlmostEqual(filtered_readings[1].timestamp, 0.3)
        
        filtered_readings = buffer.get_readings(start_time=0.25)
        self.assertEqual(len(filtered_readings), 2)
        self.assertAlmostEqual(filtered_readings[0].timestamp, 0.3)
        self.assertAlmostEqual(filtered_readings[1].timestamp, 0.4)
        
        filtered_readings = buffer.get_readings(end_time=0.15)
        self.assertEqual(len(filtered_readings), 2)
        self.assertAlmostEqual(filtered_readings[0].timestamp, 0.0)
        self.assertAlmostEqual(filtered_readings[1].timestamp, 0.1)
    
    def test_get_time_range(self):
        """Test getting the time range of the buffer."""
        buffer = IMUDataBuffer()
        
        start_time, end_time = buffer.get_time_range()
        self.assertEqual(start_time, 0.0)
        self.assertEqual(end_time, 0.0)
        
        for i in range(3):
            reading = IMUReading(
                timestamp=i * 0.1 + 1.0,  # 1.0, 1.1, 1.2
                accel=[0.0, 0.0, 0.0],
                gyro=[0.0, 0.0, 0.0]
            )
            buffer.add_reading(reading)
        
        start_time, end_time = buffer.get_time_range()
        self.assertAlmostEqual(start_time, 1.0)
        self.assertAlmostEqual(end_time, 1.2)
    
    def test_clear(self):
        """Test clearing the buffer."""
        buffer = IMUDataBuffer()
        
        for i in range(3):
            reading = IMUReading(
                timestamp=i * 0.1,
                accel=[0.0, 0.0, 0.0],
                gyro=[0.0, 0.0, 0.0]
            )
            buffer.add_reading(reading)
        
        self.assertEqual(len(buffer), 3)
        
        buffer.clear()
        
        self.assertEqual(len(buffer), 0)


if __name__ == '__main__':
    unittest.main()
