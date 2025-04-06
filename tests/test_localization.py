"""
Tests for the localization module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.imu_data import IMUReading
from src.localization import IMULocalizer


class TestIMULocalizer(unittest.TestCase):
    """Test the IMU localizer."""
    
    def test_initialization(self):
        """Test localizer initialization."""
        localizer = IMULocalizer()
        
        self.assertEqual(localizer.use_kalman, False)
        self.assertEqual(localizer.use_orientation_filter, False)
        np.testing.assert_array_equal(localizer.current_position, np.zeros(3))
        np.testing.assert_array_equal(localizer.current_velocity, np.zeros(3))
        np.testing.assert_array_equal(localizer.current_orientation, np.array([1.0, 0.0, 0.0, 0.0]))
        
        localizer = IMULocalizer(
            orientation_filter="complementary",
            use_kalman=True,
            buffer_size=500,
            alpha=0.95
        )
        
        self.assertEqual(localizer.use_kalman, True)
        self.assertEqual(localizer.use_orientation_filter, True)
        self.assertEqual(localizer.data_buffer.max_size, 500)
    
    def test_process_reading(self):
        """Test processing a single IMU reading."""
        localizer = IMULocalizer()
        
        reading = IMUReading(
            timestamp=0.01,
            accel=np.array([0.0, 0.0, 9.81]),  # Just gravity
            gyro=np.array([0.0, 0.0, 0.0])     # No rotation
        )
        
        state = localizer.process_reading(reading)
        
        self.assertIn("orientation", state)
        self.assertIn("position", state)
        self.assertIn("velocity", state)
        
        np.testing.assert_array_almost_equal(state["position"], np.zeros(3), decimal=4)
        
        np.testing.assert_array_almost_equal(state["velocity"], np.zeros(3), decimal=4)
    
    def test_process_batch(self):
        """Test processing a batch of IMU readings."""
        localizer = IMULocalizer()
        
        readings = []
        for i in range(10):
            reading = IMUReading(
                timestamp=i * 0.01,
                accel=np.array([1.0, 0.0, 9.81]),  # Acceleration in x + gravity
                gyro=np.array([0.0, 0.0, 0.0])     # No rotation
            )
            readings.append(reading)
        
        results = localizer.process_batch(readings)
        
        self.assertEqual(len(results), len(readings))
        
        self.assertGreater(results[-1]["position"][0], 0.0)
        
        self.assertGreater(results[-1]["velocity"][0], 0.0)
    
    def test_reset(self):
        """Test resetting the localizer."""
        localizer = IMULocalizer()
        
        reading = IMUReading(
            timestamp=0.01,
            accel=np.array([1.0, 0.0, 9.81]),
            gyro=np.array([0.1, 0.0, 0.0])
        )
        
        localizer.process_reading(reading)
        
        localizer.reset()
        
        np.testing.assert_array_equal(localizer.current_position, np.zeros(3))
        np.testing.assert_array_equal(localizer.current_velocity, np.zeros(3))
        np.testing.assert_array_equal(localizer.current_orientation, np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertIsNone(localizer.last_update_time)
        
        self.assertEqual(len(localizer.data_buffer), 0)
        
        localizer = IMULocalizer(orientation_filter="madgwick", use_kalman=True)
        
        localizer.process_reading(reading)
        
        localizer.reset()
        
        np.testing.assert_array_equal(localizer.current_position, np.zeros(3))
        np.testing.assert_array_equal(localizer.current_velocity, np.zeros(3))
        np.testing.assert_array_equal(localizer.current_orientation, np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertIsNone(localizer.last_update_time)
        
        self.assertEqual(len(localizer.data_buffer), 0)


if __name__ == '__main__':
    unittest.main()
