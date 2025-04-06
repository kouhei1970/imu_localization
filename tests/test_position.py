"""
Tests for the position module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.position import PositionEstimator, KalmanFilter
from src.orientation import euler_to_quaternion


class TestPositionEstimator(unittest.TestCase):
    """Test the position estimator."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = PositionEstimator()
        
        np.testing.assert_array_equal(estimator.position, np.zeros(3))
        np.testing.assert_array_equal(estimator.velocity, np.zeros(3))
        np.testing.assert_array_equal(estimator.gravity, np.array([0, 0, 9.81]))
    
    def test_update_no_acceleration(self):
        """Test that position doesn't change with no acceleration."""
        estimator = PositionEstimator()
        
        accel = np.array([0.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        pos = estimator.update(accel, quat, dt=0.01, correct_gravity=False)
        
        np.testing.assert_array_equal(pos, np.zeros(3))
    
    def test_update_constant_acceleration(self):
        """Test position update with constant acceleration."""
        estimator = PositionEstimator()
        
        accel = np.array([1.0, 0.0, 0.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        for _ in range(100):
            pos = estimator.update(accel, quat, dt=0.01, correct_gravity=False)
        
        expected_pos = np.array([0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(pos, expected_pos, decimal=2)
        
        expected_vel = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(estimator.velocity, expected_vel, decimal=2)
    
    def test_gravity_correction(self):
        """Test gravity correction."""
        estimator = PositionEstimator()
        
        accel = np.array([0.0, 0.0, 9.81])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        pos = estimator.update(accel, quat, dt=0.01, correct_gravity=True)
        
        np.testing.assert_array_almost_equal(pos, np.zeros(3), decimal=4)
        
        np.testing.assert_array_almost_equal(estimator.velocity, np.zeros(3), decimal=4)


class TestKalmanFilter(unittest.TestCase):
    """Test the Kalman filter."""
    
    def test_initialization(self):
        """Test filter initialization."""
        kf = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
        
        np.testing.assert_array_equal(kf.state, np.zeros(6))
        self.assertEqual(kf.Q[0, 0], 0.01)  # Process noise
        self.assertEqual(kf.R[0, 0], 0.1)   # Measurement noise
    
    def test_predict_no_acceleration(self):
        """Test prediction with no acceleration."""
        kf = KalmanFilter()
        
        state = kf.predict(dt=0.1)
        
        np.testing.assert_array_equal(state, np.zeros(6))
    
    def test_predict_with_acceleration(self):
        """Test prediction with constant acceleration."""
        kf = KalmanFilter()
        
        accel = np.array([1.0, 0.0, 0.0])
        state = kf.predict(dt=0.1, acceleration=accel)
        
        self.assertAlmostEqual(state[3], 0.1)  # vx = 1.0 * 0.1 = 0.1
        
        self.assertAlmostEqual(state[0], 0.0)
        
        state = kf.predict(dt=0.1, acceleration=accel)
        
        self.assertAlmostEqual(state[3], 0.2)  # vx = 0.1 + 1.0 * 0.1 = 0.2
        
        self.assertAlmostEqual(state[0], 0.01)  # x = 0.1 * 0.1 = 0.01
    
    def test_update(self):
        """Test measurement update."""
        kf = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
        
        kf.state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        
        measurement = np.array([1.1, 2.2, 3.3])
        state = kf.update(measurement)
        
        self.assertGreater(state[0], 1.0)  # x should increase
        self.assertGreater(state[1], 2.0)  # y should increase
        self.assertGreater(state[2], 3.0)  # z should increase
        
        self.assertAlmostEqual(state[3], 0.1)
        self.assertAlmostEqual(state[4], 0.2)
        self.assertAlmostEqual(state[5], 0.3)


if __name__ == '__main__':
    unittest.main()
