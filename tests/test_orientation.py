"""
Tests for the orientation module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orientation import (
    quaternion_multiply, 
    quaternion_conjugate,
    quaternion_to_euler,
    euler_to_quaternion,
    quaternion_to_rotation_matrix,
    MadgwickFilter,
    ComplementaryFilter
)


class TestQuaternionFunctions(unittest.TestCase):
    """Test quaternion utility functions."""
    
    def test_quaternion_multiply(self):
        """Test quaternion multiplication."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        
        result = quaternion_multiply(q1, q2)
        np.testing.assert_array_almost_equal(result, q2)
        
        q1 = np.array([0.7071, 0.7071, 0.0, 0.0])  # 90 degree rotation around x
        q2 = np.array([0.7071, 0.0, 0.7071, 0.0])  # 90 degree rotation around y
        
        result = quaternion_multiply(q1, q2)
        expected = np.array([0.5, 0.5, 0.5, 0.5])  # Combined rotation
        np.testing.assert_array_almost_equal(result, expected, decimal=4)
    
    def test_quaternion_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        
        result = quaternion_conjugate(q)
        expected = np.array([0.5, -0.5, -0.5, -0.5])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_quaternion_to_euler(self):
        """Test conversion from quaternion to Euler angles."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        
        result = quaternion_to_euler(q)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
        
        q = np.array([0.7071, 0.7071, 0.0, 0.0])
        
        result = quaternion_to_euler(q)
        expected = np.array([np.pi/2, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)
    
    def test_euler_to_quaternion(self):
        """Test conversion from Euler angles to quaternion."""
        euler = np.array([0.0, 0.0, 0.0])
        
        result = euler_to_quaternion(euler)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
        
        euler = np.array([0.0, 0.0, np.pi/2])
        
        result = euler_to_quaternion(euler)
        expected = np.array([0.7071, 0.0, 0.0, 0.7071])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)
    
    def test_quaternion_to_rotation_matrix(self):
        """Test conversion from quaternion to rotation matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        
        result = quaternion_to_rotation_matrix(q)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(result, expected)
        
        q = np.array([0.7071, 0.7071, 0.0, 0.0])
        
        result = quaternion_to_rotation_matrix(q)
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)


class TestMadgwickFilter(unittest.TestCase):
    """Test the Madgwick filter."""
    
    def test_initialization(self):
        """Test filter initialization."""
        filter = MadgwickFilter(beta=0.1, sample_freq=100.0)
        
        self.assertEqual(filter.beta, 0.1)
        self.assertEqual(filter.sample_freq, 100.0)
        np.testing.assert_array_equal(filter.quaternion, np.array([1.0, 0.0, 0.0, 0.0]))
    
    def test_update_gravity_alignment(self):
        """Test that the filter aligns with gravity."""
        filter = MadgwickFilter(beta=0.5, sample_freq=100.0)
        
        accel = np.array([0.0, 0.0, 1.0])
        gyro = np.array([0.0, 0.0, 0.0])
        
        for _ in range(10):
            filter.update(accel, gyro, dt=0.01)
        
        euler = quaternion_to_euler(filter.quaternion)
        
        self.assertAlmostEqual(euler[0], 0.0, places=1)  # Roll
        self.assertAlmostEqual(euler[1], 0.0, places=1)  # Pitch


class TestComplementaryFilter(unittest.TestCase):
    """Test the complementary filter."""
    
    def test_initialization(self):
        """Test filter initialization."""
        filter = ComplementaryFilter(alpha=0.98, sample_freq=100.0)
        
        self.assertEqual(filter.alpha, 0.98)
        self.assertEqual(filter.sample_freq, 100.0)
        np.testing.assert_array_equal(filter.quaternion, np.array([1.0, 0.0, 0.0, 0.0]))
    
    def test_update_gravity_alignment(self):
        """Test that the filter aligns with gravity."""
        filter = ComplementaryFilter(alpha=0.5, sample_freq=100.0)
        
        accel = np.array([0.0, 0.0, 1.0])
        gyro = np.array([0.0, 0.0, 0.0])
        
        for _ in range(10):
            filter.update(accel, gyro, dt=0.01)
        
        euler = quaternion_to_euler(filter.quaternion)
        
        self.assertAlmostEqual(euler[0], 0.0, places=1)  # Roll
        self.assertAlmostEqual(euler[1], 0.0, places=1)  # Pitch


if __name__ == '__main__':
    unittest.main()
