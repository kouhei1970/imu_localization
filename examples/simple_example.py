"""
Simple Example

This example demonstrates how to use the IMU localization package
with simulated IMU data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import List, Tuple

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.imu_data import IMUReading
from src.localization import IMULocalizer


def generate_circular_motion(duration: float = 10.0, 
                            freq: float = 100.0, 
                            radius: float = 1.0,
                            angular_velocity: float = 2*np.pi/10) -> List[IMUReading]:
    """Generate simulated IMU readings for circular motion.
    
    Args:
        duration: Duration of the simulation in seconds
        freq: Sampling frequency in Hz
        radius: Radius of the circular path in meters
        angular_velocity: Angular velocity in rad/s
        
    Returns:
        List of IMU readings
    """
    dt = 1.0 / freq
    num_samples = int(duration * freq)
    readings = []
    
    for i in range(num_samples):
        t = i * dt
        
        angle = angular_velocity * t
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0
        
        vx = -radius * angular_velocity * np.sin(angle)
        vy = radius * angular_velocity * np.cos(angle)
        vz = 0.0
        
        ax = -radius * angular_velocity**2 * np.cos(angle)
        ay = -radius * angular_velocity**2 * np.sin(angle)
        az = 0.0
        
        accel = np.array([ax, ay, az])
        
        gyro = np.array([0.0, 0.0, angular_velocity])
        
        accel += np.random.normal(0, 0.1, 3)
        gyro += np.random.normal(0, 0.01, 3)
        
        reading = IMUReading(
            timestamp=t,
            accel=accel,
            gyro=gyro
        )
        
        readings.append(reading)
    
    return readings


def plot_trajectory(true_positions: List[np.ndarray], 
                   estimated_positions: List[np.ndarray],
                   title: str = "Trajectory Comparison"):
    """Plot true and estimated trajectories.
    
    Args:
        true_positions: List of true position vectors [x, y, z]
        estimated_positions: List of estimated position vectors [x, y, z]
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    true_x = [pos[0] for pos in true_positions]
    true_y = [pos[1] for pos in true_positions]
    true_z = [pos[2] for pos in true_positions]
    
    est_x = [pos[0] for pos in estimated_positions]
    est_y = [pos[1] for pos in estimated_positions]
    est_z = [pos[2] for pos in estimated_positions]
    
    ax.plot(true_x, true_y, true_z, 'b-', label='True')
    ax.plot(est_x, est_y, est_z, 'r-', label='Estimated')
    
    ax.scatter(true_x[0], true_y[0], true_z[0], c='g', marker='o', s=100, label='Start')
    ax.scatter(true_x[-1], true_y[-1], true_z[-1], c='k', marker='x', s=100, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """Run the example."""
    print("Generating simulated IMU data...")
    readings = generate_circular_motion(duration=10.0, freq=100.0, radius=1.0)
    
    print(f"Generated {len(readings)} IMU readings")
    
    # localizer = IMULocalizer(
    # )
    localizer = IMULocalizer()
    
    print("Processing IMU data...")
    start_time = time.time()
    results = localizer.process_batch(readings)
    end_time = time.time()
    
    print(f"Processed {len(readings)} readings in {end_time - start_time:.3f} seconds")
    
    true_positions = []
    estimated_positions = []
    
    for i, reading in enumerate(readings):
        t = reading.timestamp
        angle = 2 * np.pi / 10 * t
        
        true_pos = np.array([
            1.0 * np.cos(angle),
            1.0 * np.sin(angle),
            0.0
        ])
        
        est_pos = results[i]["position"]
        
        true_positions.append(true_pos)
        estimated_positions.append(est_pos)
    
    plot_trajectory(true_positions, estimated_positions, 
                   title="IMU Localization: Circular Motion")


if __name__ == "__main__":
    main()
