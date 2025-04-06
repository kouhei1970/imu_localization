"""
Text File Example

This example demonstrates how to use the IMU localization package
with IMU data from a text file in the specified format.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import os
from typing import List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.imu_data import IMUReading
from src.localization import IMULocalizer


def load_imu_data_from_text(filename: str) -> List[IMUReading]:
    """Load IMU data from a text file with space-separated values.
    
    The expected format is:
    timestamp sample_interval gyro_x gyro_y gyro_z accel_x accel_y accel_z
    
    Args:
        filename: Path to the text file
        
    Returns:
        List of IMU readings
    """
    readings = []
    
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            values = line.strip().split()
            if len(values) < 8:
                print(f"Warning: Skipping line with insufficient data: {line.strip()}")
                continue
            
            try:
                timestamp = float(values[0])
                gyro_x = float(values[2])
                gyro_y = float(values[3])
                gyro_z = float(values[4])
                accel_x = float(values[5])
                accel_y = float(values[6])
                accel_z = float(values[7])
                
                reading = IMUReading(
                    timestamp=timestamp,
                    accel=np.array([accel_x, accel_y, accel_z]),
                    gyro=np.array([gyro_x, gyro_y, gyro_z])
                )
                
                readings.append(reading)
                
            except ValueError as e:
                print(f"Warning: Could not parse line: {line.strip()}")
                print(f"Error: {e}")
    
    return readings


def save_trajectory(positions: List[np.ndarray], filename: str) -> None:
    """Save trajectory to a text file.
    
    Args:
        positions: List of position vectors [x, y, z]
        filename: Path to the output text file
    """
    with open(filename, 'w') as f:
        f.write("# x y z\n")
        
        for pos in positions:
            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")


def plot_3d_trajectory(positions: List[np.ndarray], title: str = "3D Trajectory"):
    """Plot a 3D trajectory.
    
    Args:
        positions: List of position vectors [x, y, z]
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = [pos[0] for pos in positions]
    y = [pos[1] for pos in positions]
    z = [pos[2] for pos in positions]
    
    ax.plot(x, y, z, 'b-')
    
    ax.scatter(x[0], y[0], z[0], c='g', marker='o', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='r', marker='x', s=100, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def create_sample_data_file(filename: str, duration: float = 10.0, freq: float = 100.0):
    """Create a sample IMU data file with circular motion.
    
    Args:
        filename: Path to the output file
        duration: Duration of the simulation in seconds
        freq: Sampling frequency in Hz
    """
    dt = 1.0 / freq
    num_samples = int(duration * freq)
    
    with open(filename, 'w') as f:
        f.write("# timestamp sample_interval gyro_x gyro_y gyro_z accel_x accel_y accel_z\n")
        
        for i in range(num_samples):
            t = i * dt
            
            radius = 1.0
            angular_velocity = 2 * np.pi / 10  # One circle every 10 seconds
            
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
            
            gyro_x = 0.0
            gyro_y = 0.0
            gyro_z = angular_velocity
            
            ax += np.random.normal(0, 0.1)
            ay += np.random.normal(0, 0.1)
            az += np.random.normal(0, 0.1)
            gyro_x += np.random.normal(0, 0.01)
            gyro_y += np.random.normal(0, 0.01)
            gyro_z += np.random.normal(0, 0.01)
            
            f.write(f"{t} {dt} {gyro_x} {gyro_y} {gyro_z} {ax} {ay} {az}\n")
    
    print(f"Created sample data file: {filename}")


def main():
    """Run the example."""
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "sample_imu_data.txt"
        create_sample_data_file(data_file)
    
    print(f"Loading IMU data from {data_file}...")
    try:
        readings = load_imu_data_from_text(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loaded {len(readings)} IMU readings")
    
    # localizer = IMULocalizer(
    # )
    localizer = IMULocalizer()
    
    print("Processing IMU data...")
    start_time = time.time()
    results = localizer.process_batch(readings)
    end_time = time.time()
    
    print(f"Processed {len(readings)} readings in {end_time - start_time:.3f} seconds")
    
    positions = [result["position"] for result in results]
    
    output_file = "trajectory.txt"
    save_trajectory(positions, output_file)
    print(f"Saved trajectory to {output_file}")
    
    plot_3d_trajectory(positions, title="IMU Localization: Text File Data")


if __name__ == "__main__":
    main()
