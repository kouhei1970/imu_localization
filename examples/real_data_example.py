"""
Real Data Example

This example demonstrates how to use the IMU localization package
with real IMU data from a file.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import csv
from typing import List, Tuple, Optional

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.imu_data import IMUReading
from src.localization import IMULocalizer


def load_imu_data(filename: str) -> List[IMUReading]:
    """Load IMU data from a CSV file.
    
    The expected CSV format is:
    timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, [mag_x, mag_y, mag_z]
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        List of IMU readings
    """
    readings = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            timestamp = float(row[0])
            accel = np.array([float(row[1]), float(row[2]), float(row[3])])
            gyro = np.array([float(row[4]), float(row[5]), float(row[6])])
            
            mag = None
            if len(row) > 7:
                mag = np.array([float(row[7]), float(row[8]), float(row[9])])
            
            reading = IMUReading(
                timestamp=timestamp,
                accel=accel,
                gyro=gyro,
                mag=mag
            )
            
            readings.append(reading)
    
    return readings


def save_trajectory(positions: List[np.ndarray], filename: str) -> None:
    """Save trajectory to a CSV file.
    
    Args:
        positions: List of position vectors [x, y, z]
        filename: Path to the output CSV file
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])
        
        for pos in positions:
            writer.writerow([pos[0], pos[1], pos[2]])


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


def main():
    """Run the example."""
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        print("No data file provided. Please provide a path to an IMU data CSV file.")
        print("Usage: python real_data_example.py path/to/imu_data.csv")
        return
    
    print(f"Loading IMU data from {data_file}...")
    try:
        readings = load_imu_data(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loaded {len(readings)} IMU readings")
    
    localizer = IMULocalizer(
        orientation_filter="madgwick",
        use_kalman=True,
        beta=0.1,
        process_noise=0.01,
        measurement_noise=0.1
    )
    
    print("Processing IMU data...")
    start_time = time.time()
    results = localizer.process_batch(readings)
    end_time = time.time()
    
    print(f"Processed {len(readings)} readings in {end_time - start_time:.3f} seconds")
    
    positions = [result["position"] for result in results]
    
    output_file = "trajectory.csv"
    save_trajectory(positions, output_file)
    print(f"Saved trajectory to {output_file}")
    
    plot_3d_trajectory(positions, title="IMU Localization: Real Data")


if __name__ == "__main__":
    main()
