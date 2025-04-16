import re
import os

def extract_code_samples(markdown_file, output_dir):
    """
    Extract Python code samples from a markdown file and save them as separate files.
    
    Args:
        markdown_file: Path to the markdown file
        output_dir: Directory to save the extracted code samples
    
    Returns:
        List of paths to the extracted code files
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'```python\n(.*?)```'
    code_blocks = re.findall(pattern, content, re.DOTALL)
    
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_files = []
    
    for i, code in enumerate(code_blocks):
        file_name = determine_file_name(code, i)
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        extracted_files.append(file_path)
        print(f"Extracted: {file_path}")
    
    return extracted_files

def determine_file_name(code, index):
    """
    Determine an appropriate file name based on the code content.
    
    Args:
        code: The code block
        index: Index of the code block
    
    Returns:
        A file name for the code block
    """
    class_match = re.search(r'class\s+(\w+)', code)
    if class_match:
        return f"{class_match.group(1).lower()}.py"
    
    func_match = re.search(r'def\s+(\w+)', code)
    if func_match:
        return f"{func_match.group(1).lower()}.py"
    
    if "mpu6050" in code:
        return "mpu6050_data_acquisition.py"
    elif "calibrate_accelerometer" in code:
        return "imu_calibration.py"
    elif "calibrate_with_90deg_rotation" in code:
        return "simple_gyro_calibration.py"
    elif "earth_rotation_rate" in code:
        return "earth_rotation_compensation.py"
    elif "estimate_attitude_from_accel" in code:
        return "accelerometer_attitude_estimation.py"
    elif "EulerAttitudeEstimator" in code:
        return "euler_attitude_estimation.py"
    elif "QuaternionAttitudeEstimator" in code:
        return "quaternion_attitude_estimation.py"
    elif "VirtualIMU" in code:
        return "virtual_imu.py"
    elif "IMUPositionEstimator" in code:
        return "imu_position_estimation.py"
    elif "process_csv_data" in code:
        return "csv_data_processing.py"
    elif "calculate_allan_variance" in code:
        return "allan_variance.py"
    elif "IMU_EKF" in code:
        return "extended_kalman_filter.py"
    elif "MadgwickFilter" in code:
        return "madgwick_filter.py"
    elif "ComplementaryFilter" in code:
        return "complementary_filter.py"
    
    return f"code_sample_{index+1}.py"

if __name__ == "__main__":
    markdown_file = "/home/ubuntu/imu_blog/imu_blog.md"
    output_dir = "/home/ubuntu/imu_blog/code_samples"
    
    extracted_files = extract_code_samples(markdown_file, output_dir)
    print(f"\nExtracted {len(extracted_files)} code samples.")
