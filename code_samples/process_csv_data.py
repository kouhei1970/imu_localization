import numpy as np
import pandas as pd
from pathlib import Path

def process_csv_data(csv_path: Path):
    """CSVファイルからIMUデータを読み込んで処理"""
    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)

    # 推定器の初期化
    euler_estimator = EulerAttitudeEstimator()
    quat_estimator = QuaternionAttitudeEstimator()

    # 結果を格納するリスト
    results = []

    # データを順次処理
    for i in range(1, len(df)):
        # 時間間隔の計算
        dt = df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']

        # 現在のデータ
        current_data = df.iloc[i]

        # ジャイロデータによる姿勢推定
        euler_estimator.update(
            current_data['gyro_x'],
            current_data['gyro_y'],
            current_data['gyro_z'],
            dt
        )

        quat_estimator.update(
            current_data['gyro_x'],
            current_data['gyro_y'],
            current_data['gyro_z'],
            dt
        )

        # 加速度データによる姿勢推定
        roll, pitch = estimate_attitude_from_accel(
            current_data['acc_x'],
            current_data['acc_y'],
            current_data['acc_z']
        )

        # 結果を保存
        euler_angles = {
            'timestamp': current_data['timestamp'],
            'euler_roll': np.degrees(euler_estimator.roll),
            'euler_pitch': np.degrees(euler_estimator.pitch),
            'euler_yaw': np.degrees(euler_estimator.yaw),
            'accel_roll': np.degrees(roll),
            'accel_pitch': np.degrees(pitch)
        }
        results.append(euler_angles)

    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    return results_df

def main_csv():
    # CSVファイルのパス
    csv_path = Path('sample_imu_data.csv')

    # データの処理
    results = process_csv_data(csv_path)

    # 結果の表示
    print("\n=== CSVデータによる姿勢推定結果 ===")
    print(results)

    # 結果の保存（オプション）
    results.to_csv('attitude_estimation_results.csv', index=False)

if __name__ == '__main__':
    print("1: 仮想IMUによる実時間処理")
    print("2: CSVファイルからのデータ処理")
    choice = input("処理モードを選択してください (1/2): ")

    if choice == '1':
        main_virtual()
    elif choice == '2':
        main_csv()
    else:
        print("無効な選択です")
