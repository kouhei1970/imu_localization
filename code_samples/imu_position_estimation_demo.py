import numpy as np
def imu_position_estimation_demo():
    # 位置推定器の初期化
    estimator = IMUPositionEstimator()
    imu = VirtualIMU(noise_std=0.001)  # 高精度IMUを想定

    # メインループ
    try:
        while True:
            # IMUデータの取得
            data = imu.get_data()
            acc = np.array([data.acc_x, data.acc_y, data.acc_z])
            gyro = np.array([data.gyro_x, data.gyro_y, data.gyro_z])

            # 位置の推定
            position = estimator.update(acc, gyro, dt=0.01)

            # 結果の表示
            print(f"Position: X={position[0]:.3f}m, "
                  f"Y={position[1]:.3f}m, "
                  f"Z={position[2]:.3f}m")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n処理を終了します")
