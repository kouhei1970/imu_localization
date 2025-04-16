# class.py (IMUCalibrator) の使用方法

このサンプルコードは、IMUセンサのキャリブレーションを行うためのクラスを提供します。

## 必要なライブラリ

```
pip install numpy
```

## 使用方法

1. クラスをインポートします：
   ```python
   from class import IMUCalibrator, CalibrationConfig, CalibrationResult
   ```

2. キャリブレータのインスタンスを作成します：
   ```python
   # デフォルト設定でインスタンス化
   calibrator = IMUCalibrator()
   
   # カスタム設定でインスタンス化
   config = CalibrationConfig(
       num_static_samples=2000,
       num_rotation_samples=3000,
       temperature_samples=1000
   )
   calibrator = IMUCalibrator(config)
   ```

3. 加速度センサのキャリブレーションを行います：
   ```python
   # 6方向（上・下・左・右・前・後）の静止状態での加速度測定値
   static_acc_samples = [
       np.array([0.1, 0.2, 9.7]),   # 上向き
       np.array([0.2, 0.1, -9.9]),  # 下向き
       np.array([0.1, 9.8, 0.2]),   # 左向き
       np.array([0.2, -9.7, 0.1]),  # 右向き
       np.array([9.8, 0.1, 0.2]),   # 前向き
       np.array([-9.9, 0.2, 0.1])   # 後向き
   ]
   
   acc_scale, acc_bias = calibrator.calibrate_accelerometer(static_acc_samples)
   ```

4. ジャイロセンサのキャリブレーションを行います：
   ```python
   # 静止状態でのジャイロ測定値
   static_gyro_samples = [np.array([0.01, 0.02, -0.01]) for _ in range(100)]
   
   # 既知の角速度での回転測定値と真の角速度のペア
   rotation_samples = [
       (np.array([1.02, 0.01, 0.02]), np.array([1.0, 0.0, 0.0])),
       (np.array([0.01, 0.98, 0.01]), np.array([0.0, 1.0, 0.0])),
       (np.array([0.02, 0.01, 1.03]), np.array([0.0, 0.0, 1.0]))
   ]
   
   gyro_scale, gyro_bias = calibrator.calibrate_gyroscope(
       static_gyro_samples, rotation_samples
   )
   ```

5. 温度キャリブレーションを行います：
   ```python
   # ジャイロ測定値と温度のペア
   temp_samples = [
       (np.array([0.01, 0.02, 0.01]), 20.0),  # 20℃
       (np.array([0.02, 0.03, 0.02]), 25.0),  # 25℃
       (np.array([0.03, 0.04, 0.03]), 30.0)   # 30℃
   ]
   
   temp_coef = calibrator.calibrate_temperature(temp_samples)
   ```

6. キャリブレーション結果を適用します：
   ```python
   # キャリブレーション結果の作成
   cal_result = CalibrationResult(
       acc_scale=acc_scale,
       acc_bias=acc_bias,
       gyro_scale=gyro_scale,
       gyro_bias=gyro_bias,
       temp_coef=temp_coef
   )
   
   # 生データにキャリブレーション結果を適用
   raw_data = np.array([
       [0.1, 0.2, 9.8, 0.01, 0.02, 0.01],  # [ax, ay, az, gx, gy, gz]
       [0.2, 0.1, 9.7, 0.02, 0.01, 0.02]
   ])
   
   # 温度データなしでキャリブレーション
   calibrated_data = calibrator.apply_calibration(raw_data, cal_result)
   
   # 温度データありでキャリブレーション
   calibrated_data = calibrator.apply_calibration(raw_data, cal_result, temperature=25.0)
   ```

## キャリブレーションの手順

1. **加速度センサのキャリブレーション**:
   - センサを6つの異なる方向（上・下・左・右・前・後）に静止させて測定
   - 各方向で十分な数のサンプルを取得
   - 最小二乗法でスケール行列とバイアスベクトルを推定

2. **ジャイロセンサのキャリブレーション**:
   - 静止状態でバイアスを推定
   - 既知の角速度で回転させてスケール係数を推定

3. **温度キャリブレーション**:
   - 異なる温度環境でジャイロ出力を測定
   - 温度と出力の関係を線形回帰で推定

## 注意点

- キャリブレーションは振動のない安定した環境で行ってください
- 各測定は十分な時間をかけて安定した状態で行ってください
- 定期的に再キャリブレーションを行うことで精度を維持できます
