# imupositionestimator.py の使用方法

このサンプルコードは、IMUデータを使用して位置推定を行うためのクラスを提供します。

## 必要なライブラリ

```
pip install numpy
```

## 使用方法

1. 必要なクラスをインポートします：
   ```python
   from imupositionestimator import IMUPositionEstimator
   from madgwickfilter import MadgwickFilter  # 姿勢推定に使用
   ```

2. 位置推定器のインスタンスを作成します：
   ```python
   estimator = IMUPositionEstimator()
   ```

3. IMUデータを使用して位置を更新します：
   ```python
   # acc: 加速度データ [ax, ay, az] (m/s²)
   # gyro: 角速度データ [wx, wy, wz] (rad/s)
   # dt: 時間間隔 (秒)
   position = estimator.update(acc, gyro, dt)
   
   # 位置情報の取得
   print(f"Position: X={position[0]:.3f}m, Y={position[1]:.3f}m, Z={position[2]:.3f}m")
   ```

## 仕組み

IMUPositionEstimatorは以下のステップで位置推定を行います：
1. Madgwickフィルタを使用して姿勢を推定
2. 推定した姿勢を使用して加速度から重力成分を除去
3. 加速度の二重積分により速度と位置を計算
4. 静止状態検出によるドリフト補正

## カスタマイズ

- 加速度閾値を変更するには、`acc_threshold` を調整します
- 速度閾値を変更するには、`vel_threshold` を調整します
- 姿勢推定アルゴリズムを変更するには、`attitude` の初期化を修正します

## 注意点

- IMUのみによる位置推定は時間とともに誤差が蓄積します
- 長時間の使用には外部参照（GPS、ビジョンなど）との併用が推奨されます
- 高精度なIMUセンサーを使用することで精度が向上します
