# simple_imu_position_estimator.py の使用方法

このサンプルコードは、IMUデータを使用して単純な積分による位置推定を行うためのクラスを提供します。

## 必要なライブラリ

```
pip install numpy
```

## 使用方法

1. 必要なクラスをインポートします：
   ```python
   from simple_imu_position_estimator import SimpleIMUPositionEstimator
   ```

2. 位置推定器のインスタンスを作成します：
   ```python
   estimator = SimpleIMUPositionEstimator()
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

SimpleIMUPositionEstimatorは以下のステップで位置推定を行います：
1. 角速度の単純な積分により姿勢角を推定
2. 姿勢角から回転行列を計算して加速度から重力成分を除去
3. 加速度の二重積分により速度と位置を計算

## 注意点

- この単純な積分方式では、ジャイロのドリフトや加速度のノイズにより、短時間で大きな誤差が蓄積します
- 特に角度のドリフトは重力補正の精度に影響し、位置推定の誤差を急速に増大させます
- 実用的なシステムでは、Madgwickフィルタやカルマンフィルタなどの高度なアルゴリズムが必要です
- 長時間の使用には外部参照（GPS、ビジョンなど）との併用が必須となります

## テスト例

コードには以下のテスト関数が含まれています：
- 静止状態のテスト：重力のみが存在する状態での位置推定
- 等加速度運動のテスト：X方向に一定加速度がある場合の位置推定
- 回転運動のテスト：Z軸周りに回転する場合の位置推定と姿勢角の変化

```python
python simple_imu_position_estimator.py
```
を実行すると、テスト結果が表示されます。
