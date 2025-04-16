# complementaryfilter.py の使用方法

このサンプルコードは、相補フィルタを使用してIMUデータから姿勢推定を行うためのものです。

## 必要なライブラリ

```
pip install numpy
```

## 使用方法

1. `ComplementaryFilter` クラスをインポートします：
   ```python
   from complementaryfilter import ComplementaryFilter
   ```

2. フィルタのインスタンスを作成します：
   ```python
   # 基本的な相補フィルタ（カットオフ周波数0.1Hz）
   cf = ComplementaryFilter(fc=0.1, adaptive=False)
   
   # 適応型相補フィルタ
   cf_adaptive = ComplementaryFilter(fc=0.1, adaptive=True)
   ```

3. IMUデータを使用して姿勢を更新します：
   ```python
   # acc: 加速度データ [ax, ay, az]
   # gyro: 角速度データ [wx, wy, wz]
   # dt: 時間間隔（秒）
   roll, pitch, yaw = cf.update(acc, gyro, dt)
   ```

## 相補フィルタの仕組み

相補フィルタは以下の特徴を持ちます：
- 加速度センサからの姿勢推定（低周波成分）とジャイロセンサからの姿勢推定（高周波成分）を組み合わせます
- カットオフ周波数（`fc`）で両者の重みを調整します
- 適応型フィルタ（`adaptive=True`）では、加速度の信頼性に応じて動的に重みを調整します

## カスタマイズ

- カットオフ周波数を変更するには、インスタンス化時に `fc` パラメータを調整します
- 適応型フィルタを使用するには、`adaptive=True` を設定します
- 適応型フィルタの感度を調整するには、`acc_threshold` と `acc_gain` を変更します
