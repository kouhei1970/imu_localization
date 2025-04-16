# from.py (VirtualIMU) の使用方法

このサンプルコードは、仮想IMUを使用して姿勢推定を行うデモプログラムです。

## 必要なライブラリ

```
pip install numpy
```

## 使用方法

1. 以下のファイルが同じディレクトリに必要です：
   - `eulerattitudeestimator.py`
   - `quaternionattitudeestimator.py`
   - `estimate_attitude_from_accel.py`

2. コードを実行します：
   ```
   python from.py
   ```

3. プログラムは仮想IMUからのデータを使用して姿勢推定を行い、結果をリアルタイムで表示します。

## 出力データ

コンソールに以下の情報が表示されます：
- 経過時間
- オイラー角による推定結果（ロール、ピッチ、ヨー）

## 仕組み

1. `VirtualIMU` クラスが加速度と角速度のデータを生成します
2. `EulerAttitudeEstimator` と `QuaternionAttitudeEstimator` クラスがジャイロデータから姿勢を推定します
3. `estimate_attitude_from_accel` 関数が加速度データからロールとピッチ角を推定します
4. 結果を比較表示します

## カスタマイズ

- シミュレーション時間を変更するには、`main_virtual` 関数の `duration` パラメータを調整します
- サンプリング間隔を変更するには、`dt` パラメータを調整します
- ノイズレベルを変更するには、`VirtualIMU` のインスタンス化時に `noise_std` パラメータを調整します
