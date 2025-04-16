# process_csv_data.py の使用方法

このサンプルコードは、CSVファイルからIMUデータを読み込み、姿勢推定を行うためのものです。

## 必要なライブラリ

```
pip install numpy pandas
```

## 使用方法

1. IMUデータを含むCSVファイルを用意します。CSVファイルには以下の列が必要です：
   - `timestamp`: タイムスタンプ（秒）
   - `acc_x`, `acc_y`, `acc_z`: 加速度データ（m/s²）
   - `gyro_x`, `gyro_y`, `gyro_z`: 角速度データ（rad/s）

2. コードを実行します：
   ```
   python process_csv_data.py
   ```

3. 処理結果は `attitude_estimation_results.csv` として保存されます。

## 出力データ

出力されるCSVファイルには以下の列が含まれます：
- `timestamp`: タイムスタンプ
- `euler_roll`, `euler_pitch`, `euler_yaw`: ジャイロセンサーから推定したオイラー角（度）
- `accel_roll`, `accel_pitch`: 加速度センサーから推定したロールとピッチ角（度）

## カスタマイズ

- CSVファイルのパスを変更するには、`csv_path` 変数を編集します。
- 出力ファイル名を変更するには、`results.to_csv()` の引数を編集します。
