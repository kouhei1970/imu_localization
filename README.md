# IMU Localization

IMU Localization is a Python package for estimating position and orientation using Inertial Measurement Unit (IMU) sensor data. This package processes accelerometer and gyroscope readings to determine location in 3D space without requiring external reference systems.

## Features

- Process IMU sensor data (accelerometer and gyroscope)
- Estimate orientation using Madgwick or complementary filter
- Track position through double integration of acceleration
- Apply Kalman filtering to improve position estimates
- Handle real or simulated IMU data
- Support for text file input with specified format

## Installation

Clone the repository:

```bash
git clone https://github.com/kouhei1970/imu_localization.git
cd imu_localization
```

Install the required dependencies:

```bash
pip install numpy matplotlib
```

## Usage

### Basic Usage

```python
from src.imu_data import IMUReading
from src.localization import IMULocalizer

# Create an IMU reading
reading = IMUReading(
    timestamp=0.01,
    accel=np.array([0.1, 0.2, 9.8]),  # Acceleration in m/s^2
    gyro=np.array([0.01, 0.02, 0.03])  # Angular velocity in rad/s
)

# Initialize the localizer with default settings (simple integration)
localizer = IMULocalizer()

# Process the reading
state = localizer.process_reading(reading)

# Access the results
orientation = state["orientation"]  # Quaternion [w, x, y, z]
position = state["position"]        # Position [x, y, z] in meters
velocity = state["velocity"]        # Velocity [vx, vy, vz] in m/s
```

### Examples

The package includes example scripts in the `examples` directory:

1. `simple_example.py`: Demonstrates the package with simulated circular motion data
2. `real_data_example.py`: Shows how to use the package with real IMU data from a CSV file
3. `text_file_example.py`: Processes IMU data from a text file with space-separated values

Run the examples:

```bash
# Run the simple example with simulated data
python examples/simple_example.py

# Run the real data example with your own IMU data in CSV format
python examples/real_data_example.py path/to/your/imu_data.csv

# Run the text file example with your own IMU data in text format
python examples/text_file_example.py path/to/your/imu_data.txt
```

## IMU Data Formats

### CSV Format (for real_data_example.py)

```
timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, [mag_x, mag_y, mag_z]
```

Where:
- `timestamp` is the time in seconds
- `accel_x/y/z` are accelerometer readings in m/s²
- `gyro_x/y/z` are gyroscope readings in rad/s
- `mag_x/y/z` are optional magnetometer readings in μT

### Text File Format (for text_file_example.py)

Space-separated values in the following order:
```
timestamp sample_interval gyro_x gyro_y gyro_z accel_x accel_y accel_z
```

Where:
- `timestamp` is the time in seconds
- `sample_interval` is the time between samples in seconds
- `gyro_x/y/z` are gyroscope readings in rad/s
- `accel_x/y/z` are accelerometer readings in m/s²

Example:
```
0.00 0.01 0.001 0.002 0.628 -0.981 0.023 9.81
0.01 0.01 0.002 0.001 0.629 -0.982 0.025 9.80
...
```

## Components

The package consists of several modules:

- `imu_data.py`: Classes for handling IMU sensor data
- `orientation.py`: Algorithms for estimating orientation from IMU data
- `position.py`: Methods for tracking position through integration
- `localization.py`: High-level interface combining orientation and position tracking

## Algorithms

### Default Method: Simple Integration

By default, the package uses simple integration methods:

1. **Orientation**: Direct integration of gyroscope angular velocity data to update quaternion orientation.
2. **Position**: Double integration of acceleration data (transformed to world frame and gravity-corrected).

This approach is computationally efficient but may accumulate drift over time.

### Optional Filters

The package provides optional filters that can be enabled for improved accuracy:

#### Orientation Filters

1. **Madgwick Filter**: A computationally efficient algorithm that fuses gyroscope, accelerometer, and optionally magnetometer data to estimate orientation. It uses gradient descent to optimize quaternion orientation.

2. **Complementary Filter**: A simpler approach that combines high-pass filtered gyroscope data with low-pass filtered accelerometer data to estimate orientation.

#### Position Filtering

**Kalman Filter**: Improves position and velocity estimates by:
1. Predicting the next state based on the current state and a motion model.
2. Updating the state based on measurements.
3. Optimally combining predictions and measurements based on their uncertainties.

### Selecting Filters

You can select which filters to use when initializing the localizer:

```python
# Default: No filters (simple integration)
localizer = IMULocalizer()

# With Madgwick orientation filter
localizer = IMULocalizer(orientation_filter="madgwick")

# With Complementary orientation filter
localizer = IMULocalizer(orientation_filter="complementary")

# With Kalman filter for position
localizer = IMULocalizer(use_kalman=True)

# With both orientation and position filtering
localizer = IMULocalizer(orientation_filter="madgwick", use_kalman=True)
```

## Limitations

- IMU-only localization suffers from drift over time due to integration errors
- Best results are achieved with high-quality IMU sensors and frequent sampling
- For long-term accurate positioning, consider fusing with other sensors (GPS, visual odometry, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

# IMUローカライゼーション

IMUローカライゼーションは、慣性計測装置（IMU）センサーデータを使用して位置と向きを推定するためのPythonパッケージです。このパッケージは、加速度計とジャイロスコープの読み取り値を処理して、外部の参照システムを必要とせずに3D空間での位置を決定します。

## 特徴

- IMUセンサーデータ（加速度計とジャイロスコープ）の処理
- Madgwickフィルタまたは相補フィルタを使用した向きの推定
- 加速度の二重積分による位置の追跡
- カルマンフィルタリングによる位置推定の改善
- 実際のIMUデータまたはシミュレーションデータの処理
- 指定された形式のテキストファイル入力のサポート

## インストール

リポジトリをクローンします：

```bash
git clone https://github.com/kouhei1970/imu_localization.git
cd imu_localization
```

必要な依存関係をインストールします：

```bash
pip install numpy matplotlib
```

## 使用方法

### 基本的な使用法

```python
from src.imu_data import IMUReading
from src.localization import IMULocalizer

# IMU読み取りを作成
reading = IMUReading(
    timestamp=0.01,
    accel=np.array([0.1, 0.2, 9.8]),  # 加速度（m/s^2）
    gyro=np.array([0.01, 0.02, 0.03])  # 角速度（rad/s）
)

# デフォルト設定（単純積分）でローカライザを初期化
localizer = IMULocalizer()

# 読み取りを処理
state = localizer.process_reading(reading)

# 結果にアクセス
orientation = state["orientation"]  # クォータニオン [w, x, y, z]
position = state["position"]        # 位置 [x, y, z]（メートル）
velocity = state["velocity"]        # 速度 [vx, vy, vz]（m/s）
```

### 例

パッケージには`examples`ディレクトリに例スクリプトが含まれています：

1. `simple_example.py`：シミュレーションされた円運動データを使用したパッケージのデモ
2. `real_data_example.py`：CSVファイルからの実際のIMUデータを使用する方法を示す
3. `text_file_example.py`：スペース区切りの値を持つテキストファイルからIMUデータを処理

例を実行します：

```bash
# シミュレーションデータを使用した簡単な例を実行
python examples/simple_example.py

# CSV形式の独自のIMUデータを使用した実データ例を実行
python examples/real_data_example.py path/to/your/imu_data.csv

# テキスト形式の独自のIMUデータを使用したテキストファイル例を実行
python examples/text_file_example.py path/to/your/imu_data.txt
```

## IMUデータ形式

### CSV形式（real_data_example.py用）

```
timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, [mag_x, mag_y, mag_z]
```

ここで：
- `timestamp`は秒単位の時間
- `accel_x/y/z`はm/s²単位の加速度計の読み取り値
- `gyro_x/y/z`はrad/s単位の角速度計の読み取り値
- `mag_x/y/z`はオプションの地磁気計の読み取り値（μT単位）

### テキストファイル形式（text_file_example.py用）

以下の順序でスペース区切りの値：
```
timestamp sample_interval gyro_x gyro_y gyro_z accel_x accel_y accel_z
```

ここで：
- `timestamp`は秒単位の時間
- `sample_interval`はサンプル間の時間（秒）
- `gyro_x/y/z`はrad/s単位の角速度計の読み取り値
- `accel_x/y/z`はm/s²単位の加速度計の読み取り値

例：
```
0.00 0.01 0.001 0.002 0.628 -0.981 0.023 9.81
0.01 0.01 0.002 0.001 0.629 -0.982 0.025 9.80
...
```

## コンポーネント

パッケージはいくつかのモジュールで構成されています：

- `imu_data.py`：IMUセンサーデータを処理するためのクラス
- `orientation.py`：IMUデータから向きを推定するためのアルゴリズム
- `position.py`：積分による位置追跡のためのメソッド
- `localization.py`：向きと位置の追跡を組み合わせた高レベルインターフェース

## アルゴリズム

### デフォルトメソッド：単純積分

デフォルトでは、パッケージは単純な積分方法を使用します：

1. **向き**：ジャイロスコープの角速度データを直接積分してクォータニオン向きを更新します。
2. **位置**：加速度データ（ワールドフレームに変換され、重力補正済み）の二重積分。

このアプローチは計算効率が良いですが、時間とともにドリフトが蓄積する可能性があります。

### オプションのフィルタ

パッケージは精度向上のために有効にできるオプションのフィルタを提供しています：

#### 向きフィルタ

1. **Madgwickフィルタ**：ジャイロスコープ、加速度計、およびオプションで地磁気計のデータを融合して向きを推定する計算効率の良いアルゴリズム。勾配降下法を使用してクォータニオン向きを最適化します。

2. **相補フィルタ**：高域通過フィルタリングされたジャイロスコープデータと低域通過フィルタリングされた加速度計データを組み合わせるよりシンプルなアプローチ。

#### 位置フィルタリング

**カルマンフィルタ**：以下の方法で位置と速度の推定を改善します：
1. 現在の状態と運動モデルに基づいて次の状態を予測します。
2. 測定値に基づいて状態を更新します。
3. 不確実性に基づいて予測と測定を最適に組み合わせます。

### フィルタの選択

ローカライザを初期化するときに使用するフィルタを選択できます：

```python
# デフォルト：フィルタなし（単純積分）
localizer = IMULocalizer()

# Madgwick向きフィルタを使用
localizer = IMULocalizer(orientation_filter="madgwick")

# 相補向きフィルタを使用
localizer = IMULocalizer(orientation_filter="complementary")

# 位置用のカルマンフィルタを使用
localizer = IMULocalizer(use_kalman=True)

# 向きと位置の両方のフィルタリングを使用
localizer = IMULocalizer(orientation_filter="madgwick", use_kalman=True)
```

## 制限事項

- IMUのみのローカライゼーションは、積分誤差により時間とともにドリフトが発生します
- 最良の結果は、高品質のIMUセンサーと頻繁なサンプリングで得られます
- 長期的に正確な位置決めには、他のセンサー（GPS、視覚オドメトリなど）との融合を検討してください

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細はLICENSEファイルを参照してください。
