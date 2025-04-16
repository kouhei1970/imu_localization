# IMUセンサー入門：基礎から応用まで

## 必要なライブラリのインストール

このブログで紹介するサンプルコードを実行するには、以下のライブラリが必要です：

```bash
# 基本的な数値計算ライブラリ
pip install numpy

# データ処理・分析用ライブラリ
pip install pandas

# グラフ描画ライブラリ（オプション）
pip install matplotlib

# MPU6050センサー用ライブラリ（ハードウェア接続時のみ必要）
pip install mpu6050-raspberrypi
```

## はじめに

IMU（Inertial Measurement Unit：慣性計測装置）は、物体の動きや姿勢を計測するためのセンサーデバイスです。ロボット工学、ドローン、スマートフォン、VR/ARデバイスなど、様々な分野で活用されています。

## IMUとは

IMUは以下の主要なセンサーを組み合わせたものです：

- 加速度センサー：直線運動の加速度を検出
- ジャイロスコープ：角速度（回転の速さ）を検出
- 磁気センサー（オプション）：方位を検出

## 主な用途

1. ロボティクス
   - 自己位置推定
   - 姿勢制御
   - 動作安定化

2. モバイルデバイス
   - 画面の向き検出
   - ステップカウント
   - ナビゲーション補助

3. ドローン
   - 飛行安定化
   - 自動制御
   - ホバリング制御

## センサーの基本仕様

一般的なIMUセンサーの仕様：

- 加速度センサー：±2g～±16g
- ジャイロスコープ：±250～±2000度/秒
- サンプリングレート：100Hz～1000Hz

## データの取得方法

一般的なIMUセンサーは以下のインターフェースでデータを取得できます：

- I2C
- SPI
- UART

```python
# MPU6050を使用した基本的なデータ取得例
from mpu6050 import mpu6050
from time import sleep

# センサーの初期化
sensor = mpu6050(address=0x68)

# データ読み取り
accel_data = sensor.get_accel_data()
gyro_data = sensor.get_gyro_data()

print(f"加速度: {accel_data}")
print(f"角速度: {gyro_data}")
```

## キャリブレーション

IMUセンサーの正確な測定には、適切なキャリブレーションが不可欠です。

### 1. キャリブレーションの理論

#### 1.1 加速度センサーのキャリブレーション

加速度センサーの出力は以下のモデルで表現できます：

```
acc_measured = S * acc_true + B + N
```

ここで：
- `acc_measured`: 測定値
- `S`: スケール係数行列（3x3）
- `acc_true`: 真の加速度
- `B`: バイアス（オフセット）ベクトル
- `N`: ノイズ

#### 1.2 ジャイロセンサーのキャリブレーション

ジャイロセンサーの出力は以下のモデルで表現できます：

```
gyro_measured = S * gyro_true + B + B_t(T) + N
```

ここで：
- `gyro_measured`: 測定値
- `S`: スケール係数行列
- `gyro_true`: 真の角速度
- `B`: 静的バイアス
- `B_t(T)`: 温度依存バイアス
- `N`: ノイズ

### 2. キャリブレーションの実装

```python
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

# -------------------------------
# キャリブレーションの設定を格納するクラス
# -------------------------------
@dataclass
class CalibrationConfig:
    num_static_samples: int = 1000      # 静止状態での加速度センサ用サンプル数
    num_rotation_samples: int = 2000    # 回転中のジャイロセンサ用サンプル数
    temperature_samples: int = 500      # 温度キャリブレーション用サンプル数

# -------------------------------
# キャリブレーション結果を格納するクラス
# -------------------------------
@dataclass
class CalibrationResult:
    acc_scale: np.ndarray      # 加速度センサのスケール行列 (3x3)
    acc_bias: np.ndarray       # 加速度センサのバイアス (3次元ベクトル)
    gyro_scale: np.ndarray     # ジャイロのスケール行列 (3x3)
    gyro_bias: np.ndarray      # ジャイロのバイアス (3次元ベクトル)
    temp_coef: np.ndarray      # 温度補正係数（温度依存の傾きとバイアス） (2x3)

# -------------------------------
# IMUキャリブレーター本体
# -------------------------------
class IMUCalibrator:
    def __init__(self, config: CalibrationConfig = CalibrationConfig()):
        self.config = config
        self.reference_gravity = 9.81  # 重力加速度（m/s^2）

    # --------------------------
    # 加速度センサのキャリブレーション
    # --------------------------
    def calibrate_accelerometer(self, static_samples: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        6方向（上・下・左・右・前・後）の静止状態での加速度測定値から、
        スケール行列とバイアスを求める。
        """
        measurements = np.array(static_samples)  # 実測値 6x3

        # 各方向における理想的な重力加速度ベクトル
        reference = np.array([
            [0, 0, self.reference_gravity],    # 上向き
            [0, 0, -self.reference_gravity],   # 下向き
            [0, self.reference_gravity, 0],    # 左向き
            [0, -self.reference_gravity, 0],   # 右向き
            [self.reference_gravity, 0, 0],    # 前向き
            [-self.reference_gravity, 0, 0]    # 後向き
        ])

        # 最小二乗法で補正行列を求める（拡張行列Aを作成）
        A = np.hstack([measurements, np.ones((6, 1))])  # 6x4行列
        x, _, _, _ = np.linalg.lstsq(A, reference, rcond=None)  # 解 x は 4x3

        # 上3行がスケール行列、下1行がバイアスベクトル
        scale_matrix = x[:3, :]  # 3x3行列
        bias_vector = x[3, :]    # 1x3ベクトル

        return scale_matrix, bias_vector

    # --------------------------
    # ジャイロセンサのキャリブレーション
    # --------------------------
    def calibrate_gyroscope(
        self,
        static_samples: List[np.ndarray],
        rotation_samples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        静止時のバイアスと、既知の角速度でのスケール係数を推定。
        """
        static_data = np.array(static_samples)
        bias_vector = np.mean(static_data, axis=0)  # 静止状態の平均でバイアス推定

        # 回転中のデータ
        measured_rates = np.array([s[0] for s in rotation_samples])
        true_rates = np.array([s[1] for s in rotation_samples])
        corrected_rates = measured_rates - bias_vector  # バイアスを差し引く

        # 軸ごとにスケール係数を求める（対角行列）
        scale_factors = []
        for i in range(3):
            A = corrected_rates[:, i].reshape(-1, 1)
            b = true_rates[:, i]
            scale, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            scale_factors.append(scale[0])
        scale_matrix = np.diag(scale_factors)

        return scale_matrix, bias_vector

    def calibrate_temperature(self, temp_samples: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        ジャイロの値が温度によってどう変化するかを線形回帰で推定。
        temp_coef[0] は温度依存係数、temp_coef[1] はバイアス項
        """
        gyro_data = np.array([s[0] for s in temp_samples])  # N x 3
        temp_data = np.array([s[1] for s in temp_samples])  # N

        A = np.vstack([temp_data, np.ones(len(temp_data))]).T  # N x 2
        temp_coef = np.zeros((2, 3))  # 傾きとオフセットを軸ごとに保持

        for i in range(3):
            coef, _, _, _ = np.linalg.lstsq(A, gyro_data[:, i], rcond=None)
            temp_coef[:, i] = coef

        return temp_coef  # 2 x 3

    def apply_calibration(
        self,
        raw_data: np.ndarray,             # nx6行列（加速度 + ジャイロ）
        cal_result: CalibrationResult,    # キャリブレーション結果
        temperature: float = None         # 温度（任意）
    ) -> np.ndarray:
        acc_data = raw_data[:, :3]
        gyro_data = raw_data[:, 3:]

        # 加速度補正：バイアス除去→スケール補正
        acc_calibrated = (acc_data - cal_result.acc_bias) @ cal_result.acc_scale

        # ジャイロ補正：バイアス除去→スケール補正
        gyro_calibrated = (gyro_data - cal_result.gyro_bias) @ cal_result.gyro_scale

        # 温度補正（ある場合のみ）
        if temperature is not None:
            temp_slope = cal_result.temp_coef[0]   # 傾き
            temp_offset = cal_result.temp_coef[1]  # オフセット
            temp_correction = temp_slope * temperature + temp_offset
            gyro_calibrated -= temp_correction

        # 加速度とジャイロを結合して返す
        return np.hstack([acc_calibrated, gyro_calibrated])

# ------------------------------------
# 使用例（テスト用のダミーデータで実行）
# ------------------------------------
def calibration_example():
    calibrator = IMUCalibrator()

    # 加速度センサの静止データ（上・下・左・右・前・後）
    static_acc_samples = [
        np.array([0.1, 0.2, 9.7]),   # 上向き
        np.array([0.2, 0.1, -9.9]),  # 下向き
        np.array([0.1, 9.8, 0.2]),   # 左向き
        np.array([0.2, -9.7, 0.1]),  # 右向き
        np.array([9.8, 0.1, 0.2]),   # 前向き
        np.array([-9.9, 0.2, 0.1])   # 後向き
    ]

    # キャリブレーション実行
    acc_scale, acc_bias = calibrator.calibrate_accelerometer(static_acc_samples)

    # 結果表示
    print("✅ 加速度センサーキャリブレーション結果:")
    print("スケール行列（scale_matrix）:")
    print(acc_scale)
    print("バイアスベクトル（bias_vector）:")
    print(acc_bias)

# 実行例
if __name__ == "__main__":
    calibration_example()
        )

        return scale_matrix, bias_vector

    def calibrate_temperature(
        self,
        temp_samples: List[Tuple[np.ndarray, float]]
    ) -> np.ndarray:
        """温度依存性のキャリブレーション

        Args:
            temp_samples: ジャイロ測定値と温度のペア

        Returns:
            temp_coef: 温度係数（3次元ベクトル）
        """
        # データの準備
        gyro_data = np.array([sample[0] for sample in temp_samples])
        temp_data = np.array([sample[1] for sample in temp_samples])

        # 温度と出力の関係を線形回帰
        A = np.vstack([temp_data, np.ones(len(temp_data))]).T
        temp_coef = np.linalg.lstsq(A, gyro_data, rcond=None)[0]

        return temp_coef

    def apply_calibration(
        self,
        raw_data: np.ndarray,
        cal_result: CalibrationResult,
        temperature: float = None
    ) -> np.ndarray:
        """キャリブレーション結果を適用

        Args:
            raw_data: 生データ（nx6行列、加速度とジャイロ）
            cal_result: キャリブレーション結果
            temperature: 温度データ（オプション）

        Returns:
            calibrated_data: キャリブレーション済みデータ
        """
        # データの分離
        acc_data = raw_data[:, :3]
        gyro_data = raw_data[:, 3:]

        # 加速度データの補正
        acc_calibrated = np.dot(
            acc_data - cal_result.acc_bias,
            cal_result.acc_scale
        )

        # ジャイロデータの補正
        gyro_calibrated = np.dot(
            gyro_data - cal_result.gyro_bias,
            cal_result.gyro_scale
        )

        # 温度補正（温度データがある場合）
        if temperature is not None:
            temp_correction = cal_result.temp_coef * temperature
            gyro_calibrated -= temp_correction

        # 結果の結合
        return np.hstack([acc_calibrated, gyro_calibrated])

# 使用例
def calibration_example():
    # キャリブレータの初期化
    calibrator = IMUCalibrator()

    # サンプルデータの生成（実際のデータに置き換えてください）
    static_acc_samples = [
        np.array([0.1, 0.2, 9.7]),   # 上向き
        np.array([0.2, 0.1, -9.9]),  # 下向き
        np.array([0.1, 9.8, 0.2]),   # 左向き
        np.array([0.2, -9.7, 0.1]),  # 右向き
        np.array([9.8, 0.1, 0.2]),   # 前向き
        np.array([-9.9, 0.2, 0.1])   # 後向き
    ]

    # キャリブレーションの実行
    acc_scale, acc_bias = calibrator.calibrate_accelerometer(
        static_acc_samples
    )

    print("加速度センサーキャリブレーション結果:")
    print(f"スケール係数:\n{acc_scale}")
    print(f"バイアス: {acc_bias}")
```

### 3. キャリブレーションの手順

1. **静的キャリブレーション**
   - センサーを完全に静止させた状態で測定
   - バイアスとノイズレベルの推定
   - 複数の姿勢で測定してスケール係数を推定

2. **動的キャリブレーション**
   - 既知の角速度で回転させて測定
   - ジャイロセンサーのスケール係数を推定
   - クロスアキシス感度の評価

3. **90度回転による簡易キャリブレーション**

```python
class SimpleGyroCalibrator:
    def __init__(self):
        self.sampling_rate = 100  # Hz
        self.target_angle = np.pi/2  # 90度

    def calibrate_with_90deg_rotation(
        self,
        gyro_data: np.ndarray,
        axis: int
    ) -> float:
        """ジャイロデータの90度回転によるスケール係数の推定

        Args:
            gyro_data: ジャイロデータ（Nx3行列）
            axis: 回転軸（0=X, 1=Y, 2=Z）

        Returns:
            scale_factor: スケール係数
        """
        # 角速度の積分
        dt = 1.0 / self.sampling_rate
        integrated_angle = np.sum(gyro_data[:, axis]) * dt

        # スケール係数の計算
        scale_factor = self.target_angle / integrated_angle
        return scale_factor

    def perform_calibration(self):
        """キャリブレーション手順の例"""
        print("ジャイロセンサーの90度回転キャリブレーション")
        print("手順:")
        print("1. センサーを水平な面に設置")
        print("2. X軸回りに90度回転し、データを記録")
        print("3. Y軸回りに90度回転し、データを記録")
        print("4. Z軸回りに90度回転し、データを記録")

        # 実装例
        scale_factors = []
        for axis in range(3):
            input(f"{['X', 'Y', 'Z'][axis]}軸の90度回転を完了したらEnterを押してください")

            # ここで実際のデータ取得を行う
            # 例としてダミーデータを使用
            dummy_data = np.array([[0.1, 0.0, 0.0]] * 100)
            scale_factor = self.calibrate_with_90deg_rotation(dummy_data, axis)
            scale_factors.append(scale_factor)

        return np.diag(scale_factors)
```

この手法の特徴：
- 特別な装置が不要
- 手動で正確に90度回転
- 各軸のスケール係数を個別に推定
- 回転速度は任意（積分値で評価）

注意点：
- 回転はできるだけ正確に90度に
- 回転中は他の軸の動きを最小限に
- 各軸のキャリブレーションを複数回実施して平均を取る

4. **温度キャリブレーション**
   - 異なる温度環境で測定
   - 温度依存性の係数を推定
   - 温度補正モデルの作成

### 4. キャリブレーション時の注意点

1. **環境条件**
   - 振動のない安定した場所で実施
   - 温度が安定した環境で実施
   - 磁気の影響を受けない場所で実施

2. **測定手順**
   - 十分な数のサンプルを取得
   - 各姿勢で安定するまで待機
   - 温度変化は緩やかに実施

3. **データ処理**
   - 外れ値の除去
   - ノイズフィルタリング
   - 結果の妥当性確認

4. **定期的な再キャリブレーション**
   - センサー特性の経時変化に注意
   - 環境変化への対応
   - 定期的な精度確認

### 5. 地球の自転の影響と補正

高精度なIMUセンサーでは、地球の自転角速度（約 0.004°/s または 7.27×10⁻⁵ rad/s）を検知できます。

```python
class EarthRotationCompensator:
    def __init__(self, latitude: float):
        """
        Args:
            latitude: 緯度（度単位）
        """
        self.earth_rotation_rate = 7.2921150e-5  # rad/s
        self.latitude_rad = np.radians(latitude)

    def get_earth_rotation_rates(self) -> np.ndarray:
        """各軸における地球の自転角速度を計算

        Returns:
            np.ndarray: [x, y, z]軸方向の地球自転角速度 [rad/s]
        """
        # 北方向（X軸）の成分
        omega_x = self.earth_rotation_rate * np.cos(self.latitude_rad)

        # 東方向（Y軸）の成分
        omega_y = 0.0

        # 上方向（Z軸）の成分
        omega_z = self.earth_rotation_rate * np.sin(self.latitude_rad)

        return np.array([omega_x, omega_y, omega_z])

    def compensate_gyro_data(
        self,
        gyro_data: np.ndarray,
        orientation: np.ndarray = None
    ) -> np.ndarray:
        """ジャイロデータから地球の自転の影響を除去

        Args:
            gyro_data: ジャイロデータ（Nx3行列）
            orientation: 現在の姿勢（オイラー角）。Noneの場合、
                        センサーが水平に設置されていると仮定。

        Returns:
            np.ndarray: 補正済みのジャイロデータ
        """
        earth_rotation = self.get_earth_rotation_rates()

        if orientation is not None:
            # 姿勢から回転行列を計算
            roll, pitch, yaw = orientation
            R = self.euler_to_rotation_matrix(roll, pitch, yaw)

            # 地球の自転をセンサー座標系に変換
            earth_rotation = R @ earth_rotation

        # 地球の自転の影響を除去
        compensated_data = gyro_data - earth_rotation

        return compensated_data

    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """オイラー角から回転行列を計算"""
        # 各軸の回転行列
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # ZYX順の回転行列の合成
        R = Rz @ Ry @ Rx
        return R

# 使用例
def earth_rotation_example():
    # 東京の緯度（35.6762° N）での補正例
    compensator = EarthRotationCompensator(latitude=35.6762)

    # 地球の自転角速度の計算
    earth_rates = compensator.get_earth_rotation_rates()
    print("地球の自転角速度（rad/s）:")
    print(f"X軸（北方向）: {earth_rates[0]:.9f}")
    print(f"Y軸（東方向）: {earth_rates[1]:.9f}")
    print(f"Z軸（上方向）: {earth_rates[2]:.9f}")

    # ジャイロデータの補正例
    # ダミーデータを使用
    gyro_data = np.array([
        [0.0001, 0.0, 0.0001],  # rad/s
        [0.0001, 0.0, 0.0001],
        [0.0001, 0.0, 0.0001]
    ])

    # 水平設置の場合
    compensated_data = compensator.compensate_gyro_data(gyro_data)

    # 任意の姿勢の場合
    orientation = np.array([np.pi/6, np.pi/4, np.pi/3])  # オイラー角
    compensated_data_with_orientation = compensator.compensate_gyro_data(
        gyro_data, orientation
    )

    print("\n補正前のジャイロデータ:")
    print(gyro_data[0])
    print("\n補正後のジャイロデータ（水平設置）:")
    print(compensated_data[0])
```

地球の自転の影響を考慮する上での重要なポイント：

1. **緯度依存性**
   - 自転の影響は緯度によって変化
   - 赤道付近では水平成分が最大
   - 極地付近では垂直成分が最大

2. **センサーの姿勢**
   - センサーの姿勢によって地球自転の影響が変化
   - 正確な補正には現在の姿勢を考慮する必要

3. **センサーの精度**
   - 地球の自転角速度は非常に小さい
   - 高精度なセンサーでないと検知が困難
   - ノイズやドリフトの影響を考慮する必要

4. **応用例**
   - 慣性航法システム
   - 高精度な姿勢推定
   - 地球物理学的な測定

## 姿勢推定の基礎

### 座標系
- 慣性座標系（固定座標系）: 地球に固定された座標系
- ボディ座標系（移動座標系）: センサーに固定された座標系

### 姿勢表現方法
1. ZYXオイラー角
   - ヨー角（ψ）: Z軸周りの回転
   - ピッチ角（θ）: Y軸周りの回転
   - ロール角（φ）: X軸周りの回転

2. クォータニオン
   - 4つのパラメータ（q0, q1, q2, q3）で3次元の回転を表現
   - ジンバルロックを回避可能
   - 計算効率が良い

## 姿勢推定の実装

### 1. 加速度センサーによる姿勢推定

加速度センサーは重力加速度を検出できるため、静止時の姿勢推定に使用できます。

```python
import numpy as np
from math import atan2, sqrt

def estimate_attitude_from_accel(ax, ay, az):
    """
    加速度データからオイラー角を推定

    Parameters:
    ax, ay, az: 各軸の加速度 [m/s^2]

    Returns:
    roll, pitch: ロール角、ピッチ角 [rad]
    """
    # ロール角の計算
    roll = atan2(ay, sqrt(ax*ax + az*az))

    # ピッチ角の計算
    pitch = atan2(-ax, sqrt(ay*ay + az*az))

    return roll, pitch

def test_estimate_attitude():
    """テスト用の関数 - 水平姿勢と45度傾いた姿勢をテスト"""
    ax, ay, az = 0.0, 0.0, -9.81
    roll, pitch = estimate_attitude_from_accel(ax, ay, az)
    print(f"水平姿勢: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")
    
    ax, ay, az = 0.0, 9.81 * np.sin(np.pi/4), -9.81 * np.cos(np.pi/4)
    roll, pitch = estimate_attitude_from_accel(ax, ay, az)
    print(f"X軸45度回転: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")

if __name__ == "__main__":
    test_estimate_attitude()

# ヨー角は加速度センサーのみでは求められない
```

### 使用方法

このコードは加速度センサーのデータから姿勢（ロールとピッチ角）を推定します。

1. 関数をインポートします：
   ```python
   from estimate_attitude_from_accel import estimate_attitude_from_accel
   ```

2. 加速度データを使用して姿勢を推定します：
   ```python
   # ax, ay, az: 加速度データ（m/s²）
   roll, pitch = estimate_attitude_from_accel(ax, ay, az)
   
   # 角度を度に変換
   roll_deg = np.degrees(roll)
   pitch_deg = np.degrees(pitch)
   ```

注意点：
- この方法は静止状態または低加速度状態でのみ正確です
- 動的な加速度が存在する場合、推定結果は不正確になります
- 完全な姿勢推定には、ジャイロセンサーや磁気センサーとの組み合わせが必要です

### 2. ジャイロセンサーによる姿勢推定

#### 2.1 オイラー角による実装

```python
import numpy as np
from math import sin, cos, tan

class EulerAttitudeEstimator:
    def __init__(self):
        self.roll = 0.0   # ロール角 [rad]
        self.pitch = 0.0  # ピッチ角 [rad]
        self.yaw = 0.0    # ヨー角 [rad]

    def update(self, wx, wy, wz, dt):
        """
        角速度データからオイラー角を更新

        Parameters:
        wx, wy, wz: 各軸の角速度 [rad/s]
        dt: 時間間隔 [s]
        """
        # オイラー角の変化率
        roll_dot = wx + sin(self.roll) * tan(self.pitch) * wy + \
                   cos(self.roll) * tan(self.pitch) * wz
        pitch_dot = cos(self.roll) * wy - sin(self.roll) * wz
        yaw_dot = sin(self.roll) / cos(self.pitch) * wy + \
                  cos(self.roll) / cos(self.pitch) * wz

        # オイラー角の更新（積分）
        self.roll += roll_dot * dt
        self.pitch += pitch_dot * dt
        self.yaw += yaw_dot * dt
```

### 使用方法

このクラスはジャイロセンサーのデータからオイラー角による姿勢推定を行います。

1. クラスをインポートします：
   ```python
   from eulerattitudeestimator import EulerAttitudeEstimator
   ```

2. 推定器のインスタンスを作成します：
   ```python
   estimator = EulerAttitudeEstimator()
   ```

3. 角速度データを使用して姿勢を更新します：
   ```python
   # wx, wy, wz: 角速度データ（rad/s）
   # dt: 時間間隔（秒）
   estimator.update(wx, wy, wz, dt)
   
   # 推定された姿勢の取得
   roll = estimator.roll
   pitch = estimator.pitch
   yaw = estimator.yaw
   
   # 角度を度に変換
   roll_deg = np.degrees(roll)
   pitch_deg = np.degrees(pitch)
   yaw_deg = np.degrees(yaw)
   ```

注意点：
- ジャイロセンサーのみによる姿勢推定は時間とともにドリフトが発生します
- ピッチ角が±90度に近づくとジンバルロックが発生する可能性があります
- 長時間の使用には加速度センサーとの併用が推奨されます

#### 2.2 クォータニオンによる実装

クォータニオンは以下の形式で表現します：

q = (q0, q1, q2, q3)

ここで：
- q0: スカラー部（実部）
- (q1, q2, q3): ベクトル部（虚部）で回転軸ベクトルを表す

```python
import numpy as np
from math import atan2, asin

class QuaternionAttitudeEstimator:
    def __init__(self):
        # クォータニオン初期化 [q0, q1, q2, q3]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro: np.ndarray, dt: float):
        """
        角速度データからクォータニオンを更新

        Args:
            gyro: 角速度ベクトル [rad/s] (x, y, z)
            dt: 時間間隔 [s]
        """
        # 角速度ベクトルの大きさ
        w_norm = np.linalg.norm(gyro)

        if w_norm > 1e-10:  # ゼロ除算を防ぐ
            # 回転軸の単位ベクトル
            rotation_axis = gyro / w_norm

            # 回転角
            rotation_angle = w_norm * dt

            # 回転のクォータニオン
            dq = np.array([
                np.cos(rotation_angle/2),
                rotation_axis[0] * np.sin(rotation_angle/2),
                rotation_axis[1] * np.sin(rotation_angle/2),
                rotation_axis[2] * np.sin(rotation_angle/2)
            ])

            # クォータニオンの更新（クォータニオン積）
            self.q = self._quaternion_multiply(self.q, dq)

            # 正規化
            self.q = self.q / np.linalg.norm(self.q)

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """クォータニオンの積を計算

        Args:
            q1: 1番目のクォータニオン [q0, q1, q2, q3]
            q2: 2番目のクォータニオン [q0, q1, q2, q3]

        Returns:
            np.ndarray: クォータニオンの積 [q0, q1, q2, q3]
        """
        q10, q11, q12, q13 = q1
        q20, q21, q22, q23 = q2

        return np.array([
            q10*q20 - q11*q21 - q12*q22 - q13*q23,
            q10*q21 + q11*q20 + q12*q23 - q13*q22,
            q10*q22 - q11*q23 + q12*q20 + q13*q21,
            q10*q23 + q11*q22 - q12*q21 + q13*q20
        ])

    def to_euler(self):
        """クォータニオンからオイラー角への変換"""
        q0, q1, q2, q3 = self.q

        # ロール（X軸周り）
        roll = atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))

        # ピッチ（Y軸周り）
        pitch = asin(2*(q0*q2 - q3*q1))

        # ヨー（Z軸周り）
        yaw = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))

        return roll, pitch, yaw
```

### 使用方法

このクラスはジャイロセンサーのデータからクォータニオンによる姿勢推定を行います。

1. クラスをインポートします：
   ```python
   from quaternionattitudeestimator import QuaternionAttitudeEstimator
   ```

2. 推定器のインスタンスを作成します：
   ```python
   estimator = QuaternionAttitudeEstimator()
   ```

3. 角速度データを使用して姿勢を更新します：
   ```python
   # gyro: 角速度ベクトル [wx, wy, wz] (rad/s)
   # dt: 時間間隔（秒）
   gyro_vector = np.array([wx, wy, wz])
   estimator.update(gyro_vector, dt)
   
   # オイラー角の取得
   roll, pitch, yaw = estimator.to_euler()
   
   # 角度を度に変換
   roll_deg = np.degrees(roll)
   pitch_deg = np.degrees(pitch)
   yaw_deg = np.degrees(yaw)
   ```

利点：
- ジンバルロックが発生しない
- 回転の合成が容易
- 数値的に安定している

注意点：
- ジャイロセンサーのみによる姿勢推定は時間とともにドリフトが発生します
- 長時間の使用には加速度センサーとの併用が推奨されます

### 3. 実装例

#### 3.1 仮想IMUを使用した実装

```python
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IMUData:
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

class VirtualIMU:
    def __init__(self, noise_std: float = 0.01, dt: float = 0.01):
        """仮想IMUセンサーの初期化

        Args:
            noise_std: センサーノイズの標準偏差
            dt: シミュレーションの時間間隔
        """
        self.orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.noise_std = noise_std
        self.dt = dt
        self.gravity = np.array([0.0, 0.0, 9.81])

    def get_data(self) -> IMUData:
        """センサーデータの取得

        Returns:
            IMUData: 加速度と角速度のデータ
        """
        # 角速度（ゆっくりとした回転を仮定）
        t = time.time()
        gyro = np.array([
            0.01 * np.sin(t),
            0.01 * np.cos(t),
            0.01 * np.sin(2 * t)
        ]) + np.random.normal(0, self.noise_std, size=3)

        # 姿勢を更新
        self.orientation += gyro * self.dt

        # 加速度（重力+ノイズ）
        acc = self.gravity + np.random.normal(0, self.noise_std, size=3)

        return IMUData(
            acc_x=acc[0], acc_y=acc[1], acc_z=acc[2],
            gyro_x=gyro[0], gyro_y=gyro[1], gyro_z=gyro[2]
        )

# メイン処理
def main_virtual(duration: float = 10.0, dt: float = 0.1):
    """仮想IMUによる姿勢推定のデモ

    Args:
        duration: シミュレーション時間 [秒]
        dt: サンプリング間隔 [秒]
    """
    # センサーと推定器の初期化
    imu = VirtualIMU(noise_std=0.01, dt=dt)
    euler_estimator = EulerAttitudeEstimator()
    quat_estimator = QuaternionAttitudeEstimator()

    # シミュレーションの開始時刻
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            # センサーデータの取得
            imu_data = imu.get_data()

            # ジャイロデータによる姿勢推定
            euler_estimator.update(
                imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z, dt
            )
            gyro_vector = np.array([imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z])
            quat_estimator.update(gyro_vector, dt)

            # 加速度データによる姿勢推定
            roll, pitch = estimate_attitude_from_accel(
                imu_data.acc_x, imu_data.acc_y, imu_data.acc_z
            )

            # 結果の表示
            print("\n=== 仮想IMUによる姿勢推定 ===")
            print(f"Elapsed Time: {time.time() - start_time:.1f}s")
            print("オイラー角による推定:")
            print(f"Roll: {np.degrees(euler_estimator.roll):.1f}°")
            print(f"Pitch: {np.degrees(euler_estimator.pitch):.1f}°")
            print(f"Yaw: {np.degrees(euler_estimator.yaw):.1f}°")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n処理を終了します")
```

#### 3.2 IMUのみによる位置推定

IMUのみを使用した位置推定（IMU Dead Reckoning）は、以下の手順で実現できます：

1. 姿勢推定：クォータニオンまたはEKFで姿勢を推定
2. 重力補正：推定した姿勢を使用して加速度から重力成分を除去
3. 二重積分：重力補正後の加速度を積分して速度と位置を推定

以下に実装例を示します：

```python
import numpy as np
import collections
from madgwickfilter import MadgwickFilter

class IMUPositionEstimator:
    def __init__(self):
        """Madgwickフィルタ + EKFによる位置推定器の初期化"""
        # 状態変数の初期化
        self.position = np.zeros(3)  # [x, y, z]
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.attitude = MadgwickFilter()

        # ノイズ除去用のパラメータ
        self.acc_threshold = 0.05  # 加速度閾値[m/s^2]
        self.vel_threshold = 0.02  # 速度閾値[m/s]

    def reset(self):
        """状態をリセット"""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = MadgwickFilter()

    def update(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """IMUデータから位置を推定

        Args:
            acc: 加速度[m/s^2] [ax, ay, az]
            gyro: 角速度[rad/s] [wx, wy, wz]
            dt: 時間間隔[s]

        Returns:
            np.ndarray: 推定位置[m] [x, y, z]
        """
        # 1. 姿勢の更新
        self.attitude.update(gyro, acc, dt)

        # 2. 重力補正（センサー座標系→グローバル座標系）
        R = self.attitude.get_rotation_matrix()
        acc_global = R @ acc - np.array([0, 0, 9.81])

        # 3. ノイズ除去（静止状態の検出）
        if np.linalg.norm(acc_global) < self.acc_threshold:
            acc_global = np.zeros(3)

        # 4. 速度の更新（第1積分）
        self.velocity += acc_global * dt

        # 5. 速度ドリフトの補正
        if np.linalg.norm(self.velocity) < self.vel_threshold:
            self.velocity = np.zeros(3)

        # 6. 位置の更新（第2積分）
        self.position += self.velocity * dt

        return self.position
```

### 使用方法

このクラスはIMUデータを使用して位置推定を行います。

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

仕組み：
- Madgwickフィルタを使用して姿勢を推定
- 推定した姿勢を使用して加速度から重力成分を除去
- 加速度の二重積分により速度と位置を計算
- 静止状態検出によるドリフト補正

注意点：
- IMUのみによる位置推定は時間とともに誤差が蓄積します
- 長時間の使用には外部参照（GPS、ビジョンなど）との併用が推奨されます
- 高精度なIMUセンサーを使用することで精度が向上します

位置推定の精度を向上させるためのポイント：

1. **高精度IMUの選定**
   - バイアス安定性: < 1°/hr
   - ランダムウォーク: < 0.1°/√hr
   - 加速度計バイアス: < 50μg

2. **キャリブレーションの重要性**
   - 温度補正の実施
   - アライメント誤差の最小化
   - 定期的な再キャリブレーション

3. **ノイズ対策**
   - 適切なローパスフィルタの適用
   - 静止状態の検出と補正
   - 速度ドリフトの補正

使用例：

```python
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
```

### 3.3 CSVファイルからのデータ読み込みと処理

```python
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

        gyro_vector = np.array([
            current_data['gyro_x'],
            current_data['gyro_y'],
            current_data['gyro_z']
        ])
        quat_estimator.update(gyro_vector, dt)

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
```

## センサー性能の分析

### アラン分散（Allan Variance）

アラン分散はセンサーの安定性やノイズ特性を評価するための手法です。異なる時間スケールでのセンサーの安定性を評価できます。

```python
def calculate_allan_variance(data: np.ndarray, fs: float, max_clusters: int = 100) -> tuple:
    """アラン分散を計算

    Args:
        data: センサーデータ
        fs: サンプリング周波数[Hz]
        max_clusters: 最大クラスタ数

    Returns:
        tau: 時間スケール
        avar: アラン分散
    """
    n = len(data)
    # クラスタサイズの設定
    clusters = np.logspace(0, np.log10(n/2), max_clusters).astype(int)
    clusters = np.unique(clusters)

    tau = clusters / fs
    avar = np.zeros(len(clusters))

    for i, k in enumerate(clusters):
        # k個のデータ点で平均を取る
        n_groups = int(n / k)
        if n_groups < 2:
            break

        # データをk個ずつのグループに分割
        groups = np.array_split(data[:n_groups*k], n_groups)
        group_means = np.array([np.mean(group) for group in groups])

        # アラン分散の計算
        avar[i] = np.sum(np.diff(group_means)**2) / (2 * (n_groups-1))

    return tau[:i], np.sqrt(avar[:i])  # アラン偏差を返す

def plot_allan_deviation(tau: np.ndarray, adev: np.ndarray, title: str = ''):
    """アラン偏差をプロット

    Args:
        tau: 時間スケール
        adev: アラン偏差
        title: タイトル
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(tau, adev)
    plt.grid(True)
    plt.xlabel('平均時間 τ [s]')
    if title.lower().find('gyro') != -1:
        plt.ylabel('角速度アラン偏差 σ(τ) [deg/s]')
    elif title.lower().find('acc') != -1:
        plt.ylabel('加速度アラン偏差 σ(τ) [m/s²]')
    else:
        plt.ylabel('アラン偏差 σ(τ)')
    if title:
        plt.title(title)
    plt.show()
```

### アラン分散の読み方

アラン偏差の単位は入力データの単位と同じになりますが、目的に応じて適切な単位に変換して表示します：

- 角速度センサー：
  - 短時間の安定性評価： deg/s または rad/s
  - 長時間のドリフト評価： deg/h または rad/h
- 加速度センサー：
  - 動的特性評価： m/s²
  - 静的特性評価： μg または mg

単位変換例：
```python
# deg/s から deg/h への変換
adev_deg_s = 0.01  # 角速度ノイズ [deg/s]
adev_deg_h = adev_deg_s * 3600  # 3600秒 = 1時間
print(f"角速度ノイズ: {adev_deg_s:.6f} deg/s = {adev_deg_h:.6f} deg/h")

# m/s² から μg への変換
adev_ms2 = 0.001  # 加速度ノイズ [m/s²]
adev_ug = adev_ms2 * 1e6 / 9.81  # 1g = 9.81 m/s²
print(f"加速度ノイズ: {adev_ms2:.6f} m/s² = {adev_ug:.6f} μg")
```

### 使用方法

このコードはIMUセンサーのノイズレベルを異なる単位で表現する方法を示しています。

1. 角速度ノイズの単位変換：
   ```python
   # 角速度ノイズ [deg/s]
   adev_deg_s = 0.01
   
   # deg/s から deg/h への変換
   adev_deg_h = adev_deg_s * 3600  # 3600秒 = 1時間
   ```

2. 加速度ノイズの単位変換：
   ```python
   # 加速度ノイズ [m/s²]
   adev_ms2 = 0.001
   
   # m/s² から μg への変換
   adev_ug = adev_ms2 * 1e6 / 9.81  # 1g = 9.81 m/s²
   ```

単位変換の目的：
- 角速度ノイズは長時間安定性を評価する場合、deg/hで表現すると分かりやすい
- 加速度ノイズは静的特性を評価する場合、μgで表現すると他のセンサーと比較しやすい

### アラン分散とアラン偏差の関係

重要な注意点として、アラン分散（Allan Variance）とアラン偏差（Allan Deviation）の違いを理解する必要があります：

- アラン偏差はアラン分散の平方根です
- 多くの文献ではグラフ化にアラン偏差を使用しています
- そのため、グラフの傾きの解釈が異なることに注意が必要です

### ノイズ特性の詳細

| ノイズ種別 | アラン分散の傾き<br>(log-log) | アラン偏差の傾き<br>(log-log) | 主に現れるτの範囲（目安） | 特徴・意味 |
|---|---|---|---|---|
| **量子化ノイズ**<br>(Quantization Noise) | -2 | -1 | 最も短いτ（~0.01～0.1秒） | デジタル変換時の丸め誤差。非常に高速に観測される |
| **角度ランダムウォーク**<br>(Angle Random Walk) | -1 | -0.5 | 短いτ（~0.1～1秒） | 角速度の短時間ノイズ。IMUの精度を支配する成分 |
| **バイアス不安定性**<br>(Bias Instability) | 0 | 0 | 中間 τ（~1～10秒） | ゆっくり変動するオフセット。グラフの谷に現れる |
| **レートランダムウォーク**<br>(Rate Random Walk) | +1 | +0.5 | 長いτ（~10～100秒以上） | 長時間でのドリフト傾向。IMUがずれる原因 |
| **角加速度ランダムウォーク**<br>(Angular Accel. RW) | +3 | +1.5 | 極めて長いτ（数百秒以上） | 加速度の変化がノイズに与える影響。通常はあまり現れない |

これらの特性を理解することで、センサーの適切な使用方法や必要なフィルタ処理を判断できます。また、センサーの品質評価や比較にも役立ちます。

## 実装における注意点

1. センサーの特性と課題
   - ジャイロセンサーの単純積分ではドリフトが避けられない
   - 加速度センサーは動的な状況での正確な姿勢推定が困難
   - 各センサーの短所を補うフィルタアルゴリズムが必要

2. フィルタアルゴリズムの選択
   - 拡張カルマンフィルタ：確率的アプローチで高精度な推定が可能
   - Madgwickフィルタ：効率的な勾配降下法で安定した推定が可能
   - 相補フィルタ：ハイパス・ローパスフィルタの組み合わせでドリフトとノイズを除去

3. 座標系の考慮
   - オイラー角では特定の姿勢でジンバルロックが発生
   - クォータニオンを使用することで特異点を回避可能

## 今後の展望

近年、IMU技術は急速な進化を遂げています：

1. **マルチセンサーアレイの登場**
   - 複数のセンサーを組み合わせることで高精度化を実現
   - 安価なセンサーでも高精度な測定が可能に

2. **高感度化の進展**
   - 地球の自転を検知できるレベルの高感度センサーが一般にも入手可能に
   - 小型化と低価格化が進み、広く普及

3. **広がる応用範囲**
   - ウェアラブルデバイスやヘルスケア機器への展開
   - スマートホームやIoT機器での活用
   - 自動運転やロボット工学での高度な応用

これらの進展により、IMUはより軌密な動きの検出や高精度な姿勢推定が可能となり、新しいアプリケーションの創出につながっていくことが期待されます。

## 高度な姿勢推定フィルタ

### 1. 拡張カルマンフィルタ（Extended Kalman Filter: EKF）

拡張カルマンフィルタは非線形システムに対応したカルマンフィルタの拡張版です。

```python
class IMU_EKF:
    def __init__(self):
        # 状態ベクトル: [q0, q1, q2, q3, bx, by, bz]
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # 共分散行列
        self.P = np.eye(7) * 0.1
        # プロセスノイズ
        self.Q = np.eye(7) * 0.001
        # 観測ノイズ
        self.R = np.eye(6) * 0.1
        # 重力加速度
        self.gravity = np.array([0, 0, 9.81])

    def predict(self, gyro: np.ndarray, dt: float):
        """予測ステップ"""
        # ジャイロバイアスを考慮した角速度
        wx = gyro[0] - self.x[4]
        wy = gyro[1] - self.x[5]
        wz = gyro[2] - self.x[6]

        # 状態遷移行列の計算
        F = self._compute_state_transition(wx, wy, wz, dt)

        # 状態予測
        self.x[:4] = self._quaternion_update(self.x[:4], wx, wy, wz, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, acc: np.ndarray, mag: np.ndarray = None):
        """更新ステップ"""
        # 観測行列とイノベーションの計算
        H = self._compute_observation_matrix()
        z = self._compute_measurement(acc, mag)
        y = z - self._compute_expected_measurement()

        # カルマンゲインの計算
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状態と共分散の更新
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P

        # クォータニオンの正規化
        q_norm = np.linalg.norm(self.x[:4])
        self.x[:4] /= q_norm

    def _compute_state_transition(self, wx: float, wy: float, wz: float, dt: float) -> np.ndarray:
        """状態遷移行列の計算"""
        q0, q1, q2, q3 = self.x[:4]
        F = np.eye(7)

        # クォータニオンの状態遷移
        F[0:4, 0:4] = np.array([
            [1, -wx*dt/2, -wy*dt/2, -wz*dt/2],
            [wx*dt/2, 1, wz*dt/2, -wy*dt/2],
            [wy*dt/2, -wz*dt/2, 1, wx*dt/2],
            [wz*dt/2, wy*dt/2, -wx*dt/2, 1]
        ])

        # バイアスの状態遷移
        F[0:4, 4:7] = np.array([
            [q1*dt/2, q2*dt/2, q3*dt/2],
            [-q0*dt/2, q3*dt/2, -q2*dt/2],
            [-q3*dt/2, -q0*dt/2, q1*dt/2],
            [q2*dt/2, -q1*dt/2, -q0*dt/2]
        ])

        return F

    def _quaternion_update(self, q: np.ndarray, wx: float, wy: float, wz: float, dt: float) -> np.ndarray:
        """クォータニオンの更新"""
        omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return q + 0.5 * omega @ q * dt

    def _compute_observation_matrix(self) -> np.ndarray:
        """観測行列の計算"""
        q0, q1, q2, q3 = self.x[:4]
        H = np.zeros((6, 7))

        # 加速度に関するヤコビアン
        H[0:3, 0:4] = np.array([
            [2*(q0*q2 - q1*q3), 2*(q1*q2 + q0*q3), 1-2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 + q0*q2), 1-2*(q0**2 + q3**2), 2*(q2*q3 - q0*q1), 2*(q1*q2 - q0*q3)],
            [1-2*(q1**2 + q2**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2), 2*(q2*q3 + q0*q1)]
        ]) * self.gravity[2]

        return H

    def _compute_measurement(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """観測値の計算"""
        z = np.zeros(6)
        z[:3] = acc
        if mag is not None:
            z[3:] = mag
        return z

    def _compute_expected_measurement(self) -> np.ndarray:
        """期待観測値の計算"""
        q0, q1, q2, q3 = self.x[:4]
        h = np.zeros(6)

        # 加速度の期待値
        R = self._quaternion_to_rotation_matrix(q0, q1, q2, q3)
        h[:3] = R @ self.gravity

        return h

    @staticmethod
    def _quaternion_to_rotation_matrix(q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
        """クォータニオンから回転行列への変換"""
        return np.array([
            [1-2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1-2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1-2*(q1**2 + q2**2)]
        ])
```

特徴：
- 確率的なアプローチでノイズを処理
- ジャイロバイアスの推定が可能
- 計算コストが比較的高い

### 使用方法

このクラスは拡張カルマンフィルタを使用してIMUデータから姿勢推定を行います。

1. クラスをインポートします：
   ```python
   from imu_ekf import IMU_EKF
   ```

2. フィルタのインスタンスを作成します：
   ```python
   ekf = IMU_EKF()
   ```

3. IMUデータを使用して姿勢を更新します：
   ```python
   # gyro: 角速度データ [wx, wy, wz] (rad/s)
   # acc: 加速度データ [ax, ay, az] (m/s²)
   # dt: 時間間隔（秒）
   
   # 予測ステップ
   ekf.predict(gyro, dt)
   
   # 更新ステップ
   ekf.update(acc)
   
   # 磁気センサーがある場合
   # ekf.update(acc, mag)
   ```

EKFの仕組み：
- 予測ステップでジャイロデータから姿勢を予測
- 更新ステップで加速度データを使用して予測を修正
- 確率的なアプローチでセンサーノイズを処理
- ジャイロバイアスを自動的に推定して補正

### 2. Madgwickフィルタ

Madgwickフィルタは効率的な勾配降下法に基づくフィルタです。

```python
import numpy as np

class MadgwickFilter:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # q = [q0, q1, q2, q3]

    def update(self, gyro: np.ndarray, acc: np.ndarray, dt: float):
        """Madgwickフィルタの更新"""
        if np.linalg.norm(acc) == 0:
            print("警告: accのノルムがゼロです。更新をスキップします。")
            return

        acc = acc / np.linalg.norm(acc)

        q0, q1, q2, q3 = self.q

        # 目的関数 f とそのヤコビアン J による動的な調整
        f = np.array([
            2*(q1*q3 - q0*q2) - acc[0],
            2*(q0*q1 + q2*q3) - acc[1],
            2*(0.5 - q1*q1 - q2*q2) - acc[2]
        ])
        J = np.array([
            [-2*q2,  2*q3, -2*q0, 2*q1],
            [ 2*q1,  2*q0,  2*q3, 2*q2],
            [   0, -4*q1, -4*q2,    0]
        ])
        grad = J.T @ f
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm

        # クォータニオンの微分（ジャイロスコープ + 動的な動きの調整）
        qDot = 0.5 * np.array([
            -q1 * gyro[0] - q2 * gyro[1] - q3 * gyro[2],
             q0 * gyro[0] + q2 * gyro[2] - q3 * gyro[1],
             q0 * gyro[1] - q1 * gyro[2] + q3 * gyro[0],
             q0 * gyro[2] + q1 * gyro[1] - q2 * gyro[0]
        ]) - self.beta * grad

        # クォータニオンの積分と正規化
        self.q += qDot * dt
        self.q /= np.linalg.norm(self.q)

    def get_euler_angles(self) -> np.ndarray:
        """クォータニオンからオイラー角（roll, pitch, yaw）を取得"""
        q0, q1, q2, q3 = self.q

        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
        pitch = np.arcsin(2*(q0*q2 - q3*q1))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))

        return np.array([roll, pitch, yaw])
        
    def get_rotation_matrix(self) -> np.ndarray:
        """クォータニオンから回転行列を取得"""
        q0, q1, q2, q3 = self.q
        
        return np.array([
            [1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2)]
        ])
```

### 使用方法

このクラスはMadgwickフィルタを使用してIMUデータから姿勢推定を行います。

1. クラスをインポートします：
   ```python
   from madgwickfilter import MadgwickFilter
   ```

2. フィルタのインスタンスを作成します：
   ```python
   # 標準的なパラメータ設定
   madgwick = MadgwickFilter(beta=0.1)
   ```

3. IMUデータを使用して姿勢を更新します：
   ```python
   # gyro: 角速度データ [wx, wy, wz] (rad/s)
   # acc: 加速度データ [ax, ay, az] (m/s²)
   # dt: 時間間隔（秒）
   madgwick.update(gyro, acc, dt)
   
   # オイラー角の取得
   roll, pitch, yaw = madgwick.get_euler_angles()
   
   # 回転行列の取得
   R = madgwick.get_rotation_matrix()
   ```

Madgwickフィルタの仕組み：
- 勾配降下法を使用して最適な姿勢を推定
- 加速度データから重力方向を推定
- 効率的なアルゴリズムで計算コストを抑制
- betaパラメータで収束速度とノイズ耐性のバランスを調整

### 3. フィルタパラメータの設定と影響

#### 3.1 相補フィルタのカットオフ周波数

相補フィルタのカットオフ周波数（fc）は、ジャイロと加速度計のデータの統合方法を制御する重要なパラメータです：

- **高い周波数（fc > 1Hz）**
  - ジャイロの影響が大きくなる
  - 短期的な動きの追従性が向上
  - ドリフトの影響を受けやすい
  - 用途：VR/ARやゲームなどのリアルタイムアプリケーション

- **低い周波数（fc < 0.1Hz）**
  - 加速度計の影響が大きくなる
  - 長期的な安定性が向上
  - 応答性が低下
  - 用途：ナビゲーションや姿勢推定の長期安定性が必要な場合

推奨値の例：
```python
# 一般的な用途
fc = 0.1  # Hz

# 高応答性が必要な場合
fc = 1.0  # Hz

# 長期安定性重視
fc = 0.05  # Hz
```

#### 3.2 適応フィルタリングの効果

適応フィルタリングを導入することで、以下のような利点が得られます：

- **動的なパラメータ調整**
  - 加速度の変動が大きい場合はジャイロを重視
  - 安定している場合は加速度計を重視

- **自動パラメータチューニング**
  - センサーの状態に応じて最適なパラメータを選択
  - 手動でのパラメータ調整が不要

#### 3.3 Madgwickフィルタのbetaパラメータ

Madgwickフィルタのbetaパラメータは、ジャイロと加速度計のバランスを調整する重要なパラメータです：

- **大きい値（β > 0.5）**
  - 加速度計の影響が強くなる
  - 急な動きに対する追従性が向上
  - ノイズの影響を受けやすくなる
  - 用途：ドローンやロボットの急な動きの追従

- **小さい値（β < 0.1）**
  - ジャイロの影響が強くなる
  - より滑らかな姿勢推定が可能
  - 姿勢の収束が遅くなる
  - 用途：センサーのノイズが多い環境での安定化

推奨値の例：
```python
# 一般的な用途
beta = 0.1

# 高速な動きの追従
beta = 0.3
# 高応答性が必要な場合
fc = 1.0  # Hz

# 長期安定性重視
fc = 0.05  # Hz
```

```python
class ComplementaryFilter:
    def __init__(self, fc=0.1, adaptive=False):  # fc: カットオフ周波数[Hz]
        self.fc = fc  # 基本のカットオフ周波数
        self.adaptive = adaptive  # 適応フィルタリングの有効/無効

        # 状態変数
        self.prev_angle = np.zeros(3)  # [roll, pitch, yaw]
        self.prev_gyro = np.zeros(3)
        self.prev_acc_angle = np.zeros(2)  # [roll, pitch]

        # 適応フィルタリング用パラメータ
        self.acc_buffer = collections.deque(maxlen=10)
        self.min_fc = 0.05  # 最小カットオフ周波数
        self.max_fc = 1.0   # 最大カットオフ周波数
        self.acc_threshold = 0.5  # 加速度の閾値[m/s^2]
        self.first_update = True

    def update(self, gyro: np.ndarray, acc: np.ndarray, dt: float):
        """フィルタ更新（適応フィルタリング対応）

        Args:
            gyro: 角速度[rad/s] [wx, wy, wz]
            acc: 加速度[m/s^2] [ax, ay, az]
            dt: 時間間隔[s]
        """
        if self.first_update:
            self.prev_gyro = gyro
            self.first_update = False
            return

        # 加速度の正規化と角度計算
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 0:
            acc_normalized = acc / acc_norm
        else:
            acc_normalized = acc

        roll_acc = np.arctan2(acc_normalized[1], acc_normalized[2])
        pitch_acc = np.arctan2(-acc_normalized[0],
                              np.sqrt(acc_normalized[1]**2 + acc_normalized[2]**2))
        
        # ジャイロによる角度の更新
        gyro_angle = self.prev_angle + gyro * dt

        # 適応フィルタリングの場合、カットオフ周波数を動的に調整
        if self.adaptive:
            self.acc_buffer.append(acc_norm)
            if len(self.acc_buffer) == self.acc_buffer.maxlen:
                acc_std = np.std(list(self.acc_buffer))
                # 加速度の変動が大きい場合はジャイロを重視（fcを下げる）
                if acc_std > self.acc_threshold:
                    current_fc = self.min_fc
                else:
                    # 加速度が安定している場合は加速度計を重視（fcを上げる）
                    current_fc = self.max_fc
            else:
                current_fc = self.fc
        else:
            current_fc = self.fc

        # 相補フィルタの係数を計算
        alpha = dt / (dt + 1/(2*np.pi*current_fc))

        # ジャイロの積分（台形則）
        roll_gyro = self.prev_angle[0] + (gyro[0] + self.prev_gyro[0])*dt/2
        pitch_gyro = self.prev_angle[1] + (gyro[1] + self.prev_gyro[1])*dt/2
        yaw_gyro = self.prev_angle[2] + (gyro[2] + self.prev_gyro[2])*dt/2

        # 相補フィルタでroll, pitchを計算
        roll = (1 - alpha)*roll_gyro + alpha*roll_acc
        pitch = (1 - alpha)*pitch_gyro + alpha*pitch_acc

        # yawはジャイロの積分のみ（磁気センサーがない場合）
        yaw = yaw_gyro

        # 状態の更新
        self.prev_angle = np.array([roll, pitch, yaw])
        self.prev_gyro = gyro

        return self.prev_angle
```

特徴：
- 実装が非常に簡単
- 計算コストが低い
- 高度なノイズ処理は難しい

### 使用方法

このクラスは相補フィルタを使用してIMUデータから姿勢推定を行います。

1. クラスをインポートします：
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

相補フィルタの仕組み：
- 加速度センサーからの姿勢推定（低周波成分）とジャイロセンサーからの姿勢推定（高周波成分）を組み合わせます
- カットオフ周波数（`fc`）で両者の重みを調整します
- 適応型フィルタ（`adaptive=True`）では、加速度の信頼性に応じて動的に重みを調整します

### 各フィルタの比較

1. **精度**
   - EKF: 最も高精度（適切なモデル化が必要）
   - Madgwick: 高精度で安定性が高い
   - 相補: 中程度の精度

2. **計算コスト**
   - EKF: 高い（行列演算が多い）
   - Madgwick: 中程度
   - 相補: 低い（単純な線形結合）

3. **パラメータ調整**
   - EKF: 複数のパラメータ調整が必要
   - Madgwick: 単一パラメータ（beta）
   - 相補: 単一パラメータ（alpha）

4. **利点・注意点**
   - EKF
     - バイアス推定が可能
     - モデル化が重要
     - 初期化が重要
   - Madgwick
     - 確率モデル不要
     - 確実な収束
     - 磨気の影響を考慮可能
   - 相補
     - 実装が簡単
     - 直感的な調整
     - 加速度ノイズに弱い

### 3.4 高度な応用：IMUのみによる位置推定

IMUのみを使用した位置推定（IMU Dead Reckoning）では、以下の2つのフィルタを組み合わせることで高精度な推定が可能です：

1. **姿勢推定: Madgwickフィルタ**
   - 計算効率が良く、安定した姿勢推定が可能
   - ジャイロバイアスの自動補正機能を内蔵

2. **位置・速度推定: 拡張カルマンフィルタ**
   - 状態の不確実性を適切に考慮
   - ZUPT（Zero-velocity Update）によるドリフト補正
   - 加速度計バイアスのオンライン推定

以下に実装例を示します：

```python
class IMUPositionEstimator:
    def __init__(self):
        """Madgwickフィルタ + EKFによる位置推定器の初期化"""
        # Madgwickフィルタ（姿勢推定用）
        self.attitude = MadgwickFilter(beta=0.1)

        # EKFの状態変数 [x, y, z, vx, vy, vz, ax_bias, ay_bias, az_bias]
        self.x = np.zeros(9)
        self.P = np.eye(9)  # 共分散行列

        # システムノイズ
        self.Q = np.diag([
            0.01, 0.01, 0.01,  # 位置ノイズ
            0.1, 0.1, 0.1,      # 速度ノイズ
            0.001, 0.001, 0.001  # バイアスノイズ
        ])

        # ZUPT用パラメータ
        self.R_zupt = np.eye(3) * 0.01  # ZUPT観測ノイズ
        self.acc_buffer = collections.deque(maxlen=10)
        self.zupt_threshold = 0.1  # [m/s^2]

    def reset(self):
        """状態をリセット"""
        self.x = np.zeros(9)
        self.P = np.eye(9)
        self.attitude.reset()
        self.acc_buffer.clear()

    def update(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """IMUデータから位置を推定

        Args:
            acc: 加速度[m/s^2] [ax, ay, az]
            gyro: 角速度[rad/s] [wx, wy, wz]
            dt: 時間間隔[s]

        Returns:
            np.ndarray: 推定位置[m] [x, y, z]
        """
        # 1. Madgwickフィルタによる姿勢推定
        self.attitude.update(gyro, acc, dt)
        R = self.attitude.get_rotation_matrix()

        # 2. 加速度の座標変換とバイアス補正
        acc_global = R @ acc - np.array([0, 0, 9.81])
        acc_global -= self.x[6:9]  # バイアス補正

        # 3. EKFの状態予測ステップ
        # システム行列
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -np.eye(3) * dt

        # 状態予測
        self.x[0:3] += self.x[3:6] * dt + 0.5 * acc_global * dt**2
        self.x[3:6] += acc_global * dt

        # 共分散予測
        self.P = F @ self.P @ F.T + self.Q * dt

        # 4. ZUPT更新
        self.acc_buffer.append(np.linalg.norm(acc))
        if len(self.acc_buffer) == self.acc_buffer.maxlen:
            if self._detect_zupt():
                self._apply_zupt()

        return self.x[0:3]

    def _detect_zupt(self) -> bool:
        """静止状態の検出"""
        acc_std = np.std(list(self.acc_buffer))
        return acc_std < self.zupt_threshold

    def _apply_zupt(self):
        """静止状態での速度補正"""
        H = np.zeros((3, 9))
        H[:, 3:6] = np.eye(3)  # 速度の観測

        # カルマンゲイン
        S = H @ self.P @ H.T + self.R_zupt
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状態と共分散の更新
        self.x = self.x - K @ self.x[3:6]  # 速度をゼロに補正
        self.P = (np.eye(9) - K @ H) @ self.P````
```

位置推定の精度を向上させるためのポイント：

1. **高精度IMUの選定**
   - バイアス安定性: < 1°/hr
   - ランダムウォーク: < 0.1°/√hr
   - 加速度計バイアス: < 50μg

2. **キャリブレーションの重要性**
   - 温度補正の実施
   - アライメント誤差の最小化
   - 定期的な再キャリブレーション

3. **ノイズ対策**
   - 適切なローパスフィルタの適用
   - 静止状態の検出と補正
   - 速度ドリフトの補正

## まとめと今後の展望

### 現状の到達点

IMUは現代のセンサーテクノロジーの中で最も重要なデバイスの一つです。正しい理解と適切な使用方法を習得することで、様々な革新的なアプリケーションを開発することができます。特に姿勢推定と位置推定においては、用途に応じて適切な手法とフィルタリングを選択することが重要です。

### 技術革新と将来展望

近年、IMU技術は大きな革新を遂げており、以下のような発展が見られます：

1. **マルチセンサーアレイの登場**
   - 複数の安価なセンサーを組み合わせることで高精度化を実現
   - 地球の自転を検知できるレベルの高感度化が一般にも利用可能に

2. **広がる応用分野**
   - ウェアラブルデバイスやヘルスケア機器への展開
   - スマートホームやIoT機器での活用
   - 自動運転やロボット工学での高度な応用

これらの進展により、IMUはより精密な動きの検出や高精度な姿勢推定が可能となり、新しいアプリケーションの創出につながっていくことが期待されます。

## 参考文献

1. ["An efficient orientation filter for inertial and inertial/magnetic sensor arrays"](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/) - Sebastian O.H. Madgwick (2010)
   - 論文と実装コードが公開されています

2. ["Strapdown Inertial Navigation Technology, 3rd Edition"](https://digital-library.theiet.org/doi/book/10.1049/pbra017e) - D. H. Titterton and J. L. Weston, IET (2014)
   - IMUの基礎理論から実装までを網羅した標準的な教科書です

3. ["Fusion: A MARG Orientation Filter for Inertial, Magnetic, and GPS Sensors"](https://github.com/xioTechnologies/Fusion) - x-io Technologies
   - Madgwickフィルターの最新実装です

4. ["ROS IMU Tools"](http://wiki.ros.org/imu_filter_madgwick) - ROS Wiki
   - ROSでの実装例とドキュメントです
