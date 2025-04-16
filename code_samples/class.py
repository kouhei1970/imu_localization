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
if __name__ == "__main__":
    calibration_example()
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
