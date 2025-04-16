import numpy as np
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
                self.prev_gyro[i] + gyro[i] - self.prev_gyro[i]
            )

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
