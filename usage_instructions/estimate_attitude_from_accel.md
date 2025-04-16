# estimate_attitude_from_accel.py の使用方法

このサンプルコードは、加速度センサーのデータから姿勢（ロールとピッチ角）を推定する関数を提供します。

## 必要なライブラリ

```
pip install numpy
```

## 使用方法

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

## 仕組み

この関数は、重力加速度ベクトルの方向から姿勢を推定します：
- ロール角（X軸周りの回転）は、Y軸と Z軸の加速度から計算されます
- ピッチ角（Y軸周りの回転）は、X軸と Z軸の加速度から計算されます
- ヨー角（Z軸周りの回転）は加速度センサーのみでは求められません

## 注意点

- この方法は静止状態または低加速度状態でのみ正確です
- 動的な加速度が存在する場合、推定結果は不正確になります
- 完全な姿勢推定には、ジャイロセンサーや磁気センサーとの組み合わせが必要です

## テスト関数

コードには、水平姿勢と45度傾いた姿勢をテストする関数が含まれています：
```python
python estimate_attitude_from_accel.py
```
を実行すると、テスト結果が表示されます。
