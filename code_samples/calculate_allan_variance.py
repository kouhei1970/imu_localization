import numpy as np
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
