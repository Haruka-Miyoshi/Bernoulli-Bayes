import numpy as np

# いつの日かくるはずの勉強会用の資料
# ベルヌーイ分布の学習と予測
class Bernoulli:
    def __init__(self, alpha=1.0, beta=1.0):
        # ハイパーパラメータ
        self.__alpha = alpha
        self.__beta = beta
        # 事後分布のパラメータを格納変数
        self.__alpha_hat = 1.0
        self.__beta_hat = 1.0
        # 尤もらしいパラメータを格納変数
        self.__mu = 0
        # 対数が0になるのを回避する用の値
        self.__eps = 1e-7
        
    # 事後分布の計算
    def update(self, data):
        # パラメータ
        mu = np.arange(0,100,0.01)
        # 対数尤度
        llhs = []
        # データ数
        N = len(data)
        
        # 事後分布のパラメータ
        self.__alpha_hat = self.__alpha + np.sum(data)
        self.__beta_hat = self.__beta + N - np.sum(data)

        # パラメータの学習
        for i in range(0, 100):
            llh = (self.__alpha_hat - 1) * np.log(mu[i] + self.__eps) + (self.__beta_hat - 1) * np.log(1 - mu[i] + self.__eps)
            llhs.append(llh)
        
        # 最大対数尤度のmuパラメータ
        mu_max = mu[int(np.argmax(llhs))]
        self.__mu = mu_max
        return mu_max
    
    # xの予測確率p(x)
    def predict(self, x):
        mu = self.__alpha_hat / (self.__alpha_hat + self.__beta_hat)
        prob_x = (mu) ** (x)
        prob_x *= (1-mu) ** (1-x)
        return prob_x

# 実行文
if __name__ == '__main__':
    # データを読み込む
    data = np.loadtxt('./data/bernoulli.txt')
    # インスタンス生成
    bern = Bernoulli()
    # 学習・推論
    mu = bern.update(data)
    # 未観測xに対する予測分布p(x|X)
    prob_zero = bern.predict(0)
    prob_one = bern.predict(1)

    # 推論したパラメータ
    print(f'推論したパラメータμ={mu}')
    print(f'真のパラメータμ={0.25}')
    # 予測確率
    print(f'x=0である確率p(x=0)={prob_zero}')
    print(f'x=1である確率p(x=1)={prob_one}')