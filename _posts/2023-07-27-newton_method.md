---
title:  "Optimization: Newton's method"
excerpt: "최적화의 기본원리, how computers find solutions"

categories:
  - Math/Statistics
tags:
  - Newton's method
  - optimization
  - 최적화
  - calculus

use_math: true
comments: true

last_modified_at: 2023-07-27T08:06:00-05:00
---

미적분학을 활용한 최적화 이론중 가장 기본이 되는 **Newton's method**를 정리해본다. 

여기에서의 최적화란 주어진 수식이 있을 때 수식의 해(solution)을 찾아나가는 것을 의미한다. 이를 응용하면 함수의 국지적(local) 최소값이나 최대값을 찾는 데에도 활용할 수 있다. 

우리가 고등학교 과정이나 기초 미적분학 수업에서 접하는 문제들은 인수분해 등을 통해 손으로 풀어서 solution을 찾을 수 있는 것들인데, 수식이 복잡하거나 딱 떨어지는 해가 없는 경우에도 Newton's method를 통해 근사적인 방법으로 solution을 찾을 수 있다.

또한 Newon's Method는 컴퓨터가 방정식의 solution을 찾는 방식과도 연결된다. 컴퓨터 속 반도체가 우리가 수식을 손으로 푸는 것처럼 풀어서 solution을 내놓는 것이 아니라 여러번의 연산을 통해 solution에 근접한 값을 찾아 내놓는 것이다. 

아래에서는 Newton's method의 개념을 1변수(single variable) 함수의 solution을 찾는 과정부터 다변수(multi variables)함수의 solution을 찾는 과정에 걸쳐 정리해본다.


## 1. 단일변수 함수 f(x) = 0의 해를 찾는 과정    

아래 그림과 같은 함수 f(x)에 대해 f(x) = 0을 만족하는 x값 solution을 구한다고 생각하자.

![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/assets/images/newton_method/newton method.jpg?raw=true)
<br>
(출처: Thomas' Calculus 12E)

계산을 매우 잘하는 컴퓨터가 있다면 x에 무수히 많은 값을 넣어보고 함수값이 0에 제일 가까운 x를 solution이라고 제시할 것이다. 근데 함수가 매우 복잡한 형태를 갖고 있다면?? 혹은 좀 더 체계적인 방법으로 solution을 찾아 나갈 수는 없을까?? 하는 생각이 든다.

이에 대해 아래와 같은 과정을 생각해 볼 수 있다.


(1) 초기값 $x_{0}$에 대해 f($x_{0}$)을 계산해본다. 당연히 임의의 $x_{0}$에 대해 f($x_{0}$)값은 높은 확률로 0과 다를 것이다.

(2) 위의 그림에서처럼 ($x_{0}$, f($x_{0}$))을 지나면서 f(x)에 접하는 접선을 구하고 이 접선이 x축과 만나는 절편을 $x_{1}$로 두자.

위의 접선은 y = f($x_{0}$) + f'(x)(x - $x_{0}$) 로 표현할 수 있고 이 접선의 해, 즉 x절편은 $x_{1}$은 아래와 같다. 

$$
x_{0} - \frac{f(x_{0})}{f'(x_{0})}
$$

그림에서 볼 수 있듯이 $x_{1}$은 함수의 해(그림에서 Root sought)에 조금 더 다가간 것을 확인할 수 있다. 그렇다면 이 과정을 계속해서 반복하면 반복횟수가 커질수록 $x_{n}$은 solution에 가까워질 것이고 위 그림이 그 과정을 보여준다.

(3) 위에서 설명한 근사작업을 계속하면 $x_{n+1}$ = $x_{n}$ - $\frac{f(x_{n})}{f'(x_{n})}$


$x_{1}$, $x_{2}$ ... 와 같은 확률변수가 $\textit{iid}$ 이고, $x_{i}$의 평균이 유한하다고 가정할 때, 대수의 법칙을 한문장으로 표현하면 **'표본의 평균(sample mean)은 모집단 평균(population mean)에 확률적으로 수렴한다.'**와 같다.

이를 수식으로 표현하면 아래와 같다. 

$$
\bar{x} = \frac{1}{n}\sum^n_{i=1}(x_i)  \overset{p}{\to}  \textit{E}(x_i) = \mu
$$

여기에서 **확률적 수렴**이라는 개념이 나오는데 이는 일반적인 수열의 수렴과 어떻게 다른 것일까?

non-random한 수열이 수렴한다는 것은 그 수열 자체가 어떤 **상수**에 수렴한다는 의미이다. 반면에 random한 **확률변수**의 수열의 **확률적 수렴**은 어떤 확률변수가 특정한 상수(c)에 수렴할 **확률이 1에 가까워진다**는 의미로서 수식으로는 아래와 같다. 

$$
임의의\; 양수\; \epsilon, 확률변수\; y_{n}에 대해, \lim_{n \to \infty}P(|y_{n} - c| > \epsilon) = 0  
$$

즉, **대수의 법칙**은 확률변수인 **표본의 평균**이 미지의 **모집단 평균**이라는 상수($\mu$) 에 수렴할 확률이 표본의 크기인 n이 커짐에 따라 1에 수렴한다는 것이다. 

이에 대한 증명은 Markov Inequality, Chebyshev Inequality을 이용하는데, 위에 언급한 [Quant Econ](https://python.quantecon.org/lln_clt.html)에 잘 소개되어있다.

이제 이를 python을 통해 그래프로 묘사해본다. 
<br>

```python
# Module import
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (11, 5)
import random
import numpy as np
from scipy.stats import t, beta, lognorm, expon, gamma, uniform, cauchy
from scipy.stats import gaussian_kde, poisson, binom, norm, chi2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from scipy.linalg import inv, sqrtm

# 
```
pyupbit에서 다운받을 수 있는 가격 ticker는 다음과 같다. 
```python
pyupbit.get_tickers(fiat = 'USD')

>>> ['USDT-BTC', 'USDT-ETH', 'USDT-LTC', 'USDT-XRP', 'USDT-ETC', 'USDT-OMG', 'USDT-ADA', 'USDT-TUSD', 'USDT-SC', 'USDT-TRX', 'USDT-BCH', 'USDT-DGB', 'USDT-DOGE', 'USDT-ZRX', 'USDT-RVN', 'USDT-BAT']
```

여기에서는 The Decentralized Financial Crisis: Attacking DeFi 논문의 replication을 참고하여 2018.1.1 ~ 2021.12.31일의 ETH-USDT 종가 data를 다운받는다. 

```python
# 함수에 넣을 arguments 세팅
ticker = 'USDT-ETH'
interval = 'day1'
to = '2021-12-31'
from_ = '2018-01-01'
count = int(str(pd.to_datetime(to) - pd.to_datetime(from_)).split(' ')[0])

# eth_data에 해당기간 자료를 저장
# get_ohlcv를 통해 일별 가격정보(open, high, low, close), 거래량(volume) 등의 정보를 저장
eth_data = pyupbit.get_ohlcv(ticker = ticker, interval = interval, to = to, count = count)
```

```python
eth_data.head()

                         open        high  ...        volume         value
2017-12-07 09:00:00  415.0000  427.000000  ...  49142.959116  2.005423e+07
2017-12-08 09:00:00  410.0000  452.565702  ...  39610.775920  1.679753e+07
2017-12-09 09:00:00  440.0000  486.860000  ...  42490.245612  1.969863e+07
2017-12-10 09:00:00  461.0099  461.450722  ...  25390.988800  1.102167e+07
2017-12-11 09:00:00  431.0000  510.100000  ...  30042.320160  1.415567e+07
```

분석을 위해 eth_data dataframe중 종가(close) 정보만 남기고 날짜를 'yyyy-mm-dd' 형태로 정리한다.

```python
# eth_data의 index(날짜)와 종가를 열로 하는 dataframe 생성
eth_price = eth_data.reset_index(drop = False)[['index', 'close']]

# column 이름 변경
eth_price.rename(columns = {'index':'date', 'close':'close'}, inplace = True)

# 시-분-초까지 표현되어 있는 date를 'yyyy-mm-dd' 형식으로 변경
eth_price['date'] = pd.to_datetime(eth_price['date']).dt.date

# 원하는 자료 구간으로 자르기, get_ohlcv에서 count는 영업일 기준이기 때문에 우리가 원하는 기간 시작점과 차이가 있음
date_idx = int(np.where(eth_price['date'] == pd.to_datetime(from_))[0])
eth_price = eth_price.loc[date_idx: ]
eth_price = eth_price.reset_index(drop = True)

eth_price.head()

          date       close
0  2018-01-01  751.911550
1  2018-01-02  857.550000
2  2018-01-03  941.574258
3  2018-01-04  936.729524
4  2018-01-05  969.802500
```

해당 기간(2018.1.1 ~ 2021.12.31)의 이더리움 가격을 그래프로 표현해보자.
```python
# 그래프 그리기
plt.figure(figsize= (15, 10))
plt.plot(eth_price['date'], eth_price['close'])
plt.xlabel('Date')
plt.ylabel('ETH/USD')
plt.title('ETH/USD price')
plt.show()
```
![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/assets/images/eth_price.jpg?raw=true)

<br>
## 3. Geometric Brownian Motion 시뮬레이션 

위에서 구한 이더리움 가격 data를 바탕으로 Geometric Brownian Motion 시뮬레이션을 구현해본다.  
<br>
앞서 소개한 Geometric Brownian Motion 수식을 다시 써보면 아래와 같다. 

$$
S_{t} = S_{0}\textit{e}^{(\mu -\frac{1}{2}\sigma ^{2})t+\sigma \textit{W}_{t}}\;\;\;(2)\\
$$  

이더리움 가격에 대한 시뮬레이션은 위 (2)식의 변수에 값을 대입하여 $S_{0}$으로부터 $S_{1}$, $S_{2}$ ...  $S_{t}$ 
 값들을 계산해 나가는 과정이다.  

이때 초기값($\textit{S}_{0}$)은 2020.2.7일의 이더리움 가격이며, $\mu$는 2018.1.1 ~ 2021.12.31 기간 중 이더리움 일일 수익률의 평균값, $\sigma$는 일일 수익률의 표준편차를 이용한다.  

Brownian motion 항($W_{t}$)은 각 k 시점마다의 충격, $b_{k} \sim N(0, k)$ 의 합 즉, $W_{t} = \sum_{k=1}^{t}b_{k}$ 으로 주어지며, 여기에서는 이더리움의 일별 가격자료를 활용하므로 $k$ = 1로 둔다.  

실제 계산에서는 (2)식의 $e^{(\mu -\frac{1}{2}\sigma ^{2})t+\sigma W_{t}}$ 부분을 drift 항($e^{(\mu -\frac{1}{2}\sigma ^{2}t)}$)과  diffusion 항($e^{\sigma W_{t}}$)으로 분리하여 $S_{0}$에 곱하는 식으로 $S_{t}$을 구한다.
<br>

이제 simulation에 필요한 변수들을 아래와 같이 설정한다. 

```python
## Simulation 변수 설정

# dt: 이더리움의 일별 가격을 이용하기 때문에 시간 간격을 1로 설정)
dt = 1

# T: 예측기간, 여기에서는 100일 뒤까지의 가격 경로를 시뮬레이션
T = 100

# N 
N = T / dt

# n: simulation path의 수
n = 100

# s0: 앞서 구한 이더리움 가격의 최종값을 시뮬레이션 초기 가격으로 설정
s0 = eth_price.loc[eth_price.shape[0] - 1, 'close']

# mu: 이더리움 데이터로부터의 역사적 일별수익률의 평균
returns = (eth_price.loc[1:, 'close'] - eth_price.shift(1).loc[1:, 'close']) / eth_price.shift(1).loc[1:, 'close']

mu = np.mean(returns)

print(mu)
>>> mu
0.003249842232051855

# sigma: mu의 표준편차
sigma = np.std(returns)

print(sigma)
>>> sigma
0.065230344178387
```

아래에서는 GBM(Geometric Brownian Motion) 클래스안에 simulate 함수를 정의한다.
```python
# GBM(Geometric Brownian Motion) 클래스 정의
class GBM:

    def simulate(self):
        while(self.T > 0):
            # drift 항 계산
            drift = (self.mu - (0.5 * self.sigma**2)) * (T - self.T + 1)
            # diffusion 항 계산
            diffusion = self.sigma * self.b.cumsum()[T - self.T]
            # 새롭게 계산된 St 값을 self.prices 변수에 추가
            self.prices.append(self.current_price * np.exp(drift + diffusion))
            # loop 마다 1일이 지남을 반영 
            self.T -= self.dt


    def __init__(self, s0, mu, sigma, dt, T):
        # 초기 변수들
        self.current_price = s0
        self.mu = mu
        self.sigma = sigma
        self.b = np.random.normal(0, dt, int(N))
        self.dt = dt
        self.T = T
        self.prices = []
        # simulate 함수를 실행
        self.simulate()   
```

위에서 설정한 simulation에 필요한 변수를 GBM 클래스에 대입하여 $S_{1}$, $S_{2}$ ... $S_{T}$을 계산하고, 이를 그래프로 표현한다.
 
``` python
# simulation 결과를 저장할 비어있는 list 생성
simulations = []

# random.seed 설정(코드 반복 수행시 동일한 random값을 얻기 위해서)
np.random.seed(100)

# simulation 수행 및 그래프 그리기(총 n번의 simulation 수행)
plt.figure(figsize=(20, 10))
for i in range(0, n):
    simulations.append(GBM(s0, mu, sigma, dt, T))

for sim in simulations:
    plt.plot(np.arange(0, len(sim.prices)), sim.prices)
    plt.ylabel('Price evolution')
    plt.xlabel('Time steps(days)')
    plt.title("ETH/USD price Monte Carlo Simulation")

plt.show()
```
![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/assets/images/simulation.jpg?raw=true)