---
title:  "Geometric Brownian Motion 시뮬레이션(이더리움(ETH)에 대해)"
excerpt: "Geometric Brownian Motion 시리즈 2"

categories:
  - 프로젝트
tags:
  - brownian motion
  - 이더리움
  - simulation
  - 파이썬

use_math: true
comments: true

last_modified_at: 2022-01-04T08:06:00-05:00
---

이번 포스트에서는 Geometric Brownian Motion을 이용하여 이더리움의 미래 가격경로에 대한 시뮬레이션을 수행해본다.  


## 1. Geometric Browniam Motion    

우선 Geometric Brownian Motion의 수식은 아래와 같다.

$$
\textit{ds} = \mu \textit{dt}+\sigma \textit{dW}\;\;\;(1)\\
$$

$$
\cdot \textit{ds}: Change\; in\; asset\; prices\; in\; very\; short\; time\; period\\
\cdot \textit{S}: Asset\;price\;for\;the\;initial\;period\\
\cdot \mu: Expected\;return\;for\;the\;time\;period\;or\;the\;drift\\
\cdot \textit{dt}: Change\;in\;time\\     
\cdot \sigma: Volatility\; of\; asset\; price\\
\cdot \textit{dW}: Change\;in\;Brownian\; motion\;term\\
$$

(1) 수식은 짧은 시간 간격 동안의 자산가격 변화가 짧은 시간 간격(dt) 동안의 평균적인 수익률(drift)과 랜덤한 충격에 의한 변동분의 합으로 나타남을 의미한다.

(1) 수식에 Ito formula(다른 포스트에서 추가 설명 예정)를 적용하여 SDE(Stochastic Differential Equation)를 풀면 아래와 같은 수식을 유도할 수 있다.

$$
S_{t} = S_{0}\textit{e}^{(\mu -\frac{1}{2}\sigma ^{2})t+\sigma \textit{W}_{t}}\;\;\;(2)\\
$$  

<br>
  
## 2. 업비트(Upbit)로부터 이더리움 가격 data 구하기 

Python의 Pyupbit 패키지를 이용하여 업비트에서 이더리움(ETH/USDT) 가격 data를 구해본다.  
<br>

```python
# Module import
import pyupbit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from dateutil.relativedelta import relativedelta
register_matplotlib_converters()
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

eth_price.head()

          date       close
25  2018-01-01  751.911550
26  2018-01-02  857.550000
27  2018-01-03  941.574258
28  2018-01-04  936.729524
29  2018-01-05  969.802500
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

Brownian motion 항($\textit{W}_{t})$은 각 k 시점마다의 충격 $\textit{b}_{k} \sim \textit{N}(0, k)$ 의 합 즉, $\textit{W}_{t} = \sum_{k=1}^{t}\textit{b}_{k}$ 으로 주어지며, 여기에서는 이더리움의 일별 가격자료를 활용하므로 $\textit{k}$ = 1로 둔다.  

실제 계산에서는 (2)식의 $\textit{e}^{(\mu -\frac{1}{2}\sigma ^{2})t+\sigma \textit{W}_{t}}$ 부분을 drift 항($\textit{e}^{(\mu -\frac{1}{2}\sigma ^{2})}t$)과  diffusion 항($\textit{e}^{\sigma \textit{W}_{t}}$)으로 분리하여 $\textit{S}_{0}$에 곱하는 식으로 $\textit{S}_{t}$을 구한다.


$\max\limits_\theta L_{\theta_0}(\theta)$, subject to $D_{KL}^{\rho_{\theta_0}}(\theta_0,\theta)\le\delta$, where $D_{KL}^\rho(\theta_1,\theta_2)=\mathbb{E}\_{s\sim\rho}[D\_{KL}(\pi\_{\theta_1}(\cdot\vert s)\mid\mid\pi\_{\theta_2}(\cdot\vert s))]$







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

위에서 설정한 simulation에 필요한 변수를 GBM 클래스에 대입하여 $\textit{S}_{1}$, $\textit{S}_{2}$ ... $\textit{S}_{T}$을 계산하고, 이를 그래프로 표현한다.
 
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