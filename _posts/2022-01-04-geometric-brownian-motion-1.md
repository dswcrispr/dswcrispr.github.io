---
title:  "Geometric Brownian Motion 시뮬레이션(이더리움(ETH) 가격에 대해)"
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

이번 포스트에서는 Geometric Brownian Motion을 이용하여 이더리움의 미래 가격에 대한 시뮬레이션을 수행해본다.  


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
\cdot \sigma: Volatility of asset price\\
\cdot \textit{dW}: Change\;in\;Brownian motion\;term\\
$$

(1) 수식은 짧은 시간 간격 동안의 자산가격 변화가 짧은 시간 간격(dt) 동안의 평균적인 수익률(drift)과 랜덤한 충격에 의한 변동분의 합으로 나타남을 의미한다.

(1) 수식에 Ito formula를 적용하면 아래와 같은 Geometric Brownian motion 수식을 유도할 수 있다.

$$
S_{t} = S_{0}\textit{e}^{(\mu -\frac{1}{2}\sigma ^{2})t+\sigma \textit{W}_{t}}\;\;\;(2)\\
$$  

<br>
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

여기에서는 The Decentralized Financial Crisis: Attacking DeFi 논문의 replication을 위해 2018.1.1 ~ 2020.2.7일의 ETH-USDT 종가 data를 다운받는다. 

```python
# 함수에 넣을 argumets 세팅
ticker = 'USDT-ETH'
interval = 'day1'
to = '2020-02-07'
count = 752

## eth_data에 해당기간 자료를 저장
# get_ohlcv를 통해 일별 가격정보(open, high, low, close), 거래량(volume) 등의 정보를 저장
eth_data = pyupbit.get_ohlcv(ticker = ticker, interval = interval, to = to, count = count)
```

분석을 위해 eth_data dataframe중 종가(close) 정보만 남기고 날짜를 'yyyy-mm-dd' 형태로 정리한다.

```python
# eth_data의 index(날짜)와 종가를 열로 하는 dataframe 생성
eth_price = eth_data.reset_index(drop = False)[['index', 'close']]

# column 이름 변경
eth_price.rename(columns = {'index':'date', 'close':'close'}, inplace = True)

# 시-분-초까지 표현되어 있는 date를 'yyyy-mm-dd' 형식으로 변경
eth_price['date'] = pd.to_datetime(eth_price['date']).dt.date
```

해당 기간(2018.1.1 ~ 2020.2.7)의 이더리움 가격을 그래프로 표현해보자.
```python
# 그래프 그리기
plt.figure(figsize= (15, 10))
plt.plot(eth_price['date'], eth_price['close'])
plt.xlabel('Date')
plt.ylabel('ETH/USD')
plt.show()
```

![](https://devinlife.com/assets/images/bio-photo-keyboard-small.jpg)

![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/images/eth_price.jpg)




이 글의 제목은 {{ page.title }}이고
마지막으로 수정된 시간은 {{ page.last_modified_at }}이다..
$x+y = 1$

$$
  A_{m,n} = \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{m,1} & a_{m,2} & \cdots & a_{m,n}
  \end{pmatrix}
 $$

