---
title:  "MakerDao 프로토콜에서의 Collateral Margin 계산 및 시뮬레이션"
excerpt: "The decentralized financial crisis_Imperial college of London 논문 replication 시리즈3"

categories:
  - 프로젝트
tags:
  - defi financial crisis
  - 이더리움
  - simulation
  - collateral margin

use_math: true
comments: true

last_modified_at: 2022-01-28T08:06:00-05:00
---

이번 포스트에서는 지난 포스트의 Correlated Geometric Brownian Motion을 기반으로 MakerDao Defi 프로토콜에서의 Collateral Margin을 계산해보고 이더리움(ETH)과 메이커(MKR)의 가격경로에 따른 Margin 시뮬레이션을 수행해본다.



## 1. Collateral Margin 개요    

MakerDao 프로토콜에서는 이더리움(ETH)을 담보로 DAI를 대출 받을 수 있으며 일반적으로 DAI의 미달러화(USD) 가치의 1.5배의 초과 담보를 요구한다. 또한 MakerDao 프로토콜의 거버넌스 토큰인 메이커(MKR)는 동 프로토콜에서 reserve asset으로서 기능한다. 

MakerDao 프로토콜의 Collateral Margin은 프로토콜 내 담보자산(ETH)의 가치와 거버넌스 토큰(MKR) 가치의 합에서 시스템 부채(발행된 DAI의 가치)를 뺀 것으로 정의할 수 있다. 

이를 수식으로 표현하면 아래와 같다. 

$$
M_{t} = \sum_{k=1}^{K}P_{k, t}Q_{k, t} + P_{r, t}Q_{r, t} - \lambda\sum_{k=1}^{K}d_{k, t}\\
$$

$$
\cdot \ M_{t}: Collateral\; margin\; at\; time\; t\\
\cdot \textit{k}: Each\; agent's\; Etherium\; in \; protocol\\
\cdot \ P_{r,t}: Price\; of\;MKR\;at\;time\;t\\
\cdot \ Q_{r,t}: Quantities\; of\;MKR\;at\;time\;t\\
\cdot \ d_{k,t}: Each\; agent's\;debt(DAI)\;at\;time\;t\\
\cdot \lambda: Overcollateralization\; ratio\\
$$

MakerDao 프로토콜이 안정성을 유지하기 위해서는 $M_{t}$의 값이 0보다 커야하며 이 값이 0보다 작아질 경우 프로토콜에 의한 강제 청산이 이뤄지게 된다. 

<br>
  
## 2. Collateral Margin의 계산과 시뮬레이션

앞서 정의한 Collateral Margin을 실제로 계산해보고 이전 포스트에서 다룬 이더리움과 메이커의 가격경로를 이용해 collateral Margin을 시뮬레이션해본다.    
<br>



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
## 변수 설명 및 설정

# D0: initial debt
# lambda_c: over-collateralization factor
# rho_ill: illiquidity parameter
# L0: initial amount of ETH that can be sold per day
# L: liquidity(ETH)
# Q0: initial amount of ETH escrowed in protocol
# R0: initial amount of reserve asset(MKR)

# 변수 설정(논문에서 제시한 값을 사용)
R0 = 10**6
D0 = 4*(10**8)
lambda_c = 1.5
rho_ill = 0.01
L0 = 30000
```

이제 collateral margin을 계산하는 cal_margin함수를 정의하자.

```python
def cal_margin(prices_eth, prices_mkr, R0, D0, lambda_c, rho_ill, L0):
    Q0 = (D0 * lambda_c) / prices_eth[0]
    Q = Q0
    D = D0
    margin = []
    debt = []

    for i in range(len(prices_eth)):
        eth_value = prices_eth[i] * Q
        mkr_value = prices_mkr[i] * R0
        margin.append(eth_value + mkr_value - (D * lambda_c))
        debt.append(D)
        L = L0 * np.exp(-rho_ill * i)
        D = max(D - (prices_eth[i] * L), 0)
        Q = max(Q - L, 0)

    return margin, debt
```

지난 포스트에서 구한 이더리움과 메이커의 가격경로 시뮬레이션 값과 앞서 설정한 변수를 cal_margin 함수에 대입하여 collateral margin을 시뮬레이션 해본다.  

```python
## 각각의 path에 대해 margin 그래프 그리기
plt.figure(figsize=(20, 10))
for sim in simulations:
    margin = cal_margin(sim.prices_eth, sim.prices_mkr, R0, D0, lambda_c, rho_ill, L0)[0]
    plt.plot(np.arange(0, len(sim.prices_eth)), margin)
    plt.xlabel('Time steps(days)')
    plt.ylabel('USD')
    plt.title('Initial debt: 400,000,000')

## 각각의 path에 대해 remaining debt 그래프 그리기
plt.figure(figsize=(20, 10))
for sim in simulations:
    debt = cal_margin(sim.prices_eth, sim.prices_mkr, R0, D0, lambda_c, rho_ill, L0)[1]
    plt.plot(np.arange(0, len(sim.prices_eth)), debt)
    plt.xlabel('Time steps(days)')
    plt.ylabel('USD')
    plt.title('Initial debt: 400,000,000')
```
margin과 remaining debt에 대한시뮬레이션의 결과는 아래와 같다. 

![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/assets/images/margin_sim.jpg?raw=true)

![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/assets/images/debt_sim.jpg?raw=true)

