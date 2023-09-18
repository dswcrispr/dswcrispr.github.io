---
title:  "log-normal 분포의 성질과 시뮬레이션"
excerpt: "log-normal 분포 예시, 시뮬레이션"

categories:
  - Math/Statistics
tags:
  - log-normal distribution
  - python
  - simulation

use_math: true
comments: true

last_modified_at: 2023-09-16T08:06:00-05:00
---

대학원 입시를 준비하는 친구가 입학고사의 기출 문제를 보여줬는데, 풀이와 함께 **log-normal 분포의 성질** 등을 정리해보는 것이 의미가 있을 것이라 생각되어 포스팅으로 남겨본다.

우선 해당 문제는 아래와 같다. 

_아래와 같은 식의 확률변수를 고려하자._

$$
logX_{t} = logX_{t-1} + \epsilon_{t}
$$

여기서 $\epsilon_{t}$는 CLT(central limit theroem)에 의하여 평균이 0인 **정규분포**를 따른다고 하자. $X_0 = C$(상수)라고 할때, $X_{t}$의 **점근적 분포**는 어떻게 되는가?

### (1) 문제풀이

위 문제에 주어진 식은 $X_{t}$에 log가 취해진 것을 제외하면 일반적인 random walk의 표현식인 $X_{t} = X_{t-1} + \epsilon_{t}$과 유사한 형태이다. 

우선 아래와 같이 축차대입법을 통해 $logX_{t}$를 구할 수 있고, **$logX_t$는 초기값 $logX_{0}$과 $\epsilon_i$값들의 합으로 표현가능**함을 알 수 있다.

$$
\begin{align*} logX_1 - logX_0 = \epsilon_1 \\
+ logX_2 - logX_1 = \epsilon_2 \\
+ logX_3 - logX_2 = \epsilon_3 \\
.....\\ 
+logX_t - logX_{t-1} = \epsilon_t \end{align*}\\
$$

$$
\begin{align*} & logX_t = logX_{0} + \epsilon_0 + \epsilon_1 + ... \epsilon_t  \\ &
logX_t = logX_{0} + \sum_{i=1}^t\epsilon_i \end{align*}
$$

문제에서 주어진 $X_{0} = C$, $\epsilon_t ~ N(0, \sigma^2)$ 등의 조건을 활용하면 $logX_t$는 평균이 $logc$,분산이 $t\sigma^2$인 정규분포를 따름을 알 수 있다.(정규분포를 따르는 확률변수의 합은 정규분포를 따르므로)
즉, 아래와 같이 표현 가능하다.

$$
logX_t \sim N(logc, t\sigma^2)
$$

위에서 볼 수 있듯 $X_t$는 log를 취한 값이 정규분포를 따르는 log-normal distribution을 따르는 확률변수이다. 

### (2) log-normal 분포의 성질(pdf, 기대값, 분산)

#### log-normal 분포의 pdf

log-normal 분포를 따르는 확률변수 $X$의 pdf(probability density function, 확률밀도함수)는 다음과 같이 유도할 수 있다. 

$$
\begin{align*} Y & = logX \sim N(\mu, \sigma^2)\\ 

F_X(x) & = P(X \leq x) = P(e^y \leq x) = P(Y \leq logx) = F_Y(logx) = F_Y(y) \end{align*}
$$

위로부터 $f_X(x) = f_Y(y)\frac{\partial{y}}{\partial{x}}$ 를 통해 X의 pdf를 아래처럼 구할 수 있다.

$$
f_X(x) = \frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(logx-\mu)^2}{2\sigma^2})\frac{1}{x} \\

(\because \frac{\partial{y}}{\partial{x}} = \frac{1}{x})
$$

#### log-normal 분포의 기대값

log-normal 분포의 기대값은 $E(X) = \int^\infty_{-\infty}{x f(x)dx}$을 통해 계산가능하다. 다만 약간의 기술적인 수식변형이 들어가니 이에 유의하여 계산하면 아래와 같다.

$$
\begin{align*} E(X) & = \int^\infty_{0}{\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(logx-\mu)^2}{2\sigma^2})\frac{1}{x}\cdot xdx} \\
& = \int^\infty_{-\infty}{\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(t-\mu)^2}{2\sigma^2})\cdot exp(t)dt},\; (\because logx = t, dx = exp(t)dt) \\
& = \int^\infty_{-\infty}{\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(t-\mu)^2 + 2\sigma^2t}{2\sigma^2})dt} \\
& = \int^\infty_{-\infty}{\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(t-(\mu+\sigma^2))^2}{2\sigma^2} +\mu+ \frac{1}{2}\sigma^2)dt} \\
& = exp(\mu+\frac{1}{2}\sigma^2)\; (\because \int^\infty_{-\infty}{\frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(t-(\mu+\sigma^2))^2}{2\sigma^2})dt} = 1)\\

\end{align*}
$$

#### log-normal 분포의 분산

log-normal 분포의 분산 또한 기대값과 유사하게 구할 수 있는데, 계산과정은 생략하고 여기에서는 결과만 남겨둔다.<br>
$Var(X) = exp(2\mu+\sigma^2)\cdot (exp(\sigma^2) - 1)$


### (3) log-normal 분포의 활용

위에서 다룬 log-normal 분포를 어디에 활용할 수 있을까? 우선적으로 떠올려볼 수 있는 것은 주식가격과 수익률의 분포에 적용해보는 것이다. 

예를 들어 t-1기의 주식가격을 $P_{t-1}$, t기의 주식가격을 $P_t$로 두고, 해당 기간동안의 수익률이 $r_t$이라면 다음식이 성립한다. $\frac{P_t}{P_{t-1}} = 1+r_t$

위 식에 log를 취하고 크지 않은 r에 대해 $log(1+r) \approx r$임을 이용하면 아래의 식을 얻는다

$$
\begin{align*} log\frac{P_t}{P_{t-1}} & = log(1+r_t) \approx r_t \\
\\
logP_t - logP_{t-1} & \approx r_t
\end{align*}
$$

만약 주식 수익률 $r_t$가 **평균이 0인 정규분포를 따른다면 위 식은 포스팅 서두에 등장한 모 대학원 기출문제와 동일한 구조**를 가짐을 알 수 있다. 이를 통해 **T시점의 주식가격의 로그값은 아래와 같이 최초 주식가격의 로그값에 해당 시점까지의 수익률을 누적해서 더한 값으로 근사**해서 표현할 수 있다.  

$$
logP_T = logP_0 +  \sum_{i=1}^Tr_i
$$

그리고 수익률의 분포가 정규분포를 따른다면 주식가격의 로그값은 정규분포를 따르고, 주식가격 자체는 log-normal 분포를 따를 것이다. 

이처럼 **log-normal 분포는 주식가격처럼 항상 양수를 갖는 확률변수를 모델링**할 때 활용할 수 있다.

### (4) python을 통한 simulation

이제 아래에서는 python을 이용해서 위 문제의 수식으로부터 생성되는 $logX_T$, $X_T$가 각각 normal, log-normal 분포를 따르는지 simulation으로 확인해보고자 한다.

이를 위해 다음과 같이 $X_0$, $\sigma$, T, N(표본수) 등이 주어졌을 때 $logX_T$ 및 $X_T$를 계산하는 'log-normal'이라는 class를 정의한다.

```python
# Module import
import numpy as np
import matplotlib.pyplot as plt
import sseaborn as sns
plt.rcParams['figure.figsize'] = (10, 6)

class log_normal:
  
      def __init__(self, x0, sigma, T, N):
        self.logx0 = np.log(x0) # 초기값 x0
        self.sigma = sigma # sigma 초기값
        self.T = T 
        self.N = N # 표본수
        self.epsilon = None 
        self.logX_t = None
        self.logX = None
        self.logX_Ts = [] 
        self.X_Ts = []

    # X_T, logX_T를 저장할 list 초기화 method    
    def _init_X0(self):
        self.logX_t = self.logx0
        self.logX = [self.logx0]

    # epsilon 초기화    
    def _init_epsilon(self):
        self.epsilon = np.random.normal(0, self.sigma, self.T - 1)

    # logXt 계산    
    def _logXt_calculator(self):
        self._init_X0()
        self._init_epsilon()
        for i in range(self.T - 1):
            self.logX_t += self.epsilon[i]
            self.logX.append(self.logX_t)

    # 계산된 X_T, logX_T를 저장하는 list         
    def logX_T_collector(self):
        for i in range(self.N):
            self._logXt_calculator()
            self.logX_Ts.append(self.logX_t)
            self.X_Ts.append(np.exp(self.logX_t))

    # histogram plotting        
    def plotting(self):
        fig, ax = plt.subplots(ncols = 2)
        plt.ylim(0, 0.03)
        sns.histplot(self.logX_Ts, kde = True, stat = 'density', ax = ax[0])
        sns.histplot(self.X_Ts, kde = True, stat = 'density', ax = ax[1])
        plt.show
```

위에서 정의한 class를 이용해서 문제에서의 수식으로부터 계산한 $X_T$의 표본수가 충분히 클 때 $logX_T$와 $X_T$ 값들의 히스토그램을 그려본다.   

X의 초기값으로는 40, $epsilon$의 표준편차는 0.02, T=500 으로 설정하고 표본수를 각각 500, 1000, 3000으로 늘려본다.

```python
# 인스턴스 생성 및 plotting
sample1 = log_normal(40, 0.02, 500, 500)
sample1.lotX_T_collector()
sample1.plotting()

sample2 = log_normal(40, 0.02, 500, 1000)
sample2.lotX_T_collector()
sample2.plotting()

sample3 = log_normal(40, 0.02, 500, 3000)
sample3.lotX_T_collector()
sample3.plotting()
```

각각의 simulation에서 구한 histogram은 아래와 같다.

![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/assets/images/lognormal/lognormal_500.png?raw=true)<br>
(그림1. N = 500, 각각 왼쪽이 $logX_T$, 오른쪽이 $X_T$)

![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/assets/images/lognormal/lognormal_1000.png?raw=true)<br>
(그림2. N = 1,000)

![](https://github.com/dswcrispr/dswcrispr.github.io/blob/master/assets/images/lognormal/lognormal_3000.png?raw=true)<br>
(그림3. N = 3,000)

위 히스토그램들에서 볼 수 있듯이 표본수인 N이 커질 수록 $logX_T$와 $X_T$의 히스토램이 각각 normal, log-normal 분포의 pdf와 닮아가는 것을 볼 수 있다. 또한 추출한 $logX_T$와 $X_T$의 표본으로부터 얻은 평균과 표준편차가 각각 위에서 정리한 분포의 평균, 표준편차에 근접한 값이 나옴을 아래와 같이 확인가능하다. 

```python
# N=3000일때 logX_T, X_T의 평균 및 분산
>>> sample3 = log_normal(40, 0.02, 500, 3000)
>>> sample3.logX_T_collector()
>>> np.mean(sample3.logX_Ts)
3.685818842465808  # log(40)인 3.6887과 유사

>>> np.std(sample3.logX_Ts)
0.4572085112864432  # sqrt(500)*sigma(0.02)인 0.44721과 유사  
```