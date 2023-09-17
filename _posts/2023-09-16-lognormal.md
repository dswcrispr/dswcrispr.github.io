---
title:  "log-normal random walk"
excerpt: "log-normal random walk 예시, 시뮬레이션"

categories:
  - Math/Statistics
tags:
  - log-normal distribution
  - python
  - random walk
  - simulation

use_math: true
comments: true

last_modified_at: 2023-09-16T08:06:00-05:00
---

대학원 입시를 준비하는 친구가 입학고사의 기출 문제를 보여줬는데, 풀이와 함께 개념 등을 정리해보는 것이 의미가 있을 것이라 생각되어 포스팅으로 남겨본다.

우선 해당 문제는 아래와 같다. 

<u>**아래와 같은 식의 확률변수를 고려하자.**</u> 

$$
logX_{t} = logX_{t-1} + \epsilon_{t}
$$

<u>**여기서 $\epsilon_{t}$는 CLT(central limit theroem)에 의하여 평균이 0인 정규본포를 따른다고 하자. $X_0$ = c(상수)라고 할때, $X_{t}$의 점근적 분포는 어떻게 되는가?**</u> 


위 문제에 주어진 식은 $X_{t}$에 log가 취해진 것을 제외하면 일반적인 random walk의 표현식인 $X_{t} = X_{t-1} + \epsilon_{t}$과 유사한 형태이다. 

우선 아래와 같이 축차대입법을 통해 $logX_{t}$를 구할 수 있고, $logX_t$는 초기값 $logX_{0}$과 $\epsilon_i$값들의 합으로 표현가능함을 알 수 있다.

$$
logX_1 - logX_0 = \epsilon_1 \\
+ logX_2 - logX_1 = \epsilon_2 \\
+ logX_3 - logX_2 = \epsilon_3 \\

.....\\
+logX_t - logX_{t-1} = \epsilon_t \\

logX_t = logX_{0} + \epsilon_0 + \epsilon_1 + ... \epsilon_t \\
logX_t = logX_{0} + \sum_{i=1}^t\epsilon_i
$$

문제에서 주어진 $X_{0} = c$, $\epsilon_t ~ N(0, \sigma^2)$ 등의 조건을 활용하면 $logX_t$는 평균이 $logc$,분산이 $t\sigma^2$인 정규분포를 따름을 알 수 있다.(정규분포를 따르는 확률변수의 합은 정규분포를 따르므로)
즉, 아래와 같이 표현 가능하다.

$$
logX_t \sim N(logc, t\sigma^2)
$$

위에서 볼 수 있듯 $X_t$는 log를 취한 값이 정규분포를 따르는 log-normal distribution을 따르는 확률변수이다. 

아래에서는 $X_0$의 초기값 c와 $\epsilon$의 표준편차인 $\sigma$에 특정한 값을 주고 $logX_t$가 실제로 정규분포를 따르는지 python을 통한 simulation으로 확인해보고자 한다. 