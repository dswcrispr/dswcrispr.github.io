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

**아래와 같은 식의 확률변수를 고려하자.** 

$$
logX_{t} = logX_{t-1} + \epsilon_{t}
$$

**여기서 $\epsilon_{t}$는 CLT에 의하여 평균이 0인 정규본포를 따른다고 하자. $X_0$ = c(상수)라고 할때, $X_{t}$의 점근적 분포는 어떻게 되는가?** 


위 문제에 주어진 식은 $X_{t}$에 log가 취해진 것을 제외하면 일반적인 random walk의 표현식인 $X_{t} = X_{t-1} + \epsilon_{t}$과 유사한 형태이다. 

우선 아래와 같이 축차대입법을 통해 $logX_{t}$를 구할 수 있다.

$$
logX_1 - logX_0 = \epsilon_1 \\
+ logX_2 - logX_1 = \epsilon_2 \\
+ logX_3 - logX_2 = \epsilon_3 \\

.....\\
+logX_t - logX_{t-1} = \epsilon_t \\

= logX_t  = logX_{0} + \epsilon_0 + \epsilon_1 + ... \epsilon_t 
$$

그림에서 볼 수 있듯이 $x_{1}$은 함수의 해(그림에서 Root sought)에 조금 더 다가간 것을 확인할 수 있다. 그렇다면 이 과정을 계속해서 반복하면 **반복횟수가 커질수록 $x_{n}$은 solution에 가까워질** 것임을 알 수 있다.

또한 위 수식에서 현재 **$x_{n}$에서의 함수값이 0에서 멀리 떨어져 있을수록 업데이트 되는 값이 커지고, $x_{n}$에서의 기울기 즉, 미분계수가 커질수록 업데이트 되는 값이 작아짐**을 볼 수 있다. 

(3) 위에서 설명한 근사작업을 계속하면 $x_{n+1}$을 아래와 같이 구할 수 있다.

$$
x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})}
$$

이때 반복횟수가 충분히 커서 즉, n이 충분히 커져서 **f($x_{n}$)이 0에 가까울 경우 $x_{n+1}$은 $x_{n}$과 비슷한 수준에서 n이 더욱 늘어도 변하지 않을 것이다. 이때의 $x_{n}$을 f(x)의 solution으로 볼 수 있다.**

컴퓨터의 경우 위의 $x_{n+1}$을 구하는 식에 대해 n을 늘려가면서 $x_{n}$을 업데이트해 나가고, **$x_{n+1}$과 $x_{n}$의 차이가 충분히 작은 어떤 양수 $\epsilon$ 보다 작다**면 업데이트를 멈추는 알고리즘을 통해 함수의 solution을 찾는다. 이와 같은 방법으로 **함수의 solution을 수치적인 방식으로 근사해나가는 방식을 'Newton's method'라 한다.** 

우리가 쓰는 R, python, matlab의 통계패키지나 프로그램들이 함수의 solution을 찾는 방식의 기본은 위와 같은 Newton' method에 바탕을 두고 있다. 
