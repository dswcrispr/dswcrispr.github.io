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

```python
s = 'python syntax highlighting'
print s
```
