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

이에 대해 아래와 같은 과정을 생각해 볼 수 있다.(아래에서 소개하는 Newton's method는 미분가능한 함수에 대해 적용가능하다.)


(1) 초기값 $x_{0}$에 대해 f($x_{0}$)을 계산해본다. 당연히 임의의 $x_{0}$에 대해 f($x_{0}$)값은 높은 확률로 0과 다를 것이다.

(2) 위의 그림에서처럼 ($x_{0}$, f($x_{0}$))을 지나면서 f(x)에 접하는 접선을 구하고 이 접선이 x축과 만나는 절편을 $x_{1}$로 두자.

위의 접선은 y = f($x_{0}$) + f'(x)(x - $x_{0}$) 로 표현할 수 있고 이 접선의 해, 즉 x절편은 $x_{1}$은 아래와 같다. 

$$
x_{1} = x_{0} - \frac{f(x_{0})}{f'(x_{0})}
$$

그림에서 볼 수 있듯이 $x_{1}$은 함수의 해(그림에서 Root sought)에 조금 더 다가간 것을 확인할 수 있다. 그렇다면 이 과정을 계속해서 반복하면 반복횟수가 커질수록 $x_{n}$은 solution에 가까워질 것임을 알 수 있다.

또한 위 수식에서 현재 $x_{n}$에서의 함수값이 0에서 멀리 떨어져 있을수록 업데이트 되는 값이 커지고, $x_{n}$에서의 기울기 즉, 미분계수가 커질수록 업데이트 되는 값이 작아짐을 볼 수 있다. 

(3) 위에서 설명한 근사작업을 계속하면 $x_{n+1}$을 아래와 같이 구할 수 있다.

$$
x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})}
$$

이때 반복횟수가 충분히 커서 즉, n이 충분히 커져서 **f($x_{n}$)이 0에 가까울 경우 $x_{n+1}$은 $x_{n}$과 비슷한 수준에서 n이 더욱 늘어도 변하지 않을 것이다. 이때의 $x_{n}$을 f(x)의 solution으로 볼 수 있다.**

컴퓨터의 경우 위의 $x_{n+1}$을 구하는 식에 대해 n을 늘려가면서 $x_{n}$을 업데이트해 나가고, **$x_{n+1}$과 $x_{n}$의 차이가 충분히 작은 어떤 양수 $\epsilon$ 보다 작다**면 업데이트를 멈추는 알고리즘을 통해 함수의 solution을 찾는다. 이와 같은 방법으로 **함수의 solution을 수치적인 방식으로 근사해나가는 방식을 'Newton's method'라 한다.** 

우리가 쓰는 R, python, matlab의 통계패키지나 프로그램들이 함수의 solution을 찾는 방식의 기본은 위와 같은 Newton' method에 바탕을 두고 있다. 
