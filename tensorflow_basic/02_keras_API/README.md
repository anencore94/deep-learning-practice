# 목차
## 파트 1 : Tensorflow 의 기본 API
- Tensor
- 변수
- Tf 에서 수학을 하는 것
- GradientTape 를 사용해 미분값을 계산하는 것
    - GradientTape 의 api 사용법
    - gradient 계산을 위해서 automatic differentiation 을 사용 ([위키 링크](https://en.wikipedia.org/wiki/Automatic_differentiation))
        - forward 방향의 연산을 전부 기록해놨다가 backward 방향으로 연산할 때 사용하기 때문에 기록해야 함
- tf.function 데코레이터
    - tf 2.x 는 default 로 eager 실행모드로 수행하여 디버깅과 코드 라인별 결과 출력에 용이
    - 하지만 시간복잡도가 큰 연산을 실행하는 함수는 정적 그래프를 활용해 연산을 수행하는 것이 속도 개선에 좋으므로, 그런 함수는 정적 그래프를 활용하도록 데코레이터를 붙여주는 것이 좋음

## 파트 2 : Keras 의 기본 API
- Layer 클래스
- 학습 가능, 학습 불가능한 가중치
- 재귀적으로 Layer 조합하는 법
- call 메소드의 training 인자
    - train, test 에서 내부 로직이 다른 Layer 의 경우 call 메소드로 training 인자 노출
    - BatchNormalization, Dropout 등
- Loss 클래스
    - CrossEntropy, KLD 등 미리 정의된 손실함수 제공
- Metric 클래스
    - Loss 는 상태를 가지지 않지만, Metric 은 상태를 가짐
    - `update_state` 로 상태 갱신, `result` 로 스칼라 형태의 value 요청
    - AUC, FalsePositives 등
    - Metric 새로 만들고 싶을 경우 설정해주어야 하는 것
        - `__init__` 내에 상태 변수 추가
        - `update_state`, `result`, `reset_states` 함수 오버라이딩

## 파트 3: TODO
-  Optimizer 클래스 & 빠른 end-to-end 학습 반복문 부터