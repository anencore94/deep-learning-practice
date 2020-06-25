# 01. Install

## 참고문헌
- [텐서플로우 문서 한글 번역 공식 사이트](https://www.tensorflow.org/install/pip?hl=ko)

## 개요
- 본 문서에서는 Virtualenv 를 사용하여 Ubuntu 18.04 환경에서 python3 프로젝트를 시작하는 방법을 다룹니다.
- virtualenv 를 사용하면 tensorflow 를 각각의 디렉토리 안에 설치하여 호스트 및 다른 프로젝트에 영향을 미치지 않으므로, venv 를 사용합니다. 

## 순서
### 1) pkg 설치
```
$ sudo apt update
$ sudo apt install python3-dev python3-pip
$ sudo pip3 install -U virtualenv  # system-wide install
```

### 2) virtualenv 가상 환경 구축
```
$ cd {$Your-Workspace}
$ mkdir venv
$ virtualenv --system-site-packages -p python3 ./venv
```

### 3) venv 활성화
```
$ source ./venv/bin/activate  # bash 환경일 경우

(venv) $ # 이제 쉘 프롬프트가 (venv) 로 시작하도록 바뀝니다.

# 나중에 venv 를 종료하려면 다음을 실행합니다.
(venv) $ deactivate # tensorflow 를 사용 중일 때는 비활성화하지 마세요.
```

### 4) 가상환경 내의 pip pkg 확인
```
(venv) $ pip install --upgrade pip
(venv) $ pip list
```

### 5) Tensorflow pip pkg 설치
```
# python version 이 3.6 인지 확인합니다.
(venv) $ python --version

# tensorflow pkg 를 설치합니다.
# CPU 전용, Python 3.6 전용 의 tensorflow 2.1.0 을 설치합니다.
(venv) $ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.1.0-cp36-cp36m-manylinux2010_x86_64.whl

# 설치를 확인합니다. System message 를 제외하고 error message 없이 tensorflow version 2.1.0 과 hello message 가 출력된다면 정상 설치된 것입니다.
(venv) $ python -c "import tensorflow as tf; x = [[2.]]; print('tensorflow version', tf.__version__); print('hello, {}'.format(tf.matmul(x, x)))"
```

### 6) IDE 에서 해당 venv 의 python 바라보도록 설정

- [Pycharm 설정](https://psychoria.tistory.com/447)
