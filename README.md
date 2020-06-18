# deep-learning-practice
Tensorflow 2.0 프레임워크 사용한 딥러닝 모델 구현 연습

## 개발 컨벤션 체크
- [tensorflow code style 참조](https://www.tensorflow.org/community/contribute/code_style)
- [Google python style 참조](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
- 모든 .py 코드에 대해 다음 실행
    - 기본 pylintrc wget 한 파일에서 `indent-after-paren=2` 만 4 -> 2 로 변경 후 사용(indent 4 대신 2 space 가 tensorflow convention)
    ```commandline
    pylint --rcfile=/tmp/pylintrc /home/kjy/git/deep-learning-practice/*/*.py
    ```
