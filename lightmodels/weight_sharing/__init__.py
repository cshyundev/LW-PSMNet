from .basic import PSMNet as basic
from .stackhourglass import PSMNet as weight_sharing

# 기존 hourglass를 별도로 3개를 중첩하는 모델 대신 하나의 hourglass를 3번 통과시켜
# 모델의 파라미터의 개수를 줄인 경량화 모델
