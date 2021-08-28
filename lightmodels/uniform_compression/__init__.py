from .basic import PSMNet as basic
from .stackhourglass import PSMNet as uniform_compression
# 매개변수와 메모리 사용량을 줄이기 위해 channel_compression 모델에서
# 2D CNN의 채널 수도 같이 줄인 경량화 실험 모델