from .basic import PSMNet as basic
from .stackhourglass import PSMNet as channel_compression
# 수행 시간을 줄이기 위해 3D CNN이 연산한 출력의 채널 수를 절반으로 줄인 경량화 실험 모델