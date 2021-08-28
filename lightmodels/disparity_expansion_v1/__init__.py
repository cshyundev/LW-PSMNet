from .basic import PSMNet as basic
from .stackhourglass import PSMNet as disparity_expansion_v1

# 3D CNN의 채널수는 절반으로 압축하고, cost volume의 disparity 차원수를 1/3로 확장시킨 모델
