from .basic import PSMNet as basic
from .stackhourglass import PSMNet as diparity_expansion_v2

# 3D CNN의 채널수는 절반으로 압축하고, cost volume의 disparity 차원수를 1/2로 확장시킨 모델
