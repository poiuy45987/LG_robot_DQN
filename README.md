# LG_robot_DQN
1. 수정 사항:
- 초기 warmup episode: Episode 수가 적으면 적당히 heuristic으로 움직이게 함. -> 학습의 방향성 설정
- Local view를 현재 로봇이 바라보는 방향에 맞게 회전시킴
- Local view를 polar coordinate으로 바꿔서 CNN이 로봇이 필요한 feature를 더 쉽게 추출할 수 있도록 함.

2. 사용 model: 2605_Polar_local_view_1.pth