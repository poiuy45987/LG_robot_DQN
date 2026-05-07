# LG_robot_DQN
1. 수정 사항:
- State 수정:
    - 여러 step 정보를 하나의 step으로 제공
    - 가장자리에 local view 밖의 map 정보를 평균값으로 압축해서 전달
    - Visited layer 제거 -> Trace layer로 대체
- Reward 수정:
    - Revisit penalty: 다음 step에서 다시 cover한 grid 수에 비례하게 증가하도록 설정
    - Turning penalty: Turning한 각도에 비례하여 penalty 증가
2. 사용 모델: Modify_state.pth, Modify_state_max_step_3000.pth