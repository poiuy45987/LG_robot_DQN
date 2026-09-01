# Legacy DQN implementation

이 디렉터리는 이전 DQN 기반 경로 계획 구현을 보관합니다.

## 포함 파일

- `agent.py`: 이전 DQN 학습·평가 구현
- `redqn_network.py`: 이전 ReDQN 네트워크 구현

## 상태

현재 기본 학습 및 평가 경로는 LSTM 구현을 사용합니다.

이 디렉터리의 코드는 `main.py`에서 호출하지 않으며,
현재 환경 및 의존성과의 호환성은 보장하지 않습니다.