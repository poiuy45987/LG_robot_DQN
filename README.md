# LG_robot_DQN

## 1. 코드 목적 및 설명
* 장애물이 분포된 2D grid map 환경에서 로봇 청소기의 청소 경로를 생성하는 모델
* 로봇 청소기의 청소 시작점 및 방향 설정: 
    * 좌상단를 초기 위치로 설정. 장애물 때문에 배치 실패 시 우상단 -> 우하단 -> 좌하단 순서로 배치 시도.
    * 초기 위치가 좌상단 또는 우상단일 때: 하단 방향을 바라봄.
    * 초기 위치가 좌하단 또는 우하단일 때: 상단 방향을 바라봄.
* LSTM 구조 및 vanilla policy gradient 방법을 활용한 model

## 2. src 파일 구조 및 설명

```text
src/path_planner/
├── path_planner/                   # 패키지 내부 파일
│   ├── utils/                      # 유틸리티 모듈 모음
│   │   ├── map_utils.py            # 맵 연산(coverable grid 계산, 직사각형 또는 원 그리기 등) 관련 utils
│   │   ├── trajectory_metrics.py   # 경로 성능 지표(Overlap rate, cleaning time) 계산
│   │   ├── visualizer.py           # 시각화 method 모듈
│   │   └── utils.py                # 기타 모듈 모음 (Device 정보 출력, torch seed 설정 등)
│   ├── LSTM_agent.py               # train 및 test를 담당하는 파일
│   ├── LSTM_network.py             # LSTM 구조를 활용한 신경망 구조가 정의된 파일
│   ├── environment.py              # 강화 학습용 환경(CoverageEnv)이 정의된 파일
│   ├── map_layer.py                # 맵 레이어 데이터 구조가 정의된 파일. 매 스텝 맵 업데이트 방식, 시작점 설정 규칙 등도 정의
│   ├── map_generator.py            # 임의의 맵 생성 모듈
│   ├── config.py                   # Hyperparameter, 맵, 환경 설정값 모음
│   ├── main.py                     # 커맨드라인 인자 정의 및 메인 실행 스크립트 파일
│   └── train_and_test.ipynb        # Notebook 파일에서 훈련 및 테스트 수행
└── setup.py                        # 명령어 "pip install -e ."로 패키지를 설치하기 위한 파일
```

## 3. 환경 설치

* 권장 Python 버전: 3.12

* Working directory 설정: LG_robot_DQN/

* 가상환경 생성 및 활성화:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

* pip 업그레이드 및 라이브러리/패키지 설치:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd src/path_planner
python -m pip install -e .
cd ../..
```

## 4. 훈련 실행
```bash
python -m path_planner.main \
    --mode train \
    --valid_freq 1 \
    --ckp_freq 1 \
    --use_train_maps \
    --use_wandb
```
* --valid_freq: Validation을 실행하는 빈도 수를 episode 단위로 설정
* --ckp_freq: Checkpoint를 저장하는 빈도 수를 episode 단위로 설정
* --use_train_maps: 미리 생성한 map으로 훈련 진행
* --use_wandb: 훈련 진행 상황을 판단할 수 있는 metric을 wandb로 저장--model_dir: Model 파일이 저장되는 폴더 경로 설정.
* --map_save_dir: Map 파일이 저장되어 있는 폴더 경로 설정. (Training 시에는 특별히 설정할 필요 없음)
* --pre_model_name: Pre-trained model 파일 이름. Parameter만 불러오고 step 수, optimizer 상태 등은 초기화됨.
* --model_name: 저장하거나 불러올 model 이름 설정. Train mode에서 이 model의 checkpoint가 있을 경우, 최신 checkpoint부터 학습을 재개.

## 5. 테스트 실행
```bash
python -m path_planner.main \
    --mode test \
    --test_map_num_per_level 250 \
    --vis_test_map_num 3
```
* --map_save_dir: Map 파일이 저장되어 있는 폴더 경로 설정. 내부에 'test' 폴더를 만들고 그 안에 test할 map들을 옮기면 해당 map으로 test가 진행됨.
    * 기본 설정 경로: src/path_planner/path_planner/maps
    * 폴더 구조:
```text
(--map_save_dir 폴더 경로)/
└── test/
    └── test_map_L1_0001.npy
    └── test_map_L1_0002.npy
    ...
```

* --test_map_num_per_level: 각 level별 test하는 map 개수
* --vis_test_map_num: 각 레벨별로 Best, Median, Worst 경로를 추출할 개수 (기본값: `3`)
* --model_name: 불러올 model 이름 설정.

* test 결과 저장 위치: src/path_planner/path_planner/result/(모델 이름): 각 map에서 생성된 경로 png 파일과 test 결과가 정리된 excel 파일이 같이 있습니다.

### 시각화 결과를 보고 싶은 경우, train_and_test.ipynb 파일의 cell을 실행하면 됩니다.
