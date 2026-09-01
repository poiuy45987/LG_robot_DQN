# Robot Vacuum Path Planner

## 1. 코드 목적 및 설명
* 장애물이 분포된 2D grid map 환경에서 로봇 청소기의 청소 경로를 생성하는 모델
* 로봇 청소기의 청소 시작점 및 방향 설정: 
    * 좌상단를 초기 위치로 설정. 장애물 때문에 배치 실패 시 우상단 -> 우하단 -> 좌하단 순서로 배치 시도.
    * 초기 위치가 좌상단 또는 우상단일 때: 하단 방향을 바라봄.
    * 초기 위치가 좌하단 또는 우하단일 때: 상단 방향을 바라봄.
* LSTM 구조 및 vanilla policy gradient 방법을 활용한 model

## 2. src 파일 구조 및 설명

```text
LG_robot_LSTM_VPG/
└─src/path_planner/                         
       ├─ path_planner/                     # 패키지 내부 파일.
       │   ├── main.py                      # 메인 실행 스크립트 파일 및 커맨드라인 인자 정의.
       │   ├── LSTM_agent.py                # train 및 test 방식, action 선택 방식, path reward 함수 정의.
       │   ├── LSTM_network.py              # LSTM 구조를 활용한 신경망 구조 정의.
       │   ├── environment.py               # 강화 학습용 환경(CoverageEnv) 정의.
       │   ├── map_layer.py                 # Map layer 구조 정의. 시작점 설정 규칙, 매 step의 map update 방식 정의.
       │   ├── map_generator.py             # 임의 map 생성 모듈.
       │   ├── gif_generator.py             # Test map에 대해 로봇 청소기가 청소하는 gif 파일 생성.
       │   ├── config.py                    # Hyperparameter 및 map과 environment의 설정값 모음.
       │   ├── utils/                       # 유틸리티 모듈 모음
       │   │   ├── map_utils.py             # Map 관련 연산 (Coverable grid 계산 등) method 모음.
       │   │   ├── trajectory_metrics.py    # 경로 길이, 청소 시간, overlap 지표 계산.
       │   │   ├── visualizer.py            # 시각화 관련 모듈.
       │   │   └── utils.py                 # 기타 유틸리티 모듈 모음. (예: Device 정보 출력, torch seed 설정 등)
       │   └── train_and_test.ipynb         # Notebook 파일에서 train 및 test 수행. 시각화 결과를 더 편하게 볼 수 있음.
       └── setup.py                         # 명령어 “pip install -e src/path_planner”로 패키지를 설치하기 위한 파일.
```

## 3. 환경 설치

* 권장 Python 버전: 3.12

* 프로젝트 루트 디렉터리에서 아래 명령을 실행합니다.

* 가상환경 생성 및 활성화:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

* pip 업그레이드 및 라이브러리/패키지 설치:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e src/path_planner
```

## 4. 훈련 실행
```bash
python -m path_planner.main train \
    --valid-freq 1 \
    --ckp-freq 1 \
    --use-train-maps \
    --use-wandb
```
* `--valid-freq`: Validation을 실행하는 빈도 수를 episode 단위로 설정
* `--ckp-freq`: Checkpoint를 저장하는 빈도 수를 episode 단위로 설정
* `--model-dir`: Model 파일이 저장되는 폴더 경로 설정
* `--map-save-dir`: Map 파일이 저장되어 있는 폴더 경로 설정
* `--pre-model-name`: Pre-trained model 파일 이름. Parameter만 불러오고 step 수, optimizer 상태 등은 초기화됨.
* `--model-name`: 저장하거나 불러올 model 이름 설정. Train mode에서 이 model의 checkpoint가 있을 경우, 최신 checkpoint부터 학습을 재개.
* `--use-train-maps`: 미리 생성한 map으로 훈련 진행
* `--use-wandb`: 훈련 진행 상황을 판단할 수 있는 metric을 wandb로 저장

## 5. 테스트 실행
```bash
python -m path_planner.main test \
    --test-map-num-per-level 250 \
    --vis-test-map-num 3
```
* `--map-save-dir`: Map 파일이 저장되어 있는 폴더 경로 설정. 내부에 `test` 폴더를 만들고 그 안에 test할 map들을 옮기면 해당 map으로 test가 진행됨.
    * 기본 설정 경로: `src/path_planner/path_planner/maps`
    * 폴더 구조:
```text
(--map-save-dir 폴더 경로)/
└── test/
    └── test_map_L1_0001.npy
    └── test_map_L1_0002.npy
    ...
```

* `--test-map-folder-name`: Map 파일이 저장되어 있는 폴더에서 test할 map이 저장된 폴더 이름. (map-save-dir의 폴더 경로)/(test-map-folder-name)에 있는 map들로 test가 진행됨.
* `--test-map-num-per-level`: 각 level별 test하는 map 개수
* `--vis-test-map-num`: 각 레벨별로 Best, Median, Worst 경로를 추출할 개수 (기본값: `3`)
* `--min-coverable-area-rate`: Map에서 청소할 수 있는 영역의 비율의 최솟값. 시작 지점을 결정할 때, 이 비율을 만족할 수 있는 시작점으로 결정하고, 이 비율을 만족할 수 있는 시작점이 없는 경우 청소를 진행하지 않음. (기본값: `0.1`)
* `--model-name`: 불러올 model 이름 설정.

* test 결과 저장 위치: src/path_planner/path_planner/result/(모델 이름): 각 map에서 생성된 경로 png 파일과 test 결과가 정리된 excel 파일이 같이 있습니다.

## 6. GIF 생성

```bash
python -m path_planner.main gif \
    --model-name model.pth \
    --map-rel-path test/test_map_L1_0000.npy
```

결과는 `result/(모델 이름)/gif_result/(맵 이름).gif`에 저장됩니다.

### 시각화 결과를 좀 더 편하게 보고 싶은 경우, train_and_test.ipynb 파일의 cell을 실행하면 됩니다.

## 7. 기타 Mode

### 7.1. Inference Mode
특정 맵에서 로봇의 청소 성능을 평가하고 결과를 출력합니다. Coverage, Overlap rate, Cleaning time, Inference time 등의 지표가 자동으로 계산되어 출력되며, 최종 청소 경로를 시각화합니다.

```bash
python -m path_planner.main inference \
    --model-name model.pth \
    --map-rel-path test/test_map_L1_0000.npy
```

#### 필수 인자
* `--model-name`: 불러올 model 이름 설정 (기본값: `model.pth`)
* `--map-rel-path`: 추론할 맵 경로 (기본값: `test/test_map_L1_0000.npy`)

#### 추가 인자
* `--start-mode`: 로봇의 시작 위치 모드. `corner` (모서리)로 설정하면 로봇이 맵의 코너에서 시작하고, `edge` (모서리 전체)로 설정하면 맵의 어느 쪽 모서리든 시작 가능 (기본값: `corner`)
* `--min-coverable-area-rate`: 청소 가능한 최소 면적 비율. 이 값 이하의 청소 가능 영역을 가진 맵은 스킵됨 (기본값: `0.1`)
* `--debug`: 대화형 단계별 디버깅 모드 활성화. 활성화 시 각 스텝마다 로봇의 상태, 각 action의 선택 확률 등을 확인하고 진행 여부를 선택 가능.

#### 출력 예시
```
============================================================
Inference Results for: test/test_map_L1_0000.npy
============================================================
Coverage:                   95.2%
Overlap Rate:               40.5%
Cleaning Time:              1.6 min
    Target Coverage 85.0%: Reached | Time: 0.9 min | Overlap: 30.3%
    Target Coverage 90.0%: Reached | Time: 1.2 min | Overlap: 34.6%
    Target Coverage 95.0%: Reached | Time: 1.6 min | Overlap: 40.5%
Total Inference Time:       5.44 s
Inference Time per Step:    6.34 ms
============================================================
```

### 7.2. Weights Mode
학습된 모델의 가중치(weights)를 시각화합니다. LSTM 신경망의 weights를 그래프로 확인할 수 있습니다.

```bash
python -m path_planner.main weights \
    --model-name model.pth
```

* `--model-name`: 가중치를 시각화할 model 이름 설정 (기본값: `model.pth`)

### 7.3. Map Mode
저장된 맵 파일을 시각화하고, 초기 경로를 확인합니다. 장애물 분포 및 로봇의 초기 위치와 방향을 확인할 때 유용합니다.

```bash
python -m path_planner.main map \
    --map-rel-path test/test_map_L1_0000.npy
```

* `--map-save-dir`: Map 파일이 저장되어 있는 폴더 경로 설정 (기본값: `src/path_planner/path_planner/maps`)
* `--map-rel-path`: 시각화할 맵 경로 (기본값: `test/test_map_L1_0000.npy`)


## 8. 명령어 도움말

```bash
python -m path_planner.main -h
python -m path_planner.main train -h
python -m path_planner.main test -h
python -m path_planner.main gif -h
python -m path_planner.main inference -h
python -m path_planner.main weights -h
python -m path_planner.main map -h
```

`pip install -e src/path_planner`를 다시 실행한 뒤에는 아래 console command도 사용할 수 있습니다.

```bash
path-planner -h
```