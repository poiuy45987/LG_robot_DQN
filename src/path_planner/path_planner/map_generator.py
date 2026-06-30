from dataclasses import dataclass
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import re
import zipfile

from path_planner.config import MapConfig, DEFAULT_SEED, MAP_SAVE_DIR, MAP_FILE_FORMAT, MAP_FILE_REGEX, GridCell
from path_planner.utils.visualizer import get_map_img, display_image

@dataclass
class ObstacleMap:
    obs_map: np.ndarray
    level: int
    H: int
    W: int
    eff_H: int
    eff_W: int
    
    @classmethod
    def load_from_file(cls, file_path: str):
        """파일 경로를 받아 map 파일 제목을 가지고 정보를 뽑아냄"""
        file_name = os.path.basename(file_path)
        match = re.search(MAP_FILE_REGEX, file_name)
        if not match:
            raise ValueError(f"파일명('{file_name}')이 규칙과 일치하지 않습니다.")
            
        obs_map = np.load(file_path)
        gd = match.groupdict()
        grid_size = MapConfig().grid_size
        
        # 정수로 변환하여 복원 (height_m, width_m은 H, W가 있으므로 복원 시에는 패스 가능)
        return cls(
            obs_map=obs_map,
            level=int(gd['level']),
            H=obs_map.shape[0],
            W=obs_map.shape[1],
            eff_H=int(int(gd['height_m'])*100 // grid_size), # 상황에 맞게 보정 가능
            eff_W=int(int(gd['width_m'])*100 // grid_size)
        )
    
class MapGenerator:
    
    def __init__(self, cfg: MapConfig = MapConfig(), rng: np.random.Generator = np.random.default_rng(DEFAULT_SEED)):
        
        self.cfg = cfg; self.rng = rng
        
        # MapGenerator가 만드는 map의 크기 설정: Grid 단위
        self.H = self.cfg.H
        self.W = self.cfg.W
    
    def _set_map_boundary(self, eff_size: tuple[float, float]) -> tuple[int, int]:
        """
        더 크기가 작은 map을 생성하기 위해 벽을 세우는 method

        Args:
            eff_size (tuple[float, float]): Map에서 사용할 영역의 (가로 길이, 세로 길이)를 cm 단위로 받음.

        Returns:
            tuple[int, int]: Map에서 사용할 영역의 (가로 길이, 세로 길이)를 grid 단위로 반환
        """
        # 유효성 검증
        is_valid = True
        if eff_size[0] > self.cfg.map_width:
            print(f"Warning: Active area width ({eff_size[0]:.2f} cm) is larger than the map width ({self.cfg.map_width:.2f} cm)")
            is_valid = False
        if eff_size[1] > self.cfg.map_height:
            print(f"Warning: Active area height ({eff_size[1]:.2f} cm) is larger than the map height ({self.cfg.map_height:.2f} cm)")
            is_valid = False

        # 입력된 eff_size가 유효하면 grid 단위로 변환하여 반환
        if is_valid:
            eff_W = min(int(eff_size[0] // self.cfg.grid_size), self.W)
            eff_H = min(int(eff_size[1] // self.cfg.grid_size), self.H)

            self.obs[:, :] = GridCell.WALL
            self.obs[:eff_H, :eff_W] = GridCell.BLANK
            
            return eff_W, eff_H
        
        # 입력된 eff_size가 유효하지 않으면 None 반환
        return None, None

    # 직사각형을 좀 더 정확하게 그리는 방법
    # Grid 크기가 충분히 작을 때 유용할 것이라고 생각
    def _get_rect_indices_for_big_rect(
        self, cx: float, cy: float, 
        width: float, height: float, 
        theta: float) -> list[tuple[int, int]]:
        """
        Grid map에서 직사각형을 그릴 때, 색칠할 grid의 indices를 출력하는 method

        Args:
            cx, cy: 그릴 직사각형의 중심 좌표
            width, height: 그릴 직사각형의 가로, 세로 길이
            theta: 직사각형이 반시계 방향으로 회전한 정도(rad 단위)
        """
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        grid_indices = []
        
        # 각 grid가 직사각형 영역에 속하는지 검사
        # 검사 범위는 직사각형의 대각선 길이를 한 변의 길이롤 하는 정사각형
        r = math.sqrt(width**2 + height**2) / 2
        x_min = max(0, int(math.floor(cx - r)))
        x_max = min(self.eff_W - 1, int(math.ceil(cx + r)))
        y_min = max(0, int(math.floor(cy - r)))
        y_max = min(self.eff_H - 1, int(math.ceil(cy + r)))

        # 각 그리드 칸의 중심점이 직사각형 내부에 있는지 검사
        for ix in range(x_min, x_max + 1):
            for iy in range(y_min, y_max + 1):
                # 그리드 칸의 중심점: (ix.5, iy.5)
                # (dx, dy): 직사각형 중심점 (cx, cy)를 원점으로 봤을 때 grid 중심점의 좌표
                dx = (ix + 0.5) - cx
                dy = (iy + 0.5) - cy
                
                # 역회전 변환 (Rotated frame -> Local axis-aligned frame)
                nx = dx * cos_t + dy * sin_t
                ny = -dx * sin_t + dy * cos_t
                
                # 범위 판정 (부동소수점 오차 방지를 위해 아주 작은 값 1e-9 추가)
                if abs(nx) <= (width / 2) + 1e-9 and abs(ny) <= (height / 2) + 1e-9:
                    grid_indices.append((ix, iy))
        
        return grid_indices

    def _get_rect_indices_for_small_rect(
        self, cx: float, cy: float, 
        width: float, height: float, 
        theta: float) -> list[tuple[int, int]]:
        """
        Grid map에서 직사각형을 그릴 때, 색칠할 grid의 indices를 출력하는 method

        Args:
            cx, cy: 그릴 직사각형의 중심 좌표
            width, height: 그릴 직사각형의 가로, 세로 길이
            theta: 직사각형이 반시계 방향으로 회전한 정도(rad 단위)
        """
        
        # 직사각형을 그릴 grid의 시작과 끝 좌표를 int 형식으로 얻음
        x_min = max(0, int(cx - width/2 + 0.5))
        x_max = min(self.eff_W, int(cx + width/2 + 0.5))
        y_min = max(0, int(cy - height/2 + 0.5))
        y_max = min(self.eff_H, int(cy + height/2 + 0.5))
        
        grid_indices = [(x, y) for x in range(x_min, x_max) for y in range(y_min, y_max)]
        
        return grid_indices

    def _add_table_legs(
        self, cx: float, cy: float,
        table_width: int, table_height: int,
        table_leg_size: int,
        theta: float = 0.0,   # 회전 각도 (rad)
    ) -> bool:
        """_summary_

        Args:
            cx, cy (int): Table의 중심 좌표
            table_row_size (int): Table의 가로 길이
            table_col_size (int): Table의 세로 길이
            table_leg_size (int, optional): Table 다리의 두께. Defaults to 2.
            theta (float, optional): Table이 반시계 방향으로 회전한 각도. Defaults to 0.0.

        Returns:
            bool: Table 배치가 성공했는지 여부
        """
        # cx, cy: 테이블 중심 좌표, row_size 또는 col_size가 짝수인 경우 중심 위치로부터 약간 틀어져 있음
        # 테이블 기본 다리 좌표 (local frame)
        
        # 테이블의 각도가 0 rad일 때 각 leg 중심의 상대적인 좌표
        no_rotate_leg_centers = [
            (-table_width/2.0 + table_leg_size/2.0, -table_height/2.0 + table_leg_size/2.0),
            ( table_width/2.0 - table_leg_size/2.0, -table_height/2.0 + table_leg_size/2.0),
            (-table_width/2.0 + table_leg_size/2.0,  table_height/2.0 - table_leg_size/2.0),
            ( table_width/2.0 - table_leg_size/2.0,  table_height/2.0 - table_leg_size/2.0),
        ]
        
        leg_grid_indices = []
        
        c = math.cos(theta)
        s = math.sin(theta)

        for px, py in no_rotate_leg_centers:
            # (px, py): Table 중심을 원점으로 했을 때, table leg의 중심 위치
            
            # Table이 돌아간 각도 theta에 따라 (px, py)를 회전 s변환
            rx = c * px - s * py
            ry = s * px + c * py

            # (x, y): Table leg 중심 위치의 global coordinate
            x = cx + rx; y = cy + ry
            
            # Table 다리가 차지하는 grid의 indices를 얻음
            if table_leg_size < 10:
                indices = self._get_rect_indices_for_small_rect(x, y, table_leg_size, table_leg_size, theta)
            else:
                indices = self._get_rect_indices_for_big_rect(x, y, table_leg_size, table_leg_size, theta)
            leg_grid_indices.extend(indices)
        
        if not leg_grid_indices:
            return False
        
        leg_grid_indices_np = np.array(leg_grid_indices)
        
        # 장애물과 table 다리 위치가 겹치면 table을 배치하지 않음    
        if not np.all(self.obs[leg_grid_indices_np[:, 1], leg_grid_indices_np[:, 0]] == GridCell.BLANK):
            return False
        
        # 장애물이 겹치지 않으면 table을 배치
        self.obs[leg_grid_indices_np[:, 1], leg_grid_indices_np[:, 0]] = GridCell.TABLE
        
        return True

    def _add_chair_legs(
        self, cx: int, cy: int,
        chair_width: int, chair_height: int,
        chair_leg_size: int = 1,
        theta: float = 0.0,   # 회전 각도 (rad)
    ) -> bool:
        """_summary_

        Args:
            cx, cy (int): Chair의 중심 좌표
            chair_width (int): Chair의 가로 길이
            chair_height (int): Chair의 세로 길이
            chair_leg_size (int, optional): Chair 다리의 두께. Defaults to 1.
            theta (float, optional): Chair이 반시계 방향으로 회전한 각도. Defaults to 0.0.

        Returns:
            bool: Table 배치가 성공했는지 여부
        """
        # cx, cy: 테이블 중심 좌표, row_size 또는 col_size가 짝수인 경우 중심 위치로부터 약간 틀어져 있음
        # 테이블 기본 다리 좌표 (local frame)
        
        # 테이블의 각도가 0 rad일 때 각 leg 중심의 상대적인 좌표
        no_rotate_leg_centers = [
            (-chair_width/2.0 + chair_leg_size/2.0, -chair_height/2.0 + chair_leg_size/2.0),
            ( chair_width/2.0 - chair_leg_size/2.0, -chair_height/2.0 + chair_leg_size/2.0),
            (-chair_width/2.0 + chair_leg_size/2.0,  chair_height/2.0 - chair_leg_size/2.0),
            ( chair_width/2.0 - chair_leg_size/2.0,  chair_height/2.0 - chair_leg_size/2.0),
        ]
        
        leg_grid_indices = []
        
        c = math.cos(theta)
        s = math.sin(theta)

        for px, py in no_rotate_leg_centers:
            # (px, py): Table 중심을 원점으로 했을 때, table leg의 중심 위치
            
            # Table이 돌아간 각도 theta에 따라 (px, py)를 회전 변환
            rx = c * px - s * py
            ry = s * px + c * py

            # (x, y): Table leg 중심 위치의 global coordinate
            x = cx + rx; y = cy + ry
            
            # Table 다리가 차지하는 grid의 indices를 얻음
            if chair_leg_size < 10:
                indices = self._get_rect_indices_for_small_rect(x, y, chair_leg_size, chair_leg_size, theta)
            else:
                indices = self._get_rect_indices_for_big_rect(x, y, chair_leg_size, chair_leg_size, theta)
            leg_grid_indices.extend(indices)
        
        if not leg_grid_indices:
            return False
        
        leg_grid_indices_np = np.array(leg_grid_indices)
        
        # 장애물과 table 다리 위치가 겹치면 table을 배치하지 않음    
        if not np.all(self.obs[leg_grid_indices_np[:, 1], leg_grid_indices_np[:, 0]] == GridCell.BLANK):
            return False
        
        # 장애물이 겹치지 않으면 table을 배치
        self.obs[leg_grid_indices_np[:, 1], leg_grid_indices_np[:, 0]] = GridCell.CHAIR
        return True

    def _add_chairs_around_table(
        self,
        cx: int, cy: int, 
        table_width: int, table_height: int,
        num_chairs: int,
        theta: float,
        chair_dist_offset: int = 4,
        #chair_size_offset: int = 2,
        max_chair_size: int = 24,
        min_chair_size: int = 20,
        chair_leg_size: int = 1,
    ):
        
        radius_angle_set = [
            (0, table_width/2.0 + chair_dist_offset),
            (math.pi/2, table_height/2.0 + chair_dist_offset),
            (math.pi, table_width/2.0 + chair_dist_offset),
            (3*math.pi/2, table_height/2.0 + chair_dist_offset),
        ]
        self.rng.shuffle(radius_angle_set)
        radius_angle_set = radius_angle_set[:num_chairs]

        for (a, r) in radius_angle_set:
            
            chair_size_local = max_chair_size
            while chair_size_local >= min_chair_size:
                # 각도/거리 노이즈
                ang = theta + a + self.rng.normal(scale=0.1)
                radius = r + self.rng.integers(-1, 2)

                # 의자의 중심 좌표
                x = int(round(cx + radius * math.cos(ang)))
                y = int(round(cy + radius * math.sin(ang)))
                
                # 의자가 배치 가능한 경우 의자 다리를 생성
                if self._add_chair_legs(
                    x, y, 
                    chair_width=chair_size_local, 
                    chair_height=chair_size_local, 
                    chair_leg_size=chair_leg_size, 
                    theta=ang):
                    break
                # 의자 다리가 장애물과 겹치거나 맵을 넘는 경우 의자 크기를 줄여가며 재시도
                else:
                    chair_size_local -= 1

    def _is_placeable(
        self,
        x_min: int, x_max: int,
        y_min: int, y_max: int,
    ) -> bool:
        
        x_min = max(0, x_min-1); x_max = min(self.eff_W, x_max+1)
        y_min = max(0, y_min-1); y_max = min(self.eff_H, y_max+1)
        if np.all(self.obs[y_min:y_max, x_min:x_max] == GridCell.BLANK):
            return True
        return False
        
    def _add_small_obstacles_in_window(
        self,
        x_min: int, x_max: int,
        y_min: int, y_max: int,
        obs_num: int, 
        small_obs_size_max: int,
        small_obs_size_min: int
    ):
        # Window에 장애물이 하나도 없는 경우에만 장애물을 배치
        if np.all(self.obs[y_min:y_max, x_min:x_max] == GridCell.BLANK):
            for _ in range(obs_num):
                obs_width = self.rng.integers(small_obs_size_min, small_obs_size_max)
                obs_height = self.rng.integers(small_obs_size_min, small_obs_size_max)
                x = self.rng.integers(x_min, x_max-obs_width)
                y = self.rng.integers(y_min, y_max-obs_height)
                if self._is_placeable(
                    x_min=x, x_max=x+obs_width,
                    y_min=y, y_max=y+obs_height,
                ):
                    self.obs[y:y+obs_height, x:x+obs_width] = GridCell.MORE_OBS
  
    def generate_house_like_obstacles(self, 
                                      level: int = 1,
                                      eff_size: tuple[float, float] = None,
                                      visualize: bool = False) -> ObstacleMap:
        """
        책상, 의자, 작은 장애물 등 집과 비슷한 형태의 map을 형성하는 method

        Args:
            level (int, optional): map 난이도. 1~3까지 설정 가능. 높을수록 장애물이 많음. Defaults to 1.
            visualize (bool, optional): Map 시각화 시 책상, 의자, 작은 장애물을 구별해서 보여줄지 여부. Defaults to False.
            eff_size (tuple[float, float], optional): Map의 공간을 제한적으로 사용하고 싶을 때, 사용할 영역의 (가로 길이, 세로 길이)를 cm 단위로 입력. Defaults to None.

        Returns:
            ObstacleMap: Obstacle이 배치된 grid map의 numpy 배열과 관련 정보.
        """
        max_table_size = self.cfg.max_table_size
        min_table_size = self.cfg.min_table_size
        table_leg_size = self.cfg.table_leg_size
        
        max_chair_size = self.cfg.max_chair_size
        min_chair_size = self.cfg.min_chair_size
        chair_leg_size = self.cfg.chair_leg_size
        chair_spread = self.cfg.chair_spread
        
        self.obs = np.zeros((self.H, self.W), dtype=np.uint8)
        self.eff_H, self.eff_W = self.H, self.W
        if eff_size is not None:
            self.eff_W, self.eff_H = self._set_map_boundary(eff_size)
            if self.eff_W is None:
                self.eff_W = self.W
            if self.eff_H is None:
                self.eff_H = self.H
        
        # 책상과 의자를 배치
        table_centers = [] # 테이블 중심 좌표와 테이블 radius를 저장
        max_trial = 20000
        for table_num in range(self.cfg.num_tables):
            for _ in range(max_trial):  # 최대 20000번 시도
                
                # 테이블의 크기 설정
                table_width = self.rng.integers(min_table_size, max_table_size + 1)
                table_height = self.rng.integers(min_table_size, max_table_size + 1)
                
                table_radius = int(math.ceil(max(
                    math.sqrt((table_width/2.0)**2 + (table_height/2.0)**2),
                    math.sqrt((table_width/2.0 + max_chair_size/2.0 + chair_spread)**2 + (max_chair_size/2.0)**2),
                    math.sqrt((table_height/2.0 + max_chair_size/2.0 + chair_spread)**2 + (max_chair_size/2.0)**2),
                ))) + 2
                
                # 테이블의 중심 좌표 설정: (cx, cy)
                if table_width >= self.eff_W - table_radius:
                    continue
                if table_height >= self.eff_H - table_radius:
                    continue
                cx = int(self.rng.integers(table_radius, self.eff_W-table_radius))
                cy = int(self.rng.integers(table_radius, self.eff_H-table_radius))
                cx += (table_width % 2) / 2.0
                cy += (table_height % 2) / 2.0
                
                ok = True # 테이블과 의자가 겹치지 않고 배치될 수 있는지 알려주는 flag
                
                # 테이블이 겹치는지 여부를 테이블 중심 좌표로 대략적으로 확인
                for (px, py, pr) in table_centers:
                    if (cx - px)**2 + (cy - py)**2 < (table_radius + pr)**2:
                        ok = False
                        break # 겹치는 테이블이 있는 경우, cx, cy, table_width, table_height를 재생성
                        
                # 테이블 중심 좌표로 판단했을 때 테이블이 다른 장애물과 겹치지 않으면 테이블 배치를 시도
                if ok: 
                    
                    theta = self.rng.uniform(-math.pi/6, math.pi/6) # 테이블이 회전한 각도 결정
                    
                    # 테이블을 map에 추가. 장애물이 겹치지 않은 경우 의자 배치 시도
                    if self._add_table_legs(
                        cx, cy, 
                        table_width, table_height,
                        table_leg_size=table_leg_size,
                        theta=theta
                    ):
                        table_centers.append((cx, cy, table_radius)) # 테이블의 중심 좌표를 list에 추가
                    else:
                        ok = False
                        continue # cx, cy, table_width, table_height를 재생성
                
                # 테이블을 실제로 배치했을 때, 다른 장애물과 겹치지 않으면 의자 배치를 시도
                if ok:

                    # 의자 개수 결정
                    num_chairs = int(self.rng.integers(self.cfg.chairs_per_table_min, self.cfg.chairs_per_table_max + 1))
                    
                    # 의자  
                    self._add_chairs_around_table(
                        cx, cy,
                        table_width, table_height,
                        num_chairs=num_chairs,
                        theta=theta,
                        chair_dist_offset=chair_spread,
                        max_chair_size=max_chair_size,
                        min_chair_size=min_chair_size,
                        chair_leg_size=chair_leg_size,
                    )
                    break # 의자 배치까지 완료했으면 책상 배치를 그만 시도
            
            #else:
                #print(f"Table {table_num+1}: Failed to place a table after max trials.")
                        
        # 고밀도 환경 조성을 위해 작은 장애물을 더 배치
        window_size = self.cfg.window_size
        gap = int(window_size // 3)
        small_obs_size_max = self.cfg.small_obs_size_max
        small_obs_size_min = self.cfg.small_obs_size_min
        
        for x in range(0, self.eff_W, gap):
            for y in range(0, self.eff_H, gap):
                
                obs_num = self.rng.integers(self.cfg.small_obs_num_per_window_min, self.cfg.small_obs_num_per_window_max+1)
                x_last = min(self.eff_W, x+window_size)
                y_last = min(self.eff_H, y+window_size)
                obs_num = int(obs_num * (x_last-x)*(y_last-y) / (window_size**2))
                self._add_small_obstacles_in_window(
                    x_min=x, x_max=x_last,
                    y_min=y, y_max=y_last,
                    obs_num=obs_num,
                    small_obs_size_max=small_obs_size_max,
                    small_obs_size_min=small_obs_size_min
                )

        if not visualize:
            self.obs = (self.obs != 0).astype(np.uint8)
        
        return ObstacleMap(
            obs_map=self.obs,
            level=level,
            H=self.H,
            W=self.W,
            eff_H=self.eff_H,
            eff_W=self.eff_W
        )
        
def generate_multiple_maps(mode: str = "test", 
                           map_num_per_cond: int = 10,
                           seed: int = DEFAULT_SEED, 
                           visualize: bool = False, 
                           map_size_list: list[tuple[float, float]] = [(400, 400), (400, 300), (300, 300)],
                           level_range: tuple[int, int] = (1, 3)):
    """
    Test map을 여러 개 얻는 method

    Args:
        mode (str, optional): 생성한 map의 용도: Train용 또는 test용. Defaults to "test".
        map_num_per_cond (int): Map 크기, level 등 조건별로 생성할 map 개수
        seed (int): Random seed
        visualize (bool, optional): Map 생성 후 시각화할지 여부. Defaults to False.
        map_size_list (list[tuple[float, float]], optional): Map 크기 조건을 (가로, 세로) 형태로 cm 단위로 입력.
        level_range (tuple[int, int], optional): Map 난이도 조건을 (min_level, max_level) 형태로 입력. Defaults to (1, 3).
    """
    
    assert mode in ["train", "test"]
    
    if mode == "test":
        seed += 100000  # test map은 train map과 겹치지 않도록 seed를 100000 이상으로 설정
    rng = np.random.default_rng(seed=seed)
    map_generator = MapGenerator(rng)
    
    # Map을 저장할 폴더 생성
    map_folder_name = os.path.join(MAP_SAVE_DIR, f"{mode}")
    png_folder_name = os.path.join(map_folder_name, "image")
    os.makedirs(png_folder_name, exist_ok=True) # map_folder_name은 상위 폴더이므로 자동 생성됨.
    
    # fig와 canvas 생성: png 저장을 위함
    fig = None; canvas = None
    if visualize:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
    
    # Map 생성 및 저장
    min_level, max_level = level_range
    for width, height in map_size_list:
        width_m = width / 100.0; height_m = height / 100.0
        for level in range(min_level, max_level + 1):
            for map_id in range(map_num_per_cond):
                
                map_file_name = MAP_FILE_FORMAT.format(
                    mode=mode,
                    height_m=int(height_m),
                    width_m=int(width_m),
                    level=level,
                    map_id=map_id+1
                )
                png_file_name = map_file_name.replace(".npy", ".png")
                
                # Map에서 실제 청소기가 움직이고 장애물이 배치될 영역 정의.
                eff_size = (width, height)
                
                # Map에 장애물 배치
                obs_map = map_generator.generate_house_like_obstacles(level=level, eff_size=eff_size, visualize=visualize)
                
                # fig에 map 시각화 후 png로 저장
                if visualize:
                    fig.clear()
                    map_img = get_map_img(obs_map.obs_map, fig=fig, canvas=canvas, map_name=map_file_name, visualized=visualize)
                    fig.savefig(os.path.join(png_folder_name, png_file_name), bbox_inches='tight', dpi=300)

                # map을 npy로 저장
                obstacles = (obs_map.obs_map != 0).astype(np.uint8)
                np.save(os.path.join(map_folder_name, map_file_name), obstacles)

def generate_map_by_seed_and_visualize(seed: int = DEFAULT_SEED, level: int = 1, eff_size: tuple[float, float] = None):

    rng = np.random.default_rng(seed=seed)  # 재현성을 위해 시드 설정
    map_generator = MapGenerator(rng=rng)

    # Map 생성
    obs_map = map_generator.generate_house_like_obstacles(level=level, eff_size=eff_size, visualize=True)
    
    # Map 시각화
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    map_img = get_map_img(obs_map.obs_map, 
                          fig=fig, canvas=canvas, 
                          map_name=f"Map size: {obs_map.eff_H} cm X {obs_map.eff_W} cm / Map seed: {seed}", 
                          visualized=True)
    display_image(map_img)
    
def zip_map_files(zip_file_name: str = 'maps.zip', mode: str = "test"):
        
    map_folder_name = os.path.join(MAP_SAVE_DIR, f"{mode}")
    if not os.path.exists(map_folder_name):
        print(f"Error: {map_folder_name} 폴더가 존재하지 않습니다.")
        return
    
    zip_file_name = os.path.join(map_folder_name, zip_file_name)
    with zipfile.ZipFile(zip_file_name, 'w') as zip_file:
        for file_name in os.listdir(map_folder_name):
            if file_name.endswith('.npy'):
                zip_file.write(os.path.join(map_folder_name, file_name), arcname=file_name)
    
    print(f"압축 완료: {zip_file_name}")

# 실행
if __name__ == "__main__":
    generate_map_by_seed_and_visualize(seed=DEFAULT_SEED, eff_size=None)