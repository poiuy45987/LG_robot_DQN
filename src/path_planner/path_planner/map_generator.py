from dataclasses import dataclass
import numpy as np
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import re
import zipfile

from path_planner.config import MapConfig, DEFAULT_SEED, MAP_SAVE_DIR, MAP_FILE_FORMAT, MAP_FILE_REGEX, GridCell
from path_planner.utils.map_utils import (
    BoundingBox, 
    get_eff_size_from_obs_map, 
    crop_obstacle_area, 
    get_rect_indices_for_big_rect, 
    get_rect_indices_for_small_rect,
    get_leg_indices,
    get_circle_indices
)
from path_planner.utils.visualizer import get_map_img, display_image

@dataclass
class ObstacleMap:
    obs_map: np.ndarray
    level: int
    H: int
    W: int
    eff_size: BoundingBox
    
    @classmethod
    def load_from_file(cls, file_path: str):
        """파일 경로를 받아 map 파일 제목을 가지고 정보를 뽑아냄"""
        file_name = os.path.basename(file_path)
        match = re.search(MAP_FILE_REGEX, file_name)
        if not match:
            raise ValueError(f"파일명('{file_name}')이 규칙과 일치하지 않습니다.")
            
        obs_map = np.load(file_path)
        gd = match.groupdict()

        return cls(
            obs_map=obs_map,
            level=int(gd['level']),
            H=obs_map.shape[0],
            W=obs_map.shape[1],
            eff_size=get_eff_size_from_obs_map(obs_map, GridCell.BLANK)
        )
    
class MapGenerator:
    
    def __init__(self, map_cfg: MapConfig = MapConfig(), rng: np.random.Generator = np.random.default_rng(DEFAULT_SEED)):
        
        self.cfg = map_cfg; self.rng = rng
        
        # MapGenerator가 만드는 map의 크기 설정: Grid 단위
        self.init_H = self.cfg.init_H
        self.init_W = self.cfg.init_W
  
    def generate_house_like_obstacles(self,
                                      robot_diameter: int,
                                      level: int = 1,
                                      map_size: tuple[int, int] = None,
                                      visualize: bool = False,) -> ObstacleMap:
        """
        책상, 의자, 작은 장애물 등 집과 비슷한 형태의 map을 형성하는 method

        Args:
            level (int, optional): map 난이도. 1~4까지 설정 가능. 높을수록 장애물이 많음. Defaults to 1.
            visualize (bool, optional): Map 시각화 시 책상, 의자, 작은 장애물을 구별해서 보여줄지 여부. Defaults to False.

        Returns:
            ObstacleMap: Obstacle이 배치된 grid map의 numpy 배열과 관련 정보.
        """
        max_table_size = self.cfg.max_table_size
        min_table_size = self.cfg.min_table_size
        table_leg_size = self.cfg.table_leg_size
        
        num_chairs = self.cfg.num_chairs_per_level.get(level, 4)
        max_chair_size = self.cfg.max_chair_size
        min_chair_size = self.cfg.min_chair_size
        chair_leg_size = self.cfg.chair_leg_size
        
        if map_size is None:
            self.obs = np.full((self.init_H, self.init_W), GridCell.BLANK, dtype=np.uint8)
        else:
            H, W = map_size
            self.obs = np.full((H, W), GridCell.BLANK, dtype=np.uint8)
        H, W = self.obs.shape

        # 책상과 의자를 세트로 배치
        table_mask = np.zeros((H, W), dtype=np.uint8) # 책상 상판을 표시하는 mask. 책상 상판끼리 겹치면 안 됨.
        chair_mask = np.zeros((H, W), dtype=np.uint8) # 의자 상판을 표시하는 mask. 의자 상판끼리 겹치거나 책상 다리와 겹치면 안 됨.
        max_trial = 20000
        for table_id in range(1, self.cfg.num_tables + 1):
            
            # 테이블 배치 시도
            put_table = False
            table_width = 0
            table_height = 0
            cx = 0; cy = 0 # 책상의 중심 좌표
            for _ in range(max_trial):  # 최대 20000번 시도
                
                # 테이블의 크기 설정
                table_width = self.rng.integers(min_table_size, max_table_size + 1)
                table_height = self.rng.integers(min_table_size, max_table_size + 1)
                table_radius = math.ceil(math.sqrt((table_width/2)**2 + (table_height/2)**2))
                if table_radius >= (W - table_radius) or table_radius >= (H - table_radius): # 중심 좌표 설정을 위한 확인 작업
                    continue
                
                # 테이블 중심 좌표 및 돌아간 각도 설정
                cx = int(self.rng.integers(table_radius, W - table_radius))
                cy = int(self.rng.integers(table_radius, H - table_radius))
                cx += (table_width % 2) / 2.0
                cy += (table_height % 2) / 2.0
                theta = self.rng.uniform(-math.pi/6, math.pi/6) # 테이블이 회전한 각도 결정
                
                # Table 배치 가능성 평가
                table_area_indices = np.array(get_rect_indices_for_big_rect(cx, cy, table_width, table_height, W, H, theta))
                table_leg_indices = get_leg_indices(cx, cy, W, H, table_width, table_height, table_leg_size, theta)
                
                if len(table_area_indices) == 0 or len(table_leg_indices) == 0:
                    continue
                
                area_xs, area_ys = table_area_indices[:, 0], table_area_indices[:, 1]
                leg_xs, leg_ys = table_leg_indices[:, 0], table_leg_indices[:, 1]
                
                # - 조건 1: 테이블 상판이 기존 테이블 상판들과 겹치는가
                # - 조건 2: 테이블 다리가 기존 의자 상판들과 겹치는가 (의자 위에 다리 놓기 방지)
                # - 조건 3: 테이블 다리 자리에 이미 다른 장애물(벽 등)이 있는가
                if table_mask[area_ys, area_xs].any() or \
                   chair_mask[leg_ys, leg_xs].any() or \
                   (self.obs[leg_ys, leg_xs] != GridCell.BLANK).any():
                    continue

                put_table = True
                table_mask[area_ys, area_xs] = table_id
                self.obs[leg_ys, leg_xs] = GridCell.TABLE
                break
            
            # 책상을 배치한 경우, 의자 배치를 시도
            if put_table:
                
                chair_slots = list(range(num_chairs)) # 책상으로부터 의자의 위치를 선택하기 위한 index pool
                chair_pos_ang_max_disturbance = math.pi/2/num_chairs # 의자 위치 선정을 위한 각도 disturbance 범위
                
                for chair_id in range(1, num_chairs+1):
                    
                    put_chair = False
                    for _ in range(max_trial):
                        
                        if not chair_slots:
                            break
                        
                        # 책상 주변에 의자를 배치할 위치를 선택: slot을 기준으로 의자를 배치
                        slot_idx = self.rng.integers(0, len(chair_slots))
                        chosen_slot = chair_slots[slot_idx]

                        # 선택된 slot을 기준으로 책상으로부터 의자를 배치할 방향을 선택
                        base_angle = (2*math.pi/num_chairs)*chosen_slot
                        ang_disturbance = self.rng.uniform(-chair_pos_ang_max_disturbance, chair_pos_ang_max_disturbance)
                        direction_theta = base_angle+ang_disturbance
                        
                        # 의자 크기 선택: 의자는 좀 더 작아질 수 있음
                        chair_size_offset = self.rng.integers(min_chair_size, max_chair_size, endpoint=True)
                        
                        # 책상 중심으로부터 의자 중심까지의 거리의 offset 설정: 책상 모서리 부근에서는 의자를 더 멀리 배치
                        temp_h = table_height - chair_size_offset - 2*table_leg_size
                        th1 = math.atan2(temp_h, table_width) if temp_h > 0 else 0
                        temp_w = table_width - chair_size_offset - 2*table_leg_size
                        th2 = math.atan2(table_height, temp_w) if temp_w > 0 else math.pi/2
                        
                        dist1 = abs(table_width/(2*math.cos(direction_theta) + 1e-6))
                        dist2 = abs(table_height/(2*math.sin(direction_theta) + 1e-6))
                        chair_dist_offset = min(dist1, dist2)
                        th = math.atan2(abs(math.sin(direction_theta)), abs(math.cos(direction_theta))) # direction_theta를 0~90 deg의 각도로 mapping
                        if th1 <= th <= th2:
                            chair_dist_offset += chair_size_offset / math.sqrt(2)
                        
                        # 의자가 돌아간 각도에 disturbance 추가
                        theta += self.rng.uniform(-math.pi/8, math.pi/8)
                                                
                        # 의자 배치
                        for chair_size in range(chair_size_offset, min_chair_size-1, -1): # 결정된 의자 위치 및 의자 배향에서 의자 배치가 불가능하면 의자 크기를 점점 줄임.
                            
                            # 책상으로부터 의자까지의 거리에 disturbance 추가
                            chair_pos_dist_max_disturbance = int(chair_size/2)
                            dist_disturbance = self.rng.uniform(-chair_pos_dist_max_disturbance/4, chair_pos_dist_max_disturbance)
                            chair_dist = chair_dist_offset + dist_disturbance
                            
                            # 의자의 중심 좌표 설정
                            chair_cx = cx + chair_dist*math.cos(direction_theta)
                            chair_cy = cy + chair_dist*math.sin(direction_theta)
                            
                            chair_area_indices = np.array(get_rect_indices_for_big_rect(chair_cx, chair_cy, chair_size, chair_size, W, H, theta))
                            chair_leg_indices = get_leg_indices(chair_cx, chair_cy, W, H, chair_size, chair_size, chair_leg_size, theta)
                            
                            if len(chair_area_indices) == 0 or len(chair_leg_indices) == 0:
                                continue
                            
                            area_xs, area_ys = chair_area_indices[:, 0], chair_area_indices[:, 1]
                            leg_xs, leg_ys = chair_leg_indices[:, 0], chair_leg_indices[:, 1]
                            
                            # - 조건 1: 의자 상판이 기존 의자 상판들과 겹치는가
                            # - 조건 2: 의자 상판에 다른 장애물이 있는가 (의자 위에 다리 놓기 방지)
                            # - 조건 3: 의자 다리 자리에 이미 다른 장애물(벽 등)이 있는가
                            if chair_mask[area_ys, area_xs].any() or \
                               (self.obs[area_ys, area_xs] != GridCell.BLANK).any() or \
                               (self.obs[leg_ys, leg_xs] != GridCell.BLANK).any():
                                continue

                            put_chair = True
                            chair_mask[area_ys, area_xs] = chair_id
                            self.obs[leg_ys, leg_xs] = GridCell.CHAIR
                            break
                        
                        if put_chair:
                            break
        
        # 책상과 의자 배치에 실패하여 장애물이 전혀 없는 경우, 작은 장애물을 임의로 배치
        if not self.obs.any():
            
            window_size = self.cfg.window_size
            gap = window_size // 2
            small_obs_size_max = self.cfg.small_obs_size_max
            small_obs_size_min = self.cfg.small_obs_size_min
            num_obs_per_window = self.cfg.num_small_obs_per_level[level]
            
            for x in range(0, W, gap):
                for y in range(0, H, gap):
                    
                    obs_num = self.rng.integers(num_obs_per_window-1, num_obs_per_window+2) # Window당 장애물 개수는 num_obs_per_window-1 ~ num_obs_per_window+1에서 랜덤 선택
                    
                    # Window의 끝부분 indices
                    x_last = min(W, x+window_size)
                    y_last = min(H, y+window_size)
                    
                    obs_num = int(obs_num * (x_last-x)*(y_last-y) / (window_size**2)) # Window 크기에 비례하여 장애물 개수 재설정
                    
                    # Window에 장애물이 하나도 없는 경우에만 장애물을 배치
                    if np.all(self.obs[y:y_last, x:x_last] == GridCell.BLANK):
                        for _ in range(obs_num):
                            obs_width = self.rng.integers(small_obs_size_min, small_obs_size_max)
                            obs_height = self.rng.integers(small_obs_size_min, small_obs_size_max)
                            obs_x = self.rng.integers(x, x_last-obs_width)
                            obs_y = self.rng.integers(y, y_last-obs_height)
                            if np.all(self.obs[obs_y:obs_y+obs_height, obs_x:obs_x+obs_width] == GridCell.BLANK):
                                self.obs[obs_y:obs_y+obs_height, obs_x:obs_x+obs_width] = GridCell.MORE_OBS
        
        
        # level이 4인 경우, 원형 장애물 추가
        if level == 4:
            num_circular_obstacle = 2
            min_radius = 9; max_radius = 11
            for circle_id in range(num_circular_obstacle):
                for _ in range(max_trial):
                    cx = self.rng.integers(0, W); cy = self.rng.integers(0, H)
                    radius = self.rng.uniform(min_radius, max_radius+1)     
                    circle_indices = np.array(get_circle_indices(cx, cy, radius, W, H))
                    circle_xs, circle_ys = circle_indices[:, 0], circle_indices[:, 1]
                    
                    if np.any(self.obs[circle_ys, circle_xs]):
                        continue
                    
                    self.obs[circle_ys, circle_xs] = GridCell.MORE_OBS
                    break
                
        if not visualize:
            self.obs = (self.obs != 0).astype(np.uint8)
        
        cropped_obs = self.obs
        if map_size is None:
            cropped_obs = crop_obstacle_area(self.obs, robot_diameter, GridCell.BLANK)
            H, W = cropped_obs.shape
        
        return ObstacleMap(
            obs_map=cropped_obs,
            level=level,
            H=H,
            W=W,
            eff_size=BoundingBox(x_min=0, x_max=W, y_min=0, y_max=H)
        )

# FIXME: 수정 필요
def generate_multiple_maps(robot_diameter: int,
                           mode: str = "test",
                           map_num_per_cond: int = 10,
                           seed: int = DEFAULT_SEED, 
                           visualize: bool = False, 
                           map_size: tuple[int, int] | None = None,
                           level_range: tuple[int, int] = (1, 4)):
    """
    Test map을 여러 개 얻는 method

    Args:
        mode (str, optional): 생성한 map의 용도: Train용 또는 test용. Defaults to "test".
        map_num_per_cond (int): Map 크기, level 등 조건별로 생성할 map 개수
        seed (int): Random seed
        visualize (bool, optional): Map 생성 후 시각화할지 여부. Defaults to False.
        map_size (tuple[float, float], optional): Map의 pixel 단위 크기를 (H, W)로 설정. None이면 config에 정의된 map 크기로 설정 후 crop이 진행됨.
        level_range (tuple[int, int], optional): Map 난이도 조건을 (min_level, max_level) 형태로 입력. Defaults to (1, 3).
    """
    
    assert mode in ["train", "test"]
    
    if mode == "test":
        seed += 100000  # test map은 train map과 겹치지 않도록 seed를 100000 이상으로 설정
    cfg = MapConfig()
    rng = np.random.default_rng(seed=seed)
    map_generator = MapGenerator(cfg, rng)
    
    # Map을 저장할 폴더 생성
    map_size_name = f"{map_size[0]}x{map_size[1]}" if map_size is not None else "Cropped"
    map_folder_name = os.path.join(MAP_SAVE_DIR, mode, map_size_name)
    png_folder_name = os.path.join(map_folder_name, "image")
    os.makedirs(png_folder_name, exist_ok=True) # map_folder_name은 상위 폴더이므로 자동 생성됨.
    
    # fig와 canvas 생성: png 저장을 위함
    fig = None; canvas = None
    if visualize:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
    
    # Map 생성 및 저장
    min_level, max_level = level_range
    for level in range(min_level, max_level + 1):
        for map_id in range(map_num_per_cond):
            
            # Map에 장애물 배치
            obs_map = map_generator.generate_house_like_obstacles(robot_diameter=robot_diameter, 
                                                                  level=level, 
                                                                  map_size=map_size,
                                                                  visualize=visualize)
            
            # Map file, png file 이름 설정
            H, W = obs_map.obs_map.shape
            map_file_name = MAP_FILE_FORMAT.format(
                mode=mode,
                H=H,
                W=W,
                level=level,
                map_id=map_id+1
            )
            png_file_name = map_file_name.replace(".npy", ".png")
            
            # fig에 map 시각화 후 png로 저장
            if visualize:
                fig.clear()
                map_img = get_map_img(obs_map.obs_map, fig=fig, canvas=canvas, map_name=map_file_name, visualized=visualize)
                fig.savefig(os.path.join(png_folder_name, png_file_name), bbox_inches='tight', dpi=300)
            
            # map을 npy로 저장
            obstacles = (obs_map.obs_map != 0).astype(np.uint8)
            np.save(os.path.join(map_folder_name, map_file_name), obstacles)


def generate_map_by_seed_and_visualize(robot_diameter: int, seed: int = DEFAULT_SEED, level: int = 1, map_size: tuple[int, int] = None):

    rng = np.random.default_rng(seed=seed)  # 재현성을 위해 시드 설정
    map_generator = MapGenerator(rng=rng)

    # Map 생성
    obs_map = map_generator.generate_house_like_obstacles(robot_diameter=robot_diameter, 
                                                          level=level, 
                                                          map_size=map_size, 
                                                          visualize=True)
    
    # Map 시각화
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    map_img = get_map_img(obs_map.obs_map, 
                          fig=fig, canvas=canvas, 
                          map_name=f"Map size: {obs_map.W} X {obs_map.H} / Map seed: {seed}", 
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