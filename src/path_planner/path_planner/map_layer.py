import numpy as np
from dataclasses import dataclass

from path_planner.config import MapConfig, DEFAULT_SEED, TRACE_MAP_MAX
from path_planner.map_generator import ObstacleMap, MapGenerator
from path_planner.utils.map_utils import *

@dataclass
class MapConfigSchema:
    """
    environment.py에서 map 생성을 주문하는 규격
    2가지 방식으로 주문 가능.
    
    1) file_path만 입력: Map file 제목을 보고 필요한 정보를 추출 후 numpy map data를 그대로 사용
    2) level, eff_H_cm, eff_W_cm 입력: Map의 난이도, map의 크기를 입력하고 map을 직접 생성하도록 함.
    """
    file_path: str | None = None
    level: int | None = None
    H: int | None = None
    W: int | None = None

@dataclass
class MapInfo:
    """
    MapManager에서 저장하는 map의 정보 양식
    """
    level: int
    H: int
    W: int
    eff_size: BoundingBox
    

# @dataclass
# class TraceMapData:
#     """
#     trace_map의 data를 표현하는 양식
#     """
#     indices: np.ndarray
#     value: np.ndarray

@dataclass
class MapHistory:
    """
    Map layer 변화의 history를 저장하는 양식
    """
    new_cleaned_cell_num: int
    new_covered_cells: np.array
    # trace_map_data: TraceMapData
    pos: tuple[float, float]


class MapLayers():
    """
    Map data 관련 연산을 하고 map layer를 update하는 class
    """
    
    def __init__(self, cfg: MapConfig=MapConfig(), 
                 map_rng: np.random.Generator=np.random.default_rng(DEFAULT_SEED), robot_size: int = None):
        
        self.cfg = cfg; self.map_rng = map_rng
        self.map_generator = MapGenerator(self.cfg, self.map_rng)
        
        # Map layer 구성
        # Map이 결정됐을 때 변하지 않는 map layer
        self.obstacles: np.ndarray | None = None       # 장애물의 위치를 표시하는 layer: uint8 [H,W] (장애물: 1, 빈 공간: 0)
        self.collision_map: np.ndarray | None = None   # Obstacle dilated map: uint8 [H,W] (Dilated obstacles: 1, 빈 공간: 0)
        self.reachable: np.ndarray | None = None       # Reachable robot centers: uint8 [H,W] (Reachable center: 1, Unreachable center: 0)
        self.coverable: np.ndarray | None = None       # Coverable cells: uint8 [H,W] (Coverable grid: 1, Uncoverable grid: 0)
        self.mode_map: np.ndarray | None = None
        
        # Map이 결정됐을 때 변하는 map layer
        self.cleaned: np.ndarray | None = None         # 로봇이 청소한 grid를 표시하는 layer: uint8 [H,W] (Cleaned: 1, Uncleaned: 0)
        self.uncleaned: np.ndarray | None = None       # 로봇이 청소하지 않은 grid를 표시하는 layer: uint8 [H,W] (Uncleaned: 1, Cleaned: 0)
        # self.trace: np.ndarray | None = None           # 로봇의 중심이 지나간 잔상을 표시하는 layer: uint8 [H,W] (방문한 수를 표시)
        
        self.map_info: MapInfo | None = None           # Map 정보
        
        # Robot 설정
        self._set_robot(robot_size)
        
        # Coverage 설정
        self.total_coverable_area: int = 0
        self.coveraged_area: int = 0
        self._prev_xs: np.ndarray | None = None
        self._prev_ys: np.ndarray | None = None
        
        # History
        self.map_history: list[MapHistory] = []
        
        # Trace layer를 위한 instant 변수
        # self.active_trace_indices: np.ndarray | None = None


    def reset(self, map_config: MapConfigSchema=None, 
              env_rng: np.random.Generator=np.random.default_rng(DEFAULT_SEED)) -> tuple[int, int]:
        
        # Map을 새로 생성
        map_file_path = None
        if map_config is not None or self.obstacles is None:
            map_cfg = map_config or MapConfigSchema()
            if map_cfg.file_path is not None:
                map_file_path = map_cfg.file_path
                obs_map = ObstacleMap.load_from_file(map_file_path)
            else:
                level = map_cfg.level or 1
                map_size = (map_cfg.H, map_cfg.W) if map_cfg.H is not None else None
                obs_map = self.map_generator.generate_house_like_obstacles(robot_diameter=self.robot_size,
                                                                           level=level, 
                                                                           map_size=map_size)
            
            # Obstacle map과 map 정보 얻음    
            self.obstacles = obs_map.obs_map
            self.map_info = MapInfo(
                level=obs_map.level,
                H=obs_map.H,
                W=obs_map.W,
                eff_size=obs_map.eff_size
            )
            
            # Collision map과 mode map 생성
            self.collision_map = dilation_obstacles(self.obstacles, self.robot_mask)
            self.mode_map = generate_navigation_mode_map(
                obs=self.obstacles,
                crop_size=self.robot_size, 
                robot_diameter=self.robot_size, 
                stride=self.robot_size//4, 
                eff_size=(obs_map.eff_H, obs_map.eff_W),
            )
            
        # 시작점 설정: 설정에 실패하면 None을 반환하여 다시 map을 만들도록 함.
        # 시작점 설정하면서 reachable map, coverable map 설정
        pos = self._get_start_pos(env_rng)
        if pos is None:
            return None
        
        # Cleaned layer, Uncleaned layer, Trace layer 초기화
        self.cleaned = np.zeros((self.map_info.H, self.map_info.W), dtype=np.uint8)
        self.uncleaned = (self.coverable == 1).astype(np.uint8) 
        # self.trace = np.zeros((self.map_info.H, self.map_info.W), dtype=np.uint8)
        # self.active_trace_indices = None
        
        # Map layer와 관련된 parameter 초기화
        self.coveraged_area = 0 # Cover한 영역의 grid 수
        self.total_coverable_area = int(self.coverable.sum(dtype=np.int64))
        self._prev_xs = None; self._prev_ys = None # 이전 step에서 cover했던 grid 좌표: Overlap 계산을 위함
        
        # History 초기화
        self.map_history = []
        
        return pos
    
    
    def update_map_layers(self, cx: float, cy: float) -> tuple[bool, float]:
        
        collided, new_cleaned_degree, revisit_degree, new_cleaned_num, new_cleaned_grid_indices_curr_step = self._update_cleaned_map_and_uncleaned_map(cx, cy)
        # trace_map_data = self._update_trace(cx, cy)
        curr_step_diff = MapHistory(new_cleaned_cell_num=new_cleaned_num,
                                    new_covered_cells=new_cleaned_grid_indices_curr_step,
                                    # trace_map_data=trace_map_data,
                                    pos=(cx, cy))
        self.map_history.append(curr_step_diff)
        
        return collided, new_cleaned_degree, revisit_degree
    
    
    def get_agent_layer(self, cx: float, cy: float):
        
        agent_layer = np.zeros((self.map_info.H, self.map_info.W), dtype=np.uint8)
        
        if cx.is_integer() and cy.is_integer():
            cx = int(cx); cy = int(cy)
            x0, x1 = int(cx-self.robot_half_size), int(cx+self.robot_half_size+1)
            y0, y1 = int(cy-self.robot_half_size), int(cy+self.robot_half_size+1)
            agent_layer[y0:y1, x0:x1] = self.robot_mask
        
        else: 
            xs, ys = self._get_footprint_coords(cx, cy)
            agent_layer[ys, xs] = 1
        
        return agent_layer
    
    def get_raw_patch(self, cx: float, cy: float, crop_radius: int) -> np.ndarray:
        
        full_layer = np.stack([self.collision_map, self.cleaned, self.uncleaned], axis=0)
        current_patch = crop_raw_patch(full_layer, cx, cy, crop_radius, value=[1, 0, 0])
        
        return current_patch
       
    
    def one_step_back(self, crop_radius: int, stack_steps_num: int = 1):
        """
        한 step 이전의 map 상태로 변환

        Args:
            remain_collision_count (bool, optional): 가상의 step을 수행한 뒤 다시 back step을 하는 경우, collision count를 유지해야 함. Defaults to False.
        """        
        last_map_vary_data = self.map_history.pop() # Map을 변화한 방식에 관한 데이터
        last_instance_data = self.map_history[-1] # 가장 마지막 position, direction, step 수 등에 관한 데이터
        
        # Instance variable 변경
        last_pos = last_instance_data.pos
        self._prev_xs, self._prev_ys = self._get_footprint_coords(*last_pos)
        
        # Map layer 변경        
        if last_map_vary_data.new_cleaned_cell_num > 0:
            self.coveraged_area -= last_map_vary_data.new_cleaned_cell_num
        
        if last_map_vary_data.new_covered_cells is not None and len(last_map_vary_data.new_covered_cells[:, 0]) > 0:
            cx = last_map_vary_data.new_covered_cells[:, 0]; cy = last_map_vary_data.new_covered_cells[:, 1]
            self.cleaned[cy, cx] -= 1
            
        # active_trace_indices = last_map_vary_data.trace_map_data.indices
        # active_trace_value = last_map_vary_data.trace_map_data.value
        # self.active_trace_indices = active_trace_indices
        # self.trace = np.zeros((self.map_info.H, self.map_info.W), dtype=np.uint8) # trace를 초기화
        # self.trace[active_trace_indices[:, 1], active_trace_indices[:, 0]] = active_trace_value
        
        self.uncleaned = ((self.obstacles == 0) & (self.cleaned == 0)).astype(np.uint8)
        
        # ----------------- patch_stack 변경을 위해 past_patch 얻음 -----------------
        
        # stack_steps_num 전의 cleaned layer 얻음
        past_cleaned = self.cleaned.copy()
        for i in range(stack_steps_num-1):
            if len(self.map_history) - i >= 2:
                map_vary_data = self.map_history[-1-i]
                if len(map_vary_data.new_covered_cells[:, 0]) > 0:
                    cx = map_vary_data.new_covered_cells[:, 0]; cy = map_vary_data.new_covered_cells[:, 1]
                    past_cleaned[cy, cx] -= 1
        
        # # stack_steps_num 전의 trace layer 얻음
        idx = -stack_steps_num
        if len(self.map_history) < stack_steps_num:
            idx = 0
        past_traj_data = self.map_history[idx]
        # past_trace = np.zeros((self.map_info.H, self.map_info.W), dtype=np.uint8)
        # active_trace_indices = past_traj_data.trace_map_data.indices
        # active_trace_value = past_traj_data.trace_map_data.value
        # past_trace[active_trace_indices[:, 1], active_trace_indices[:, 0]] = active_trace_value
        
        past_uncleaned = ((self.obstacles == 0) & (past_cleaned == 0)).astype(np.uint8)
        
        # crop을 얻고 self.patch_stack에 추가
        cx, cy = past_traj_data.pos
        full_layer = np.stack([self.collision_map, past_cleaned, past_uncleaned], axis=0)
        past_patch = crop_raw_patch(full_layer, cx, cy, crop_radius, value=[1, 0, 0])
        
        return past_patch
    
    
    def collides(self, cx: float, cy: float) -> bool:
        cx, cy = float_to_int_coord(cx, cy)
        return bool(self.collision_map[cy, cx])
    
    def has_uncleaned_grid(self, cx: float, cy: float) -> bool:
        
        if self.collides(cx, cy):
            return False
        
        cx, cy = float_to_int_coord(cx, cy)
        xs, ys = self._get_footprint_coords(cx, cy)
        new_cleaned_mark = (self.cleaned[ys, xs] == 0) & (self.coverable[ys, xs] == 1)
        
        return np.any(new_cleaned_mark)
        
    
        
    def _set_robot(self, robot_size: int):
        
        self.robot_size = robot_size
        self.robot_half_size = int(robot_size // 2)
        self.robot_mask = make_robot_mask(robot_size)
        
        y_idx, x_idx = np.nonzero(self.robot_mask)
        self.robot_mask_offsets = np.column_stack([
            x_idx - self.robot_half_size, 
            y_idx - self.robot_half_size
        ]).astype(np.int32)
        
        self.robot_area = int(self.robot_mask.sum(dtype=np.int64))

    
    def _get_start_pos(self, env_rng: np.random.Generator) -> tuple[int, int] | None:
        """
        로봇의 청소 시작 지점을 선정하는 method. 가장자리에서 시작하도록 함.
        
        [사전 조건]
        - 호출하기 전 self.map_info와 self.collision_map이 현재 map인 self.obstacle와 일치하는 형태로 최신화되어 있어야 함.
        - 이는 self.reset()에서 자동으로 만족하게 짜임.

        Args:
            env_rng (np.random.Generator): 시작 지점 설정을 위한 random generator

        Returns:
            tuple[int, int]: 선정된 시작 지점 출력
        """

        # 가장자리 좌표를 얻음
        map_x_min = self.map_info.eff_size.x_min
        map_x_max = self.map_info.eff_size.x_max
        map_y_min = self.map_info.eff_size.y_min
        map_y_max = self.map_info.eff_size.y_max
        total_area = (map_x_max-map_x_min+1)*(map_y_max-map_y_min+1)
        
        min_x, max_x = map_x_min+self.robot_half_size, map_x_max-self.robot_half_size-1
        min_y, max_y = map_y_min+self.robot_half_size, map_y_max-self.robot_half_size-1
        
        x_range = np.arange(min_x, max_x+1)
        y_range = np.arange(min_y+1, max_y)
        top = np.stack([np.full_like(x_range, max_y), x_range], axis=1)
        bottom = np.stack([np.full_like(x_range, min_y), x_range], axis=1)
        left = np.stack([y_range, np.full_like(y_range, min_x)], axis=1)
        right = np.stack([y_range, np.full_like(y_range, max_x)], axis=1)
        
        edge_pos = np.concatenate([top, bottom, left, right], axis=0)
        
        # 가장자리 좌표 중에 collision이 일어나지 않는 좌표를 선별
        edge_collision_value = self.collision_map[edge_pos[:, 0], edge_pos[:, 1]]
        valid_indices = np.where(edge_collision_value == 0)[0]
        
        # coverable_area_rate이 50% 이상인 위치만 선택
        coverable_area_rate = 0.0
        start_x = None; start_y = None # 시작 위치
        
        for _ in range(100):
        
            if len(valid_indices) > 0:
                chosen_idx = env_rng.choice(valid_indices)
                start_x = edge_pos[chosen_idx, 1]; start_y = edge_pos[chosen_idx, 0]
            else:
                return None

            self.reachable = compute_reachable_centers(self.collision_map, start_x, start_y)
            self.coverable = compute_coverable_cells_from_reachable(self.obstacles, self.reachable, self.robot_mask_offsets)
            self.total_coverable_area = int(self.coverable.sum(dtype=np.int64)) # Cover할 수 있는 영역의 넓이
            coverable_area_rate = self.total_coverable_area / total_area # coverable_area_rate이 너무 낮으면 맵을 재생성해야 함.
            if coverable_area_rate >= 0.5:
                return (float(start_x), float(start_y))
            
        else:
            return None
    
    
    def _in_bounds_center(self, cx: float, cy: float) -> bool:
        cx, cy = float_to_int_coord(cx, cy)
        cx_min = self.map_info.eff_size.x_min + self.robot_half_size
        cx_max = self.map_info.eff_size.x_max - self.robot_half_size - 1
        cy_min = self.map_info.eff_size.y_min + self.robot_half_size
        cy_max = self.map_info.eff_size.y_max - self.robot_half_size - 1
        return (cx_min <= cx <= cx_max) and (cy_min <= cy <= cy_max)
    
    
    def _get_footprint_coords(self, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
        
        cx, cy = float_to_int_coord(cx, cy)
        xs = cx + self.robot_mask_offsets[:, 0]
        ys = cy + self.robot_mask_offsets[:, 1]
        return xs, ys
    
    
    def _update_cleaned_map_and_uncleaned_map(self, cx: float, cy: float) -> tuple[bool, float, float, int, np.ndarray]:
        """
        로봇 중심이 (cx, cy)일 때 cleaned_map, uncleaned_map을 update하는 method

        Args:
            cx, cy (int): 로봇 중심의 좌표
        Return:
            tuple[int, bool]: 
                - new_cleaned_num (int): 새롭게 cover한 grid 수
                - collided (bool): 충돌 여부
                - new_cleaned_grid_indices_curr_step (np.ndarray): 새롭게 cover한 grid의 좌표. (N, 2)의 형태. Cover한 grid의 (x좌표, y좌표)가 N개 나열되어 있음
                - revisit_degree (float): 로봇 청소기가 현재 위치한 영역을 이전에 얼마나 청소했는지 표시하는 인자. 
                    1.0 이상이면 영역의 grid 모두가 이전에 청소한 적이 있음. 숫자가 클수록 더 자주 청소했다는 의미
        """
        new_cleaned_num = 0
        new_cleaned_degree = 0.0
        revisit_degree = 0.0
        new_cleaned_grid_indices_curr_step = None
        
        # 1. 로봇이 맵을 벗어나거나 충돌하는 방향으로 움직일 때, parameter들을 계산하지 않음.
        # 중심 좌표가 맵을 벗어나는 경우
        if not self._in_bounds_center(cx, cy):
            return True, new_cleaned_degree, revisit_degree, new_cleaned_num, new_cleaned_grid_indices_curr_step

        # 로봇이 장애물과 충돌하는 경우
        if self.collides(cx, cy):
            return True, new_cleaned_degree, revisit_degree, new_cleaned_num, new_cleaned_grid_indices_curr_step
        
        # 2. Parameter를 계산하기 위한 indices 정보 계산
        # xs, ys: 현재 로봇이 cover하고 있는 grid indices
        xs, ys = self._get_footprint_coords(cx, cy)
        
        # new_xs_curr_step, new_ys_curr_step: 이전 step에서 cover하지 않았는데 현재 step에서 cover하고 있는 grid indices
        new_xs_curr_step = None; new_ys_curr_step = None  
        if self._prev_xs is not None and self._prev_ys is not None:
            curr_idx = ys * self.map_info.W + xs
            prev_idx = self._prev_ys * self.map_info.W + self._prev_xs
            new_mask = ~np.isin(curr_idx, prev_idx)
            new_xs_curr_step, new_ys_curr_step = xs[new_mask], ys[new_mask]
        else:
            new_xs_curr_step, new_ys_curr_step = xs, ys
        
        # 새롭게 cover한 grid 수 세기
        new_cleaned_mark = (self.cleaned[ys, xs] == 0) & (self.coverable[ys, xs] == 1)
        new_cleaned_num = np.sum(new_cleaned_mark)
        self.coveraged_area += new_cleaned_num
        
        # 3. revisit_degree 계산: 로봇 청소기가 이전에 방문한 grid를 재방문한 정도를 측정
        new_cover_area = len(new_xs_curr_step)
        if new_cover_area == 0:
            new_cleaned_degree = 0.0
            revisit_degree = 0.0
        else:
            new_cleaned_degree = new_cleaned_num / new_cover_area
            revisit_degree = (self.cleaned[new_ys_curr_step, new_xs_curr_step].sum(dtype=np.int64) - new_cover_area) / new_cover_area
        
        # 4. new_xs_curr_step, new_ys_curr_step 보정: cleaned_map 값이 255 이상이면 더 이상 더해지지 않음. 255인 grid를 제외
        under_cleaned_max_mask = self.cleaned[new_ys_curr_step, new_xs_curr_step] < 255 # cleaned_map의 grid가 CLEANED_MAP_MAX 미만인 indices만 선별
        new_xs_curr_step = new_xs_curr_step[under_cleaned_max_mask]
        new_ys_curr_step = new_ys_curr_step[under_cleaned_max_mask]
        new_cleaned_grid_indices_curr_step = np.column_stack((new_xs_curr_step, new_ys_curr_step)) # Shape: (N, 2)
        
        # 5. cleaned_map, uncleaned_map update
        self.cleaned[new_ys_curr_step, new_xs_curr_step] += 1 # cleaned_map 업데이트
        self.uncleaned[ys, xs] = 0 # uncleand_map 업데이트
        self._prev_xs = xs.copy(); self._prev_ys = ys.copy()
        
        return False, new_cleaned_degree, revisit_degree, new_cleaned_num, new_cleaned_grid_indices_curr_step
    
    
    # def _update_trace(self, cx: float, cy: float):
        
    #     # 좌표 정수화
    #     cx, cy = float_to_int_coord(cx, cy)
        
    #     # 기존 활성 인덱스들의 값 일괄 감소
    #     if self.active_trace_indices is not None and len(self.active_trace_indices) > 0:
            
    #         xs = self.active_trace_indices[:, 0]; ys = self.active_trace_indices[:, 1]
            
    #         # 현재 활성화된 좌표들만 1씩 차감
    #         self.trace[ys, xs] -= 1
            
    #         # 값이 0보다 큰 인덱스만 유지 (필터링)
    #         keep_mask = self.trace[ys, xs] > 0
    #         self.active_trace_indices = self.active_trace_indices[keep_mask]

    #     # 새로운 좌표 추가
    #     # 만약 현재 위치가 self.active_trace_indices에 이미 있는 경우, 좌표를 추가하지 않음.
    #     if self.trace[cy, cx] <= 0:
    #         new_coord = np.array([[cx, cy]], dtype=np.uint8)
    #         if self.active_trace_indices is None:
    #             self.active_trace_indices = new_coord
    #         else:
    #             self.active_trace_indices = np.vstack([self.active_trace_indices, new_coord])
        
    #     # 맵 업데이트 (최신 위치는 항상 MAX값)
    #     self.trace[cy, cx] = TRACE_MAP_MAX
        
    #     # trace map data를 형식에 맞게 반환
    #     trace_map_data = TraceMapData(indices = self.active_trace_indices,
    #                                   value = self.trace[self.active_trace_indices[:, 1], self.active_trace_indices[:, 0]])
        
    #     return trace_map_data