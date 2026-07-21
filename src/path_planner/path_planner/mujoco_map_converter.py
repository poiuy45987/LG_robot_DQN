import numpy as np
import mujoco
import trimesh
import math
import argparse
import os

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--env_folder', 
                        type=str, 
                        default="/home/poiuy/lg_robot/src/robot_vacuum_redqn/robot_vacuum_redqn/mujoco_environments/environment/world",
                        help='Mujoco 환경이 모인 folder')
    parser.add_argument('--map_name', type=str, default="robocasa.xml", help='Mujoco 환경 map 이름')
    parser.add_argument('--grid_size', type=float, default=2.0, help=f'Grid 크기 (단위: cm, Default: 2)')
    parser.add_argument('--robot_h', type=float, default=9.34, help=f'Grid 크기 (단위: cm, Default: 9.34)')
    parser.add_argument('--obs_geom_gp', type=int, default=0, help=f'xml에서 장애물 취급할 geom의 group 번호')
    
    args = parser.parse_args()
    
    # map_name에 확장자가 없으면 추가
    name, ext = os.path.splitext(args.map_name)
    if not ext:
        args.map_name = name + ".xml"
    
    return args
    

class MujocoGridConverter:
    """
    MuJoCo XML scene을 파싱하여 기존 코드가 사용하는 2D obstacle grid로 변환.
    반환 포맷은 environment.py의 obstacle map과 동일: (H, W) np.uint8, 0=free, 1=obstacle
    """
    
    def __init__(self, xml_path: str, grid_size: float, robot_height: float = 9.34, obstacle_geom_group: int = 0):
        """
        xml_path:             MuJoCo scene XML 경로
        grid_size:            grid 한 칸 크기 (cm). EnvConfig.grid_size와 반드시 일치시킬 것
        obstacle_geom_group:  장애물로 취급할 geom의 group 번호 (XML에서 group="0" 등으로 지정, Default: 0)
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.obs_geom_gp = obstacle_geom_group
        mujoco.mj_forward(self.model, self.data)
        
        self.grid_size = grid_size / 100.0  # cm → m (MuJoCo는 m 단위)
        self.min_x, self.max_x, self.min_y, self.max_y = self.get_map_info()    # Mujoco map x, y좌표의 제한 범위 (단위: m)
        map_height = self.max_y - self.min_y; map_width = self.max_x - self.min_x
        self.H = int(map_height / self.grid_size)
        self.W = int(map_width / self.grid_size)
        self.robot_H = robot_height / 100.0 # cm → m (MuJoCo는 m 단위)
        
        self.obs_grid_map = np.zeros((self.H, self.W), dtype=np.uint8)  # 2D grid로 바꾼 mujoco map
    
    NO_OBSTACLE = 0 # 장애물, 지면 모두 아님
    UNDERGROUND = 1 # 지면
    OBSTACLE = 2    # 장애물
    
    def _is_obstacle(self, geom_id: int, robot_z: float) -> int:
        """
        ID가 geom_id인 geom이 장애물인지, 지면인지, 둘 중 어느 것도 아닌지 return 

        Args:
            geom_id (int): geom의 id
            robot_z (float): geom의 지면 여부를 판단하는 기준(robot_init_pos의 z좌표)

        Returns:
            int: 
                NO_OBSTACLE = 0 (장애물, 지면 모두 아님)
                UNDERGROUND = 1 (지면)
                OBSTACLE = 2    (장애물)
        """
        
        # Group이 0인 것만 선별
        if self.model.geom_group[geom_id] != self.obs_geom_gp:
            return self.NO_OBSTACLE
        
        # 물리적 충돌 설정 확인
        if self.model.geom_conaffinity[geom_id] == 0:
            return self.NO_OBSTACLE
        
        # 높이 확인: geom의 최대 z좌표가 robot_z보다 낮고, 회전을 z축으로만 하는 경우는 지면으로 취급
        eps = 1e-5
        if self.data.geom_xpos[geom_id][2] + self.model.geom_size[geom_id][2] <= robot_z + eps:
            quat = self.model.geom_quat[geom_id]
            is_horizontal = (abs(quat[1]) < eps and abs(quat[2]) < eps)
            if is_horizontal:
                return self.UNDERGROUND
        
        return self.OBSTACLE
    
    def get_map_info(self) -> tuple[float, float, float, float]:
        """
        Map의 x좌표, y좌표 범위를 계산 + 장애물을 감싸는 bounding box 계산 후 저장(self.obs_bbox에 저장)
        
        Returns:
            min_x: 최소 x좌표
            max_x: 최대 x좌표
            min_y: 최소 y좌표
            max_y: 최대 y좌표
        """
        
        self.robot_z = 0.0
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        self.obs_bbox = []
        
        # 1. 로봇의 위치로 바닥의 z 좌표 파악
        try:
            # 'robot_init_pos' 이름을 가진 geom이 있다면 그 위치의 z값을 사용
            self.robot_z = self.model.geom('robot_init_pos').pos[2]
        except KeyError:
            # 해당 이름의 geom이 없으면 기본값 0.0으로 설정
            self.robot_z = 0.0
        
        # 2. geom으로 맵 전체 크기 결정
        max_floor_z = float('-inf') # 지면의 z좌표
        for i in range(self.model.ngeom):
            
            pos = self.data.geom_xpos[i]
            size = self.model.geom_size[i]
            gtype = self.model.geom_type[i]
            
            # 3. plane type geom으로 맵 크기 결정: 지면이 plane으로 주어진다고 생각
            if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                
                geom_name = self.model.geom(i).name or ""
                if size[0] > 0 and size[1] > 0: # plane의 size가 0인 경우, plane이 무한 평면. 이 경우는 다른 geom의 위치 및 크기로 map 크기 결정
                    # 이름에 floor, ground 등이 있는 경우
                    if any(k in geom_name.lower() for k in ['floor', 'ground']):
                        min_x = min(min_x, pos[0]-size[0]); max_x = max(max_x, pos[0]+size[0])
                        min_y = min(min_y, pos[1]-size[1]); max_y = max(max_y, pos[1]+size[1])
                    
                    # plane의 높이가 robot_z보다 높지 않은 경우, 가장 높은 plane의 크기를 반환
                    if pos[2] <= self.robot_z:
                        if pos[2] > max_floor_z:
                            max_floor_z = pos[2]
                            min_x = min(min_x, pos[0]-size[0]); max_x = max(max_x, pos[0]+size[0])
                            min_y = min(min_y, pos[1]-size[1]); max_y = max(max_y, pos[1]+size[1])
            
            # 4. geom 요소들로 맵 크기 결정
            else:
                obs_type = self._is_obstacle(i, self.robot_z)
                if obs_type != self.NO_OBSTACLE: # 장애물 또는 지면인 경우: map 크기 결정에 사용
                    rot_mat = self.data.geom_xmat[i].reshape(3, 3)
                    
                    # geom type이 mesh인 경우: mesh의 모든 꼭짓점들을 global coordinate으로 변환 -> bbox 결정
                    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                        
                        mesh_id = self.model.geom_dataid[i]
                        
                        start_vert_adr = self.model.mesh_vertadr[mesh_id]
                        vert_num = self.model.mesh_vertnum[mesh_id]
                        vertices = self.model.mesh_vert[start_vert_adr : start_vert_adr + vert_num]
                        
                        world_verts = vertices @ rot_mat.T + pos
        
                        xmin, xmax = np.min(world_verts[:, 0]), np.max(world_verts[:, 0])
                        ymin, ymax = np.min(world_verts[:, 1]), np.max(world_verts[:, 1])
                        zmin, zmax = np.min(world_verts[:, 2]), np.max(world_verts[:, 2])
                    
                    # geom type이 구, 원기둥, 직육면체, 타원체, 캡슐 형태인 경우: 해당 도형을 감싸는 bbox를 기하학적으로 계산    
                    else:
                        dx, dy, dz = 0, 0, 0    # local 좌표계에서 x, y, z 좌표 절댓값의 최댓값
                        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                            dx, dy, dz = size[0], size[0], size[0]
                        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                            dx, dy, dz = size[0], size[0], size[1]
                        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                            dx, dy, dz = size[0], size[0], size[0]+size[1]
                        elif gtype in [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID]:
                            dx, dy, dz = size[0], size[1], size[2]
                        
                        # local 좌표계에서의 bounding box의 각 꼭짓점 좌표 (8개)
                        local_bbox = np.array([
                            [-dx, -dy, -dz],
                            [-dx, -dy,  dz],
                            [-dx,  dy, -dz],
                            [-dx,  dy,  dz],
                            [ dx, -dy, -dz],
                            [ dx, -dy,  dz],
                            [ dx,  dy, -dz],
                            [ dx,  dy,  dz],
                        ])
                        
                        # Orientation에 따라 bounding box 회전
                        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE: # geom이 구 형태면 orientation을 고려할 필요 없음
                            world_bbox = local_bbox + np.array([[pos[0], pos[1], pos[2]]])
                        else:
                            world_bbox = local_bbox @ rot_mat.T + np.array([[pos[0], pos[1], pos[2]]])
                        
                        # 회전한 local bounding box를 감싸는 global bounding box의 x, y, z좌표 범위 
                        xmin, xmax = np.min(world_bbox[:, 0]), np.max(world_bbox[:, 0])
                        ymin, ymax = np.min(world_bbox[:, 1]), np.max(world_bbox[:, 1])
                        zmin, zmax = np.min(world_bbox[:, 2]), np.max(world_bbox[:, 2])
                    
                    if obs_type == self.OBSTACLE: # geom이 지면인 경우, bbox를 저장하지 않음
                        self.obs_bbox.append({
                            'id': i,
                            'gtype': gtype,
                            'bbox': (xmin, xmax, ymin, ymax, zmin, zmax)
                        }) # 장애물을 둘러싸는 bbox 저장
                    
                    # 전체 map 크기 결정을 위해 map 내에서 x, y좌표의 최댓값, 최솟값을 저장
                    min_x = min(xmin, min_x); max_x = max(xmax, max_x)
                    min_y = min(ymin, min_y); max_y = max(ymax, max_y); 
        
        return min_x, max_x, min_y, max_y
    
    def convert_mjc2grid(self):
        """
        Mujoco map을 2D grid로 변경하여 self.obs_grid_map에 저장
        """
        eps = 1e-5
        
        # 로봇이 차지하는 z좌표 목록
        rb_zmin = self.robot_z
        rb_zmax = self.robot_z + self.robot_H
        rb_zsamples = np.arange(rb_zmin, rb_zmax, self.grid_size) # 장애물 존재 여부를 검사할 z좌표 후보
        if rb_zmax - rb_zsamples[-1] > eps:
            rb_zsamples = np.append(rb_zsamples, rb_zmax)
        
        for bbox in self.obs_bbox:
            geom_idx = bbox['id']
            gtype = bbox['gtype']
            xmin, xmax, ymin, ymax, zmin, zmax = bbox['bbox'] # 장애물이 존재하는 bounding box
            
            if((zmin > rb_zmax + eps) or (zmax < rb_zmin - eps)): # 장애물의 bounding box가 로봇의 완전 위, 아래에 있을 경우
                continue

            # 1. BBox 경계를 index 단위로 변환: Grid의 원점 = (self.min_x, self.min_y), min은 버림, max는 올림
            idx_xmin = max(0, math.floor((xmin - self.min_x) / self.grid_size))
            idx_xmax = min(self.W-1, math.ceil((xmax - self.min_x) / self.grid_size))
            idx_ymin = max(0, math.floor((ymin - self.min_y) / self.grid_size))
            idx_ymax = min(self.H-1, math.ceil((ymax - self.min_y) / self.grid_size))

            # 2. 해당 영역의 x, y 좌표 행렬(Grid index) 생성
            cols = np.arange(idx_xmin, idx_xmax + 1)
            rows = np.arange(idx_ymin, idx_ymax + 1)
            yy_idx, xx_idx = np.meshgrid(rows, cols, indexing='ij') # Shape: (idx_ymax-idx_ymin, idx_xmax-idx_xmin)

            # 3. Global coordinate으로 변환: Grid (x, y)가 칠해지는지 여부는 grid의 중점 (x+0.5, y+0.5)가 영역에 속하는지 여부로 판단
            world_x = self.min_x + (xx_idx + 0.5) * self.grid_size # Shape: (idx_ymax-idx_ymin, idx_xmax-idx_xmin)
            world_y = self.min_y + (yy_idx + 0.5) * self.grid_size # Shape: (idx_ymax-idx_ymin, idx_xmax-idx_xmin)

            # 4. Local coordinate으로 변환 후 영역에 속하는지 판단
            pos = self.data.geom_xpos[geom_idx]
            rot = self.data.geom_xmat[geom_idx].reshape(3, 3)
            
            final_obs_mask = np.zeros_like(world_x, dtype=bool) # Shape: (idx_ymax-idx_ymin, idx_xmax-idx_xmin)
            
            for check_z in rb_zsamples:
                if check_z < zmin - eps or check_z > zmax + eps:
                    continue
                
                world_z = np.full_like(world_x, check_z)
                
                # 장애물 점유 여부를 검사할 좌표 목록 (Local coordinate)
                rel_p = np.stack([world_x.ravel() - pos[0], 
                                world_y.ravel() - pos[1], 
                                world_z.ravel() - pos[2]], axis=-1) # (N, 3)
                local_p = rel_p @ rot  # rot.T @ rel_p.T 의 벡터화 버전

                lx = local_p[:, 0].reshape(world_x.shape)
                ly = local_p[:, 1].reshape(world_x.shape)
                lz = local_p[:, 2].reshape(world_x.shape)

                # 5. 각 local 좌표가 장애물 영역에 속하는지 검사
                size = self.model.geom_size[geom_idx]
                mask = np.zeros_like(world_x, dtype=bool)
                
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    mask = (np.abs(lx) <= size[0]+eps) & (np.abs(ly) <= size[1]+eps) & (np.abs(lz) <= size[2]+eps)

                elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                    mask = (lx**2 + ly**2 + lz**2) <= (size[0]+eps)**2

                elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    mask = (lx**2 + ly**2 <= (size[0]+eps)**2) & (np.abs(lz) <= size[1]+eps)

                elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    mask_body = (lx**2 + ly**2 <= (size[0]+eps)**2) & (np.abs(lz) <= size[1]+eps)
                    mask_cap = (lx**2 + ly**2 + (np.abs(lz) - size[1])**2) <= (size[0]+eps)**2
                    mask = mask_body | mask_cap

                elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                    
                    # mesh 데이터 추출
                    mesh_id = self.model.geom_dataid[geom_idx]
                    start_v = self.model.mesh_vertadr[mesh_id] # 꼭짓점
                    num_v = self.model.mesh_vertnum[mesh_id]
                    start_f = self.model.mesh_faceadr[mesh_id] # 삼각형 면
                    num_f = self.model.mesh_facenum[mesh_id]
                    
                    vertices = self.model.mesh_vert[start_v : start_v + num_v]
                    faces = self.model.mesh_face[start_f : start_f + num_f]
                    
                    # Trimesh 객체 생성
                    m = trimesh.Trimesh(vertices=vertices, faces=faces)
                    
                    # 검사하고하는 local coordinate 좌표
                    query_points = np.stack([lx.ravel(), ly.ravel(), lz.ravel()], axis=-1)
                    
                    # 검사하고자하는 점들이 mesh 내부에 있는지 판정
                    inside_mask = m.contains(query_points)
                    mask = inside_mask.reshape(lx.shape)
                    
                final_obs_mask |= mask

            # 5. Grid map에 적용
            self.obs_grid_map[idx_ymin:idx_ymax+1, idx_xmin:idx_xmax+1] |= final_obs_mask.astype(np.uint8)

if __name__ == "__main__":
    
    # args parsing
    args = parse_args()
    xml_path = os.path.join(args.env_folder, args.map_name)
    
    converter = MujocoGridConverter(
        xml_path=xml_path,
        grid_size=args.grid_size,
        robot_height=args.robot_h,
        obstacle_geom_group=args.obs_geom_gp,
    )
    converter.convert_mjc2grid()
    
    import matplotlib.pyplot as plt
    plt.imshow(converter.obs_grid_map, cmap='binary', origin='lower')
    plt.colorbar()
    plt.show()