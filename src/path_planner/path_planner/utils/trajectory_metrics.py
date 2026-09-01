"""
Input: waypoints (list / float / origin: bottom-left)
Output: Cleaning Time (minutes) or Overlap (%)
"""
from __future__ import annotations

import math
from typing import Sequence

# Map 해상도 (grid 한변 길이; m)
GRID_RESOLUTION_M = 0.025
# 로봇청소기 지름 (m)
ROBOT_DIAMETER_M = 0.36

# 선형 속도/가속도 (m/s, m/s^2)
LINEAR_VELOCITY = 0.4
LINEAR_ACCELERATION = 4.0

# 각속도/각가속도 (rad/s, rad/s^2)
ANGULAR_VELOCITY = 3.0
ANGULAR_ACCELERATION = 10.0
# 회전이 일어났다고 판단하는 기준 (30도)
TURN_THRESHOLD = math.radians(30.0)

# INPUT
Waypoint = tuple[int, int]

# ============================================================
# Main metrics
# ============================================================

# OUTPUT 1: Cleaning Time (minutes)
def cleaning_time_minutes(
    waypoints_m: Sequence[Waypoint],
) -> float:
    """청소 시간을 분 단위로 계산"""
    return float(_cleaning_time_seconds(waypoints_m) / 60.0)

# OUTPUT 2: Overlap (%)
def overlap_percent(waypoints_m: Sequence[Waypoint]) -> float:
    """Overlap (%) 계산"""
    stats = _overlap_stats(waypoints_m)
    return float(stats['overlap_percent'])

# ============================================================
# Utils (Cleaning Time)
# ============================================================

def _cleaning_time_seconds(
    waypoints_m: Sequence[Waypoint],
) -> float:
    """청소 시간을 초 단위로 계산"""
    if waypoints_m is None or len(waypoints_m) < 2:
        return 0.0

    # 연속 waypoint 사이의 이동 벡터와 거리
    steps: list[tuple[float, float, float]] = []
    for (x0, y0), (x1, y1) in zip(waypoints_m[:-1], waypoints_m[1:]):
        x0, x1, y0, y1 = round(x0), round(x1), round(y0), round(y1)
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        dist_m = math.hypot(dx, dy)
        if dist_m > 1e-12:
            steps.append((dx, dy, dist_m))

    if not steps:
        return 0.0

    # 각 이동 구간의 heading(로봇청소기가 바라보고 있는 방향)을 radian으로 계산
    turn_threshold_rad = max(0.0, float(TURN_THRESHOLD))
    headings = [math.atan2(dy, dx) for dx, dy, _ in steps]

    # threshold보다 큰 방향 변화가 있으면 새 segment로 나눔
    segments: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(steps)):
        delta = _wrap_angle_rad(headings[i] - headings[i - 1])
        if abs(delta) > turn_threshold_rad:
            segments.append((start, i - 1))
            start = i
    segments.append((start, len(steps) - 1))

    # segment별 직진 거리와 대표 heading
    seg_dists = [sum(steps[j][2] for j in range(a, b + 1)) for a, b in segments]
    seg_dirs = [headings[a] for a, _ in segments]

    # 각 segment는 직진 후 다음 segment 방향으로 회전
    total_time = 0.0
    for i, dist_m in enumerate(seg_dists):
        theta_rad = 0.0
        if i < len(seg_dirs) - 1:
            theta_rad = abs(_wrap_angle_rad(seg_dirs[i + 1] - seg_dirs[i]))
            if theta_rad <= turn_threshold_rad:
                theta_rad = 0.0
        total_time += compute_cleaning_time_of_segment(
            dist_m,
            theta_rad,
            linear_velocity=LINEAR_VELOCITY,
            angular_velocity=ANGULAR_VELOCITY,
            linear_acceleration=LINEAR_ACCELERATION,
            angular_acceleration=ANGULAR_ACCELERATION,
        )
    return float(total_time)


def compute_cleaning_time_of_segment(
    distance_m: float,
    theta_rad: float,
    *,
    linear_velocity: float,
    angular_velocity: float,
    linear_acceleration: float,
    angular_acceleration: float,
) -> float:
    """직진/회전 한 segment의 이동 시간"""
    d = abs(float(distance_m))
    theta = abs(float(theta_rad))
    if d <= 1e-12 and theta <= 1e-12:
        return 0.0

    # 직진하면서 회전하는 경우 (직진/회전 최대 속도)
    speed_inv_sq = 0.0
    if d > 1e-12:
        speed_inv_sq += (d / max(float(linear_velocity), 1e-12)) ** 2
    if theta > 1e-12:
        speed_inv_sq += (theta / max(float(angular_velocity), 1e-12)) ** 2
    speed_max = 1.0 / math.sqrt(max(speed_inv_sq, 1e-12))

    # 직진하면서 회전하는 경우 (직진/회전 최대 가속도)
    accel_inv_sq = 0.0
    if d > 1e-12:
        accel_inv_sq += (d / max(float(linear_acceleration), 1e-12)) ** 2
    if theta > 1e-12:
        accel_inv_sq += (theta / max(float(angular_acceleration), 1e-12)) ** 2
    accel_max = 1.0 / math.sqrt(max(accel_inv_sq, 1e-12))

    # 최대 속도에 도달하기까지 필요한 시간과 진행량
    t_acc = speed_max / max(accel_max, 1e-12)
    s_acc = 0.5 * accel_max * t_acc * t_acc

    # (짧은 구간) 최대 속도에 도달하지 않는 삼각형 프로파일
    if 2.0 * s_acc >= 1.0:
        return 2.0 * math.sqrt(1.0 / max(accel_max, 1e-12))

    # (긴 구간) 가속-등속-감속의 사다리꼴 프로파일
    return 2.0 * t_acc + (1.0 - 2.0 * s_acc) / max(speed_max, 1e-12)

# ============================================================
# Utils (Overlap)
# ============================================================

def _overlap_stats(
    waypoints_m: Sequence[Waypoint],
    *,
    resolution_m: float = GRID_RESOLUTION_M,
    robot_diameter_m: float = ROBOT_DIAMETER_M,
) -> dict:
    """Overlap 계산에 필요한 swept 영역 통계"""
    swept_once: set[tuple[int, int]] = set()
    previous_footprint: set[tuple[int, int]] | None = None
    total_overlap = 0

    if waypoints_m is None or len(waypoints_m) == 0:
        return {
            'overlap_ratio': 0.0,
            'overlap_percent': 0.0,
            'unique_swept_grids': 0,
            'total_overlap_grids': 0,
        }
    
    # footprint가 한 step 이동하며 새로 드러난 grid만 visit 대상으로 봄
    segments = [(waypoints_m[0], waypoints_m[0])] if len(waypoints_m) == 1 else list(zip(waypoints_m[:-1], waypoints_m[1:]))
    for p0, p1 in segments:
        for footprint in _segment_footprints(
            p0,
            p1,
            resolution_m=resolution_m,
            robot_diameter_m=robot_diameter_m,
        ):
            if previous_footprint is None:
                newly_visited = footprint
            else:
                newly_visited = footprint - previous_footprint

            total_overlap += len(newly_visited & swept_once)
            swept_once.update(newly_visited)
            previous_footprint = footprint

    # overlap = (중복해서 방문한 영역) / (coverage 영역)
    unique_swept = len(swept_once)
    overlap_ratio_value = float(total_overlap) / float(max(1, unique_swept))
    return {
        'overlap_ratio': overlap_ratio_value,
        'overlap_percent': overlap_ratio_value * 100.0,
        'unique_swept_grids': int(unique_swept),
        'total_overlap_grids': int(total_overlap),
    }

# ============================================================
# Utils (Others)
# ============================================================

def _wrap_angle_rad(angle: float) -> float:
    # 각도 차이를 [-pi, pi] 범위로 정규화
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


def _metric_cell_from_point(
    x_m: float,
    y_m: float,
    *,
    resolution_m: float,
) -> tuple[int, int]:
    # 실수 좌표가 속한 metric cell을 찾음
    gx = int(float(x_m) / float(resolution_m))
    gy = int(float(y_m) / float(resolution_m))
    return gx, gy


def _footprint_cells(
    x_m: float,
    y_m: float,
    *,
    resolution_m: float,
    robot_diameter_m: float,
) -> set[tuple[int, int]]:
    """각 waypoint에서 청소되는 grid 집합"""
    radius_m = float(robot_diameter_m) * 0.5

    # waypoint 중심이 속한 metric cell을 찾음
    center = _metric_cell_from_point(
        x_m,
        y_m,
        resolution_m=resolution_m,
    )

    # 청소되는 grid 집합 검사
    r_cells = int(math.ceil(radius_m / float(resolution_m)))
    cx, cy = center
    cells: set[tuple[int, int]] = set()
    for gy in range(cy - r_cells, cy + r_cells + 1):
        yy = (float(gy) + 0.5) * float(resolution_m)
        for gx in range(cx - r_cells, cx + r_cells + 1):
            xx = (float(gx) + 0.5) * float(resolution_m)
            if (xx - float(x_m)) ** 2 + (yy - float(y_m)) ** 2 <= radius_m ** 2:
                cells.add((gx, gy))
    return cells


def _segment_footprint_cells(
    p0: Waypoint,
    p1: Waypoint,
    *,
    resolution_m: float,
    robot_diameter_m: float,
) -> set[tuple[int, int]]:
    """로봇청소기가 청소한 segment의 전체 grid 집합"""
    cells: set[tuple[int, int]] = set()
    for footprint in _segment_footprints(
        p0,
        p1,
        resolution_m=resolution_m,
        robot_diameter_m=robot_diameter_m,
    ):
        cells.update(footprint)
    return cells


def _segment_footprints(
    p0: Waypoint,
    p1: Waypoint,
    *,
    resolution_m: float,
    robot_diameter_m: float,
):
    # 로봇청소기가 청소한 segment들 기록
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dist = math.hypot(x1 - x0, y1 - y0)

    sample_step = max(float(resolution_m) * 0.5, 1e-9)
    n = max(1, int(math.ceil(dist / sample_step)))
    for i in range(n + 1):
        t = float(i) / float(n)
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        yield _footprint_cells(
            x,
            y,
            resolution_m=resolution_m,
            robot_diameter_m=robot_diameter_m,
        )
