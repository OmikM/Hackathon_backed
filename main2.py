import math
import time
import os
import requests
from typing import List, Tuple, Dict, Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from playground import DRONE, build_lattice, _bucket, CACHE_SPATIAL_DEG, CACHE_PATH, _STATS, \
fetch_wind_and_atmos_batch, fetch_elevations, straight_route_energy, \
solve_min_energy_route, NO_CACHE

app = FastAPI(
    title="High-Performance Wind-Aware Routing API",
    version="2.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:5173", "http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Ensures OPTIONS, GET, POST, PUT are all allowed
    allow_headers=["*"],
)

# --- CONFIGURATION DATA MODELS ---

class DroneConfig(BaseModel):
    rho: float = Field(default=1.225)
    c_d: float = Field(default=0.30)
    area: float = Field(default=0.35)
    mass: float = Field(default=6.0)
    gravity: float = Field(default=9.81)
    max_ground_speed: float = Field(default=18.0)
    hover_power_factor: float = Field(default=0.08)

current_config = DroneConfig()

class Point2D(BaseModel):
    x: float
    y: float

class WindGridRequest(BaseModel):
    center_lat: float
    center_lon: float
    grid_steps: int = 1
    step_deg: float = 0.05

class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    start_point: Point2D
    goal_point: Point2D
    mode: Literal["energy", "speed"] = "energy"
    step_size: float = 80.0

class RouteResponse(BaseModel):
    optimization_mode: str
    active_drone_mass_kg: float
    max_ground_speed_ms: float
    waypoints: List[Point2D]
    total_energy_kj: float
    total_energy_wh: float
    total_time_seconds: float
    formatted_time: str


def local_point_to_lat_lon(origin_lat: float, origin_lon: float, point: Point2D) -> Tuple[float, float]:
    latitude = origin_lat + point.y / 111000.0
    longitude = origin_lon + point.x / (111000.0 * math.cos(math.radians(origin_lat)))
    return latitude, longitude


def lat_lon_to_local_point(origin_lat: float, origin_lon: float, latitude: float, longitude: float) -> Point2D:
    x = (longitude - origin_lon) * 111000.0 * math.cos(math.radians(origin_lat))
    y = (latitude - origin_lat) * 111000.0
    return Point2D(x=x, y=y)


# --- METEOROLOGICAL FETCHERS ---

def fetch_wind_field_cached(center_lat: float, center_lon: float, grid_steps: int = 1, step_deg: float = 0.05):
    """Fetches a square grid (used by the generic grid endpoint)."""
    wind_grid = {}
    lats = [center_lat + (i * step_deg) for i in range(-grid_steps, grid_steps + 1)]
    lons = [center_lon + (j * step_deg) for j in range(-grid_steps, grid_steps + 1)]

    for lat in lats:
        for lon in lons:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}&current_weather=true"
            try:
                res = requests.get(url, timeout=2).json()
                curr = res.get("current_weather", {})
                speed_ms = curr.get("windspeed", 0) / 3.6
                rad = math.radians(270 - curr.get("winddirection", 0))
                
                x_m = round((lon - center_lon) * 111000 * math.cos(math.radians(center_lat)), -2)
                y_m = round((lat - center_lat) * 111000, -2)
                wind_grid[(x_m, y_m)] = (speed_ms * math.cos(rad), speed_ms * math.sin(rad))
            except Exception:
                wind_grid[(0.0, 0.0)] = (0.0, 0.0)
    return wind_grid

def fetch_wind_along_path(origin_lat: float, origin_lon: float, start_xy: Tuple[float, float], goal_xy: Tuple[float, float], num_samples: int = 8) -> Dict:
    """Uses Open-Meteo Bulk API to instantly fetch weather exactly along the flight corridor."""
    dx, dy = goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]
    
    lats, lons, points_xy = [], [], []
    
    for i in range(num_samples):
        t = i / (num_samples - 1)
        x = start_xy[0] + t * dx
        y = start_xy[1] + t * dy
        points_xy.append((x, y))
        
        # Convert local XY meters back to GPS coordinates
        lat = origin_lat + (y / 111000.0)
        lon = origin_lon + (x / (111000.0 * math.cos(math.radians(origin_lat))))
        lats.append(f"{lat:.4f}")
        lons.append(f"{lon:.4f}")
        
    url = f"https://api.open-meteo.com/v1/forecast?latitude={','.join(lats)}&longitude={','.join(lons)}&current_weather=true"
    wind_data = {}
    
    try:
        res = requests.get(url, timeout=5).json()
        if isinstance(res, list): # Bulk response
            for i, r in enumerate(res):
                curr = r.get("current_weather", {})
                speed = curr.get("windspeed", 0) / 3.6
                rad = math.radians(270 - curr.get("winddirection", 0))
                wind_data[points_xy[i]] = (speed * math.cos(rad), speed * math.sin(rad))
        else: # Fallback if list fails
            wind_data = {pt: (0.0, 0.0) for pt in points_xy}
    except Exception:
        wind_data = {pt: (0.0, 0.0) for pt in points_xy}
        
    return wind_data

def get_wind_at(x: float, y: float, wind_grid: Dict) -> Tuple[float, float]:
    if not wind_grid:
        return 0.0, 0.0
    closest_key = min(wind_grid.keys(), key=lambda pt: (pt[0] - x)**2 + (pt[1] - y)**2)
    return wind_grid[closest_key]


# --- PHYSICS & PATH GENERATION ---

def evaluate_segment(p1: Tuple[float, float], p2: Tuple[float, float], wind_grid: Dict, mode: str, cfg: DroneConfig):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist == 0: return 0.0, 0.0

    mid_x, mid_y = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    u_wind, v_wind = get_wind_at(mid_x, mid_y, wind_grid)
    p_hover = math.sqrt(((cfg.mass * cfg.gravity) ** 3) / (2 * cfg.rho * cfg.area)) * cfg.hover_power_factor

    if mode == "speed":
        v_air_target = cfg.max_ground_speed
        dir_x, dir_y = dx / dist, dy / dist
        wind_proj = u_wind * dir_x + v_wind * dir_y
        
        v_ground = wind_proj + math.sqrt(max(0.1, v_air_target**2 - (u_wind * (-dir_y) + v_wind * dir_x)**2))
        v_ground = max(1.0, v_ground) 
        
        t_sec = dist / v_ground
        f_drag = 0.5 * cfg.rho * cfg.c_d * cfg.area * (v_air_target ** 2)
        return (f_drag * v_air_target + p_hover) * t_sec, t_sec

    else:
        best_energy, best_time = float('inf'), 0.0
        for v_g in [6.0, 10.0, 14.0, cfg.max_ground_speed]:
            if v_g > cfg.max_ground_speed: continue
            v_air = math.hypot(v_g * (dx / dist) - u_wind, v_g * (dy / dist) - v_wind)
            f_drag = 0.5 * cfg.rho * cfg.c_d * cfg.area * (v_air ** 2)
            t_sec = dist / v_g
            e_joules = (f_drag * v_air + p_hover) * t_sec
            if e_joules < best_energy:
                best_energy, best_time = e_joules, t_sec
        return best_energy, best_time

def generate_mode_aware_path(start: Tuple[float, float], goal: Tuple[float, float], mode: str, wind_grid: Dict, cfg: DroneConfig, num_points: int = 50) -> List[Tuple[float, float]]:
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    dist = math.hypot(dx, dy)
    
    if dist == 0: return [start, goal]

    nx, ny = -dy / dist, dx / dist
    
    # 1. SPEC DYNAMICS
    # Higher drag-to-mass ratio forces wider energy-saving detours
    drag_sensitivity = (cfg.c_d * cfg.area) / max(0.5, cfg.mass)
    
    # Slower drones get drifted more by crosswinds, requiring larger path adjustments
    speed_factor = 18.0 / max(5.0, cfg.max_ground_speed)
    
    num_segments = max(3, min(10, int(dist / 25000))) 
    waypoints = [start]
    
    for i in range(1, num_segments):
        t = i / num_segments
        base_x, base_y = start[0] + t * dx, start[1] + t * dy
        
        u_wind, v_wind = get_wind_at(base_x, base_y, wind_grid)
        crosswind_push = u_wind * nx + v_wind * ny
        
        if mode == "speed":
            # Speed mode curve flexes based on ground speed limits vs wind push
            offset = (dist * 0.01) + (crosswind_push * dist * 0.003 * speed_factor)
        else:
            # Energy mode scales arc width based on drone drag/mass characteristics
            base_curve = 0.03 * (1.0 + drag_sensitivity * 12.0)
            wind_curve = 0.01 * speed_factor
            offset = (dist * base_curve) + (crosswind_push * dist * wind_curve)
            
        waypoints.append((base_x + nx * offset, base_y + ny * offset))
        
    waypoints.append(goal)
    
    # Catmull-Rom Smoothing
    extended_points = [waypoints[0]] + waypoints + [waypoints[-1]]
    smoothed_path = []
    pts_per_seg = max(2, num_points // num_segments)

    for i in range(len(extended_points) - 3):
        p0, p1, p2, p3 = extended_points[i], extended_points[i+1], extended_points[i+2], extended_points[i+3]
        for j in range(pts_per_seg):
            t = j / pts_per_seg
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            smoothed_path.append((round(x, 1), round(y, 1)))

    smoothed_path.append(goal)
    return smoothed_path


# --- API ENDPOINTS ---

@app.get("/api/v1/config", response_model=DroneConfig)
def get_drone_config(): return current_config

@app.put("/api/v1/config", response_model=DroneConfig)
def update_drone_config(new_config: DroneConfig):
    global current_config
    current_config = new_config
    return current_config

@app.post("/api/v1/wind-grid")
def get_wind_grid_endpoint(payload: WindGridRequest):
    grid = fetch_wind_field_cached(payload.center_lat, payload.center_lon, payload.grid_steps, payload.step_deg)
    formatted_grid = [
        {"x_offset_m": key[0], "y_offset_m": key[1], "u_wind_ms": val[0], "v_wind_ms": val[1]}
        for key, val in grid.items()
    ]
    return {"center_lat": payload.center_lat, "center_lon": payload.center_lon, "grid_points": formatted_grid}

@app.post("/api/v1/plan-route", response_model=RouteResponse)
def plan_route_endpoint(payload: RouteRequest):
    point_a = local_point_to_lat_lon(payload.origin_lat, payload.origin_lon, payload.start_point)
    point_b = local_point_to_lat_lon(payload.origin_lat, payload.origin_lon, payload.goal_point)

    layers, dist_km, brg = build_lattice(point_a, point_b)
    all_nodes = [n for row in layers for n in row]
    unique_buckets = len(set(_bucket(*n) for n in all_nodes))

    print(f"Drone: {DRONE.name}  |  mass {DRONE.mass_kg} kg  |  "
            f"usable battery {DRONE.usable_energy_wh:.0f} Wh")
    print(f"Route: {point_a} -> {point_b}  ({dist_km:.1f} km, bearing {brg:.0f} deg)")
    print(f"{len(all_nodes)} lattice points -> {unique_buckets} unique "
            f"~{CACHE_SPATIAL_DEG*111:.1f}km buckets after spatial de-duping "
            f"(cache file: {os.path.basename(CACHE_PATH)}{' -- disabled via --no-cache' if NO_CACHE else ''})")

    atmos_lookup = fetch_wind_and_atmos_batch(all_nodes)
    elevations = fetch_elevations(all_nodes)

    print(f"Wind buckets: {_STATS['wind_hits']} from cache, "
            f"{_STATS['wind_fetched']} fetched fresh")
    print(f"Elevation buckets: {_STATS['elev_hits']} from cache, "
            f"{_STATS['elev_fetched']} fetched fresh")
    print(f"Terrain elevation range along corridor: "
            f"{min(elevations):.0f} - {max(elevations):.0f} m ASL "
            f"(Copernicus GLO-90 DEM)")

    straight_e, straight_t = straight_route_energy(layers, atmos_lookup, DRONE)
    opt_e, opt_t, opt_path = solve_min_energy_route(layers, atmos_lookup, DRONE)

    print("\n--- Straight-line route ---")
    print(f"Energy: {straight_e:.1f} Wh   Time: {straight_t/60:.1f} min")

    print("\n--- Wind-optimized route ---")
    print(f"Energy: {opt_e:.1f} Wh   Time: {opt_t/60:.1f} min")
    print(f"Waypoints: {[(round(la,4), round(lo,4)) for la, lo in opt_path]}")

    saved_pct = 100 * (straight_e - opt_e) / straight_e if straight_e else 0
    print(f"\nEnergy saved by routing with the wind: {saved_pct:.1f}%")
    print(f"Usable battery budget: {DRONE.usable_energy_wh:.0f} Wh  "
            f"(straight route uses {100*straight_e/DRONE.usable_energy_wh:.1f}% of it, "
            f"optimized uses {100*opt_e/DRONE.usable_energy_wh:.1f}%)")
    
    # wind_corridor = fetch_wind_along_path(payload.origin_lat, payload.origin_lon, start, goal, num_samples=10)
    
    # # Passed current_config so path shape reacts to mass, drag, and speed settings
    # path_tuples = generate_mode_aware_path(start, goal, payload.mode, wind_corridor, current_config, num_points=60)
    
    # total_time, total_energy = 0.0, 0.0
    # for i in range(len(path_tuples) - 1):
    #     e, t = evaluate_segment(path_tuples[i], path_tuples[i+1], wind_corridor, payload.mode, current_config)
    #     total_energy += e
    #     total_time += t
        
    # mins, secs = int(total_time // 60), int(total_time % 60)
    
    return RouteResponse(
        optimization_mode=payload.mode,
        active_drone_mass_kg=current_config.mass,
        max_ground_speed_ms=current_config.max_ground_speed,
        waypoints=[
            lat_lon_to_local_point(payload.origin_lat, payload.origin_lon, p[0], p[1])
            for p in opt_path
        ],
        total_energy_kj=round(opt_e * 3.6, 2),
        total_energy_wh=round(opt_e, 2),
        total_time_seconds=round(opt_t, 1),
        formatted_time=f"{int(opt_t/60)}m {int(opt_t%60)}s"
    )