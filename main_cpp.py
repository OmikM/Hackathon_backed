import os
import sys
import json
import time
import math
import subprocess
from typing import List, Tuple, Dict, Literal, Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from playground import DRONE, propulsion_power_w as playground_propulsion_power_w

app = FastAPI(
    title="C++ Wind-Aware Routing API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CACHING & RATE-LIMIT CONFIG ---
CACHE_PATH = ".cpp_route_cache.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALGORITHM_PATH = os.path.join(BASE_DIR, "algorithm_cpp.exe" if os.name == "nt" else "algorithm_cpp")
CACHE_SPATIAL_DEG = 0.05  # Snap grid queries to ~5.5km resolution
WIND_CACHE_TTL_S = 1800   # 30-minute cache lifetime
MIN_REQUEST_INTERVAL = 1.1 # Enforce 1.1s minimum delay between API calls to prevent 429s

LAST_API_CALL_TIME = 0.0

# --- DATA MODELS ---
class Point2D(BaseModel):
    x: float
    y: float

class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    start_point: Point2D
    goal_point: Point2D
    mode: Literal["energy", "speed"] = "speed"
    step_size: float = 80.0  # Cell size in meters
    grid_rows: int = 50
    grid_cols: int = 50

class RouteResponse(BaseModel):
    optimization_mode: str
    waypoints: List[Point2D]
    total_energy_kj: float
    total_energy_wh: float
    total_time_seconds: float
    formatted_time: str
    # Reference baseline: the direct (Bresenham) line from start to goal,
    # flown through the exact same wind field and priced with the same
    # physics as the optimized route. None when that direct bearing isn't
    # flyable (crosswind exceeds the drone's max airspeed somewhere on it).
    straight_line_time_seconds: Optional[float] = None
    straight_line_energy_wh: Optional[float] = None
    pct_time_saved_vs_straight_line: Optional[float] = None
    pct_energy_saved_vs_straight_line: Optional[float] = None


# --- UTILITIES ---
def local_point_to_lat_lon(origin_lat: float, origin_lon: float, point: Point2D) -> Tuple[float, float]:
    latitude = origin_lat + point.y / 111000.0
    longitude = origin_lon + point.x / (111000.0 * math.cos(math.radians(origin_lat)))
    return latitude, longitude

def lat_lon_to_local_point(origin_lat: float, origin_lon: float, latitude: float, longitude: float) -> Point2D:
    x = (longitude - origin_lon) * 111000.0 * math.cos(math.radians(origin_lat))
    y = (latitude - origin_lat) * 111000.0
    return Point2D(x=x, y=y)

def grid_to_lat_lon(origin_lat: float, origin_lon: float, r: int, c: int, step_m: float,
                     grid_origin_x: float = 0.0, grid_origin_y: float = 0.0) -> Tuple[float, float]:
    y_m = grid_origin_y + r * step_m
    x_m = grid_origin_x + c * step_m
    return local_point_to_lat_lon(origin_lat, origin_lon, Point2D(x=x_m, y=y_m))


def local_point_to_grid(point: Point2D, step_m: float) -> Tuple[int, int]:
    return round(point.y / step_m), round(point.x / step_m)

AIR_DENSITY_KGM3 = 1.225
V_AIR_MPS = DRONE.max_speed_mps
AVIONICS_POWER_W = DRONE.avionics_payload_power_w

def propulsion_power_w(airspeed_mps: float) -> float:
    return playground_propulsion_power_w(airspeed_mps, AIR_DENSITY_KGM3, DRONE)

def estimate_route_energy_wh(path: List[List[int]], wind_data: Dict[Tuple[float, float], Tuple[float, float]],
                             grid_coords: List[Tuple[float, float]], columns: int,
                             step_m: float) -> float:
    total_energy_wh = 0.0
    total_power_w = propulsion_power_w(V_AIR_MPS) + AVIONICS_POWER_W

    for (row_a, col_a), (row_b, col_b) in zip(path, path[1:]):
        direction_length = math.hypot(col_b - col_a, row_b - row_a)
        distance_m = step_m * direction_length
        if direction_length == 0:
            continue

        destination_coord = grid_coords[row_b * columns + col_b]
        wind_e, wind_n = wind_data.get(destination_coord, (0.0, 0.0))
        unit_east = (col_b - col_a) / direction_length
        unit_north = (row_a - row_b) / direction_length
        wind_parallel = unit_east * wind_e + unit_north * wind_n
        wind_squared = wind_e ** 2 + wind_n ** 2
        crosswind_squared = max(0.0, wind_squared - wind_parallel ** 2)
        discriminant = V_AIR_MPS ** 2 - crosswind_squared

        if discriminant < 0:
            continue

        ground_speed_mps = wind_parallel + math.sqrt(discriminant)
        if ground_speed_mps > 0:
            total_energy_wh += total_power_w * distance_m / ground_speed_mps / 3600.0

    return total_energy_wh

# --- DISK CACHE MANAGEMENT ---
def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache(cache: dict):
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass

# --- RATE-LIMITED OPEN-METEO DATA FETCHING ---
def fetch_wind_grid_batch(coords: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Tuple[float, float]]:
    """Fetches wind speed/direction in a SINGLE batched request with persistent caching and 429 protection."""
    global LAST_API_CALL_TIME
    cache = load_cache()
    now = time.time()
    
    # 1. Deduplicate & check cache
    results = {}
    missing_coords = []
    
    for lat, lon in coords:
        bucket_key = f"{round(lat / CACHE_SPATIAL_DEG) * CACHE_SPATIAL_DEG:.3f},{round(lon / CACHE_SPATIAL_DEG) * CACHE_SPATIAL_DEG:.3f}"
        if bucket_key in cache and (now - cache[bucket_key]["timestamp"] < WIND_CACHE_TTL_S):
            results[(lat, lon)] = (cache[bucket_key]["u"], cache[bucket_key]["v"])
        else:
            missing_coords.append((lat, lon, bucket_key))

    if not missing_coords:
        return results

    # 2. Batch request missing items in chunks to avoid URL length issues & rate limits
    chunk_size = 50
    unique_missing = list({(b_key): (la, lo) for la, lo, b_key in missing_coords}.items())
    
    for i in range(0, len(unique_missing), chunk_size):
        chunk = unique_missing[i:i + chunk_size]
        lats = ",".join(f"{la:.4f}" for b_key, (la, lo) in chunk)
        lons = ",".join(f"{lo:.4f}" for b_key, (la, lo) in chunk)
        
        # Rate-limiting pause
        elapsed = time.time() - LAST_API_CALL_TIME
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        params = {
            "latitude": lats,
            "longitude": lons,
            "current_weather": "true",
            "windspeed_unit": "ms"
        }
        url = f"https://api.open-meteo.com/v1/forecast?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "DroneRoutingCpp/1.0"})
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                with urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode())
                    LAST_API_CALL_TIME = time.time()
                    entries = data if isinstance(data, list) else [data]

                    for (b_key, (la, lo)), entry in zip(chunk, entries):
                        cw = entry.get("current_weather", {})
                        ws = cw.get("windspeed", 0.0)
                        wdir = cw.get("winddirection", 0.0)

                        # Meteorological to Cartesian U (East-West) & V (North-South)
                        rad = math.radians((270 - wdir) % 360)
                        u = ws * math.cos(rad)
                        v = ws * math.sin(rad)

                        cache[b_key] = {"u": u, "v": v, "timestamp": now}
                        results[(la, lo)] = (u, v)
                break

            except HTTPError as e:
                if e.code == 429 and attempt < max_retries:
                    print(f"[wind-fetch] HTTP 429; retrying in 5s ({attempt + 1}/{max_retries})...")
                    time.sleep(5)
                    continue
                print(f"[wind-fetch] Open-Meteo HTTP {e.code}; using zero wind for this chunk.")
                break

            except URLError as e:
                if attempt < max_retries:
                    backoff = 2.0 * (attempt + 1)
                    print(f"[wind-fetch] Network error ({e.reason}); retrying in "
                          f"{backoff:.0f}s ({attempt + 1}/{max_retries})...")
                    time.sleep(backoff)
                    continue
                print(f"[wind-fetch] Open-Meteo unavailable ({e.reason}); using zero wind for this chunk.")
                break
                
    save_cache(cache)
    
    # Fill remaining from cache
    for la, lo, b_key in missing_coords:
        if (la, lo) not in results and b_key in cache:
            results[(la, lo)] = (cache[b_key]["u"], cache[b_key]["v"])
        elif (la, lo) not in results:
            results[(la, lo)] = (0.0, 0.0)  # Fallback zero wind
            
    return results

# --- MAIN ROUTING ENDPOINT ---
@app.post("/api/v1/plan-route", response_model=RouteResponse)
def plan_route_endpoint(payload: RouteRequest):
    if not os.path.isfile(ALGORITHM_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Routing binary not found: {ALGORITHM_PATH}. Compile algorithm.cpp first.",
        )

    if payload.step_size <= 0:
        raise HTTPException(status_code=422, detail="step_size must be greater than zero.")
    grid_origin_x = min(0.0, payload.start_point.x, payload.goal_point.x)
    grid_origin_y = min(0.0, payload.start_point.y, payload.goal_point.y)
    start_r = round((payload.start_point.y - grid_origin_y) / payload.step_size)
    start_c = round((payload.start_point.x - grid_origin_x) / payload.step_size)
    goal_r = round((payload.goal_point.y - grid_origin_y) / payload.step_size)
    goal_c = round((payload.goal_point.x - grid_origin_x) / payload.step_size)

    # Expand the grid when necessary so the requested goal is not silently
    # clamped to a different cell and returned as the wrong waypoint.
    R = max(payload.grid_rows, start_r + 1, goal_r + 1)
    C = max(payload.grid_cols, start_c + 1, goal_c + 1)
    
    # Convert local start/goal to grid indices
    # Build coordinates for Open-Meteo
    grid_coords = [
        grid_to_lat_lon(
            payload.origin_lat,
            payload.origin_lon,
            r,
            c,
            payload.step_size,
            grid_origin_x,
            grid_origin_y,
        )
        for r in range(R) for c in range(C)
    ]

    # Fetch batch wind field
    wind_data = fetch_wind_grid_batch(grid_coords)

    # Format stdin matrix string for C++ process
    stdin_data = []
    idx = 0
    for r in range(R):
        row_str = []
        for c in range(C):
            coord = grid_coords[idx]
            u, v = wind_data.get(coord, (0.0, 0.0))
            row_str.append(f"{u:.2f} {v:.2f}")
            idx += 1
        stdin_data.append(" ".join(row_str))
    
    input_payload = "\n".join(stdin_data)

    # Execute C++ process
    cmd = [ALGORITHM_PATH, str(R), str(C), str(start_r), str(start_c), str(goal_r), str(goal_c), str(payload.step_size)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(input=input_payload)

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"C++ execution failed (exit code {proc.returncode}): {stderr.strip()}",
        )

    cpp_out = json.loads(stdout)
    cpp_path = cpp_out.get("path", [])
    if not cpp_path:
        raise HTTPException(status_code=422, detail="C++ solver could not find a flyable route.")
    
    # Convert path grid cells back to local meter offsets, matching the
    # waypoint contract used by playground.py and the frontend.
    waypoints = [
        Point2D(
            x=grid_origin_x + c * payload.step_size,
            y=grid_origin_y + r * payload.step_size,
        )
        for r, c in cpp_path
    ]
    waypoints[0] = payload.start_point
    waypoints[-1] = payload.goal_point

    tot_time = cpp_out["total_time"]
    total_energy_wh = estimate_route_energy_wh(
        cpp_path,
        wind_data,
        grid_coords,
        C,
        payload.step_size,
    )

    # --- Straight-line reference ---
    # How would a direct bee-line from start to goal fare, flown through
    # the exact same wind field? This is the baseline every wind-aware
    # route here is meant to beat -- same idea as playground.py's
    # straight_route_energy(), just on a discrete grid. The C++ solver
    # already worked out the direct (Bresenham) path and its time; reuse
    # that same path here so the energy figure is priced with the same
    # fixed-airspeed model as total_energy_wh above, keeping the two
    # numbers comparable.
    straight_line_path = cpp_out.get("straight_line_path", [])
    straight_line_time = cpp_out.get("straight_line_time")
    straight_line_feasible = cpp_out.get("straight_line_feasible", False)

    straight_line_energy_wh = None
    pct_time_saved = None
    pct_energy_saved = None

    if straight_line_feasible and straight_line_path and straight_line_time and straight_line_time > 0:
        straight_line_energy_wh = estimate_route_energy_wh(
            straight_line_path,
            wind_data,
            grid_coords,
            C,
            payload.step_size,
        )
        pct_time_saved = 100.0 * (straight_line_time - tot_time) / straight_line_time
        if straight_line_energy_wh > 0:
            pct_energy_saved = 100.0 * (straight_line_energy_wh - total_energy_wh) / straight_line_energy_wh

        print(f"[plan-route] Straight-line reference: {straight_line_time:.1f}s, {straight_line_energy_wh:.2f} Wh")
        print(f"[plan-route] Wind-optimized route:    {tot_time:.1f}s, {total_energy_wh:.2f} Wh")
        pct_msg = f"{pct_time_saved:.1f}% time"
        if pct_energy_saved is not None:
            pct_msg += f", {pct_energy_saved:.1f}% energy"
        print(f"[plan-route] Improvement vs straight line: {pct_msg}")
    else:
        print("[plan-route] Straight-line reference is not flyable at this drone's max airspeed "
              "(crosswind too strong on the direct bearing) -- no comparison available.")

    return RouteResponse(
        optimization_mode=payload.mode,
        waypoints=waypoints,
        total_energy_kj=round(total_energy_wh * 3.6, 2),
        total_energy_wh=round(total_energy_wh, 2),
        total_time_seconds=round(tot_time, 1),
        formatted_time=f"{int(tot_time // 60)}m {int(tot_time % 60)}s",
        straight_line_time_seconds=round(straight_line_time, 1) if straight_line_feasible and straight_line_time is not None else None,
        straight_line_energy_wh=round(straight_line_energy_wh, 2) if straight_line_energy_wh is not None else None,
        pct_time_saved_vs_straight_line=round(pct_time_saved, 1) if pct_time_saved is not None else None,
        pct_energy_saved_vs_straight_line=round(pct_energy_saved, 1) if pct_energy_saved is not None else None,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)