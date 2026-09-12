import os
import sys
import json
import time
import math
import subprocess
from typing import List, Tuple, Dict, Literal
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="C++ Energy-Optimal Wind-Aware Routing API",
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
# Deliberately the SAME cache file as main_cpp.py: both services fetch the
# exact same current_weather field from Open-Meteo, so there's no reason to
# pay for the API call twice just because two different solvers want it.
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cpp_route_cache.json")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALGORITHM_NAME = "algorithm_energy_cpp.exe" if os.name == "nt" else "algorithm_energy_cpp"
FALLBACK_ALGORITHM_NAME = "algorithm_energy.exe" if os.name == "nt" else "algorithm_energy"
ALGORITHM_PATH = os.path.join(
    BASE_DIR,
    ALGORITHM_NAME if os.path.isfile(os.path.join(BASE_DIR, ALGORITHM_NAME)) else FALLBACK_ALGORITHM_NAME,
)
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
    # This service only ever solves for minimum energy -- there's no
    # "speed" mode here, that's what main_cpp.py / algorithm.cpp are for.
    # Kept as a field (rather than dropped) so the two services share a
    # request shape and a frontend can point at either one interchangeably.
    mode: Literal["energy"] = "energy"
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
    # Per-edge ground speed the solver actually chose, aligned with
    # waypoints[i] -> waypoints[i+1] (so len == len(waypoints) - 1). This is
    # the whole point of the energy-optimal solver over the fixed-airspeed
    # one: speed is a decision variable, not a constant, and callers that
    # want to *show* that (e.g. animate the drone slowing down into a
    # headwind) need it exposed rather than re-derived.
    edge_ground_speeds_mps: List[float]


# --- UTILITIES (identical to main_cpp.py) ---
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


# --- DISK CACHE MANAGEMENT (identical to main_cpp.py) ---
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

# --- RATE-LIMITED OPEN-METEO DATA FETCHING (identical to main_cpp.py) ---
def fetch_wind_grid_batch(coords: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Tuple[float, float]]:
    """Fetches wind speed/direction in a SINGLE batched request with persistent caching and 429 protection."""
    global LAST_API_CALL_TIME
    cache = load_cache()
    now = time.time()

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

    chunk_size = 50
    unique_missing = list({(b_key): (la, lo) for la, lo, b_key in missing_coords}.items())

    for i in range(0, len(unique_missing), chunk_size):
        chunk = unique_missing[i:i + chunk_size]
        lats = ",".join(f"{la:.4f}" for b_key, (la, lo) in chunk)
        lons = ",".join(f"{lo:.4f}" for b_key, (la, lo) in chunk)

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

        try:
            with urlopen(req, timeout=15) as resp:
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

        except HTTPError as e:
            if e.code == 429:
                print("Hit 429 Rate Limit! Waiting 5s...")
                time.sleep(5)
            else:
                raise HTTPException(status_code=500, detail=f"Weather API Error: {e}")

    save_cache(cache)

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
            detail=f"Routing binary not found: {ALGORITHM_PATH}. Compile algorithm_energy.cpp first.",
        )

    if payload.step_size <= 0:
        raise HTTPException(status_code=422, detail="step_size must be greater than zero.")
    grid_origin_x = min(0.0, payload.start_point.x, payload.goal_point.x)
    grid_origin_y = min(0.0, payload.start_point.y, payload.goal_point.y)
    start_r = round((payload.start_point.y - grid_origin_y) / payload.step_size)
    start_c = round((payload.start_point.x - grid_origin_x) / payload.step_size)
    goal_r = round((payload.goal_point.y - grid_origin_y) / payload.step_size)
    goal_c = round((payload.goal_point.x - grid_origin_x) / payload.step_size)

    R = max(payload.grid_rows, start_r + 1, goal_r + 1)
    C = max(payload.grid_cols, start_c + 1, goal_c + 1)

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

    wind_data = fetch_wind_grid_batch(grid_coords)

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

    cmd = [ALGORITHM_PATH, str(R), str(C), str(start_r), str(start_c), str(goal_r), str(goal_c), str(payload.step_size)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(input=input_payload)

    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"C++ execution failed (exit code {proc.returncode}): {stderr.strip()}",
        )

    try:
        cpp_out = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON from energy solver: {error}. Output: {stdout.strip()}",
        ) from error
    cpp_path = cpp_out.get("path", [])
    if not cpp_path:
        raise HTTPException(status_code=422, detail="C++ solver could not find a flyable route.")

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
    # The C++ solver already integrated the exact ground speed it chose for
    # every edge, so its total_energy_wh is authoritative -- unlike
    # main_cpp.py's Python-side estimate_route_energy_wh, which had to
    # assume a single fixed airspeed because that's all algorithm.cpp gave
    # it. Recomputing energy here from a different, coarser assumption
    # would silently disagree with what the solver actually flew.
    if "total_energy_wh" not in cpp_out or "edge_speeds_mps" not in cpp_out:
        raise HTTPException(
            status_code=500,
            detail="Energy solver output is missing total_energy_wh or edge_speeds_mps.",
        )
    total_energy_wh = cpp_out["total_energy_wh"]
    edge_speeds = cpp_out.get("edge_speeds_mps", [])

    return RouteResponse(
        optimization_mode=payload.mode,
        waypoints=waypoints,
        total_energy_kj=round(total_energy_wh * 3.6, 2),
        total_energy_wh=round(total_energy_wh, 2),
        total_time_seconds=round(tot_time, 1),
        formatted_time=f"{int(tot_time // 60)}m {int(tot_time % 60)}s",
        edge_ground_speeds_mps=[round(s, 2) for s in edge_speeds],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)