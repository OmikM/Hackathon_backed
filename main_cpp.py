"""
Python/FastAPI "entourage" for algorithm.cpp.

Architecture:
    HTTP request
        -> this Python API
        -> Open-Meteo
        -> build a local grid
        -> C++ A* executable
        -> convert grid cells back to local x/y waypoints
        -> JSON response

The routing decision itself is made by algorithm.cpp.
Python is responsible for I/O, weather data, persistent caching and API compatibility.

Caching:
  - repeated identical route requests use route_cache.json and make ZERO
    weather requests
  - weather points are cached for 15 minutes by default
  - only uncached weather points are requested from Open-Meteo
  - set FORCE_REFRESH=1 to deliberately bypass both caches
  - DELETE /api/v1/cache clears both caches
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
ALGORITHM_CPP = BASE_DIR / "algorithm.cpp"
RUNNER_CPP = BASE_DIR / "algorithm_runner.cpp"
RUNNER_EXE = BASE_DIR / (
    "algorithm_runner.exe" if os.name == "nt" else "algorithm_runner"
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------------------------
# Request/cache configuration
# ---------------------------------------------------------------------------

# Persistent cache directory. It survives restarting FastAPI.
CACHE_DIR = BASE_DIR / "cache"
WIND_CACHE_FILE = CACHE_DIR / "wind_cache.json"
ROUTE_CACHE_FILE = CACHE_DIR / "route_cache.json"

# Wind observations are cached for this long. For testing, this means repeated
# requests normally require ZERO Open-Meteo calls.
WIND_CACHE_TTL_SECONDS = int(os.getenv("WIND_CACHE_TTL_SECONDS", "900"))  # 15 min

# Set FORCE_REFRESH=1 when you deliberately want fresh weather.
FORCE_REFRESH = os.getenv("FORCE_REFRESH", "0") == "1"

# Open-Meteo accepts arrays of coordinates. We deliberately make ONE request
# for all missing grid cells rather than one request per cell.
OPEN_METEO_MAX_POINTS_PER_REQUEST = 1000

# Small delay between multiple Open-Meteo batches if a very large grid is used.
OPEN_METEO_BATCH_DELAY_SECONDS = 0.25

CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="C++ Wind-Aware Routing API",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DroneConfig(BaseModel):
    # algorithm.cpp currently uses V_AIR = 15 m/s directly.
    airspeed_ms: float = Field(default=15.0, gt=0)

    # This is NOT used by the C++ routing algorithm.
    # It only provides a useful energy estimate compatible with main.py's
    # response shape.
    cruise_power_w: float = Field(default=400.0, gt=0)

    mass: float = Field(default=6.0, gt=0)
    max_ground_speed: float = Field(default=18.0, gt=0)


class Point2D(BaseModel):
    x: float
    y: float


class WindGridRequest(BaseModel):
    center_lat: float
    center_lon: float
    grid_steps: int = Field(default=1, ge=0, le=10)
    step_deg: float = Field(default=0.05, gt=0)


class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float

    start_point: Point2D
    goal_point: Point2D

    mode: Literal["energy", "speed"] = "speed"

    # This is both the physical grid spacing and the C++ cell size.
    step_size: float = Field(default=80.0, gt=1.0)

    # Number of extra cells around the start/goal corridor.
    grid_margin: int = Field(default=4, ge=0, le=20)


class RouteResponse(BaseModel):
    optimization_mode: str
    active_drone_mass_kg: float
    max_ground_speed_ms: float

    waypoints: list[Point2D]

    total_energy_kj: float | None
    total_energy_wh: float | None

    total_time_seconds: float
    formatted_time: str

    # Extra information useful to the frontend/debugging.
    grid_rows: int
    grid_cols: int
    cell_size_meters: float
    wind_source: str


current_config = DroneConfig()


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def local_point_to_lat_lon(
    origin_lat: float,
    origin_lon: float,
    point: Point2D,
) -> tuple[float, float]:
    """
    Same local-coordinate convention as main.py:
      +x = east
      +y = north
    """
    latitude = origin_lat + point.y / 111000.0
    longitude = origin_lon + point.x / (
        111000.0 * math.cos(math.radians(origin_lat))
    )
    return latitude, longitude


def lat_lon_to_local_point(
    origin_lat: float,
    origin_lon: float,
    latitude: float,
    longitude: float,
) -> Point2D:
    x = (
        (longitude - origin_lon)
        * 111000.0
        * math.cos(math.radians(origin_lat))
    )
    y = (latitude - origin_lat) * 111000.0
    return Point2D(x=x, y=y)


# ---------------------------------------------------------------------------
# C++ compilation
# ---------------------------------------------------------------------------

def ensure_cpp_runner() -> None:
    """
    Compile algorithm_runner.cpp whenever the executable does not exist or
    either C++ source is newer.

    GCC/MinGW is expected to be available as `g++`.
    """
    if not ALGORITHM_CPP.exists():
        raise RuntimeError(f"Missing {ALGORITHM_CPP}")

    if not RUNNER_CPP.exists():
        raise RuntimeError(f"Missing {RUNNER_CPP}")

    needs_build = not RUNNER_EXE.exists()

    if not needs_build:
        newest_source = max(
            ALGORITHM_CPP.stat().st_mtime,
            RUNNER_CPP.stat().st_mtime,
        )
        needs_build = newest_source > RUNNER_EXE.stat().st_mtime

    if not needs_build:
        return

    gxx = shutil.which("g++")
    if gxx is None:
        raise RuntimeError(
            "g++ was not found. Install GCC/MinGW and make sure g++ is on PATH."
        )

    command = [
        gxx,
        "-std=c++17",
        "-O2",
        str(RUNNER_CPP),
        "-o",
        str(RUNNER_EXE),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to compile C++ algorithm:\n"
            + result.stderr
        )


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------

def wind_direction_to_uv(
    speed_ms: float,
    direction_deg: float,
) -> tuple[float, float]:
    """
    Open-Meteo's wind direction is meteorological:
      0°   = wind coming FROM north
      90°  = wind coming FROM east
      180° = wind coming FROM south
      270° = wind coming FROM west

    algorithm.cpp needs the actual wind velocity vector:
      u = eastward component
      v = northward component

    Therefore:
      u = -speed * sin(direction)
      v = -speed * cos(direction)
    """
    direction = math.radians(direction_deg)

    u = -speed_ms * math.sin(direction)
    v = -speed_ms * math.cos(direction)

    return u, v


def _load_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # A corrupted cache should never prevent the API from working.
        return {}


def _save_json_cache(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(path)


def _point_key(lat: float, lon: float) -> str:
    # Six decimals is ~0.1 m in latitude and is more than enough for our grid.
    return f"{lat:.6f},{lon:.6f}"


def _route_cache_key(payload: RouteRequest) -> str:
    """
    The route result is cached separately from the weather cache.

    This is the most important part for testing:
    repeating exactly the same POST does not even call Open-Meteo or C++.
    """
    data = {
        "origin_lat": round(payload.origin_lat, 6),
        "origin_lon": round(payload.origin_lon, 6),
        "start_x": round(payload.start_point.x, 3),
        "start_y": round(payload.start_point.y, 3),
        "goal_x": round(payload.goal_point.x, 3),
        "goal_y": round(payload.goal_point.y, 3),
        "mode": payload.mode,
        "step_size": round(payload.step_size, 3),
        "grid_margin": payload.grid_margin,
        "algorithm_mtime": ALGORITHM_CPP.stat().st_mtime_ns
        if ALGORITHM_CPP.exists()
        else 0,
    }

    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


def fetch_wind(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Fetch wind for all points that are not already in the persistent cache.

    Important optimization:
      - repeated points are de-duplicated
      - cached points require no network request
      - all missing points are sent in as few Open-Meteo requests as possible
      - cached observations expire after WIND_CACHE_TTL_SECONDS
    """
    import time

    cache = _load_json_cache(WIND_CACHE_FILE)
    now = time.time()

    result: list[tuple[float, float] | None] = [None] * len(points)
    missing: list[tuple[int, float, float]] = []

    # First pass: use cached values.
    for index, (lat, lon) in enumerate(points):
        key = _point_key(lat, lon)
        entry = cache.get(key)

        if (
            not FORCE_REFRESH
            and entry
            and now - float(entry.get("timestamp", 0))
            < WIND_CACHE_TTL_SECONDS
        ):
            result[index] = (
                float(entry["u"]),
                float(entry["v"]),
            )
        else:
            missing.append((index, lat, lon))

    # De-duplicate missing coordinates. A rectangular grid normally contains
    # unique coordinates, but this also protects us from accidental duplicates.
    unique_missing: dict[str, tuple[float, float, list[int]]] = {}

    for index, lat, lon in missing:
        key = _point_key(lat, lon)

        if key not in unique_missing:
            unique_missing[key] = (lat, lon, [])

        unique_missing[key][2].append(index)

    unique_items = list(unique_missing.items())

    if unique_items:
        print(
            f"Open-Meteo: {len(unique_items)} uncached wind points "
            f"(cache supplied {len(points) - len(missing)})"
        )

    # Send batches only when necessary. A normal route should fit into one
    # request, so a normal test costs exactly one Open-Meteo request.
    for batch_start in range(
        0,
        len(unique_items),
        OPEN_METEO_MAX_POINTS_PER_REQUEST,
    ):
        batch = unique_items[
            batch_start:
            batch_start + OPEN_METEO_MAX_POINTS_PER_REQUEST
        ]

        lats = ",".join(f"{lat:.6f}" for _, (lat, _, _) in batch)
        lons = ",".join(f"{lon:.6f}" for _, (_, lon, _) in batch)

        params = {
            "latitude": lats,
            "longitude": lons,
            "current": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
        }

        try:
            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=20,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "unknown")
                raise RuntimeError(
                    "Open-Meteo returned HTTP 429 (rate limited). "
                    f"Retry-After: {retry_after}. "
                    "Try again later or use the cached data."
                )

            response.raise_for_status()
            data = response.json()

        except requests.RequestException as exc:
            raise RuntimeError(f"Open-Meteo request failed: {exc}") from exc

        if isinstance(data, dict):
            data = [data]

        if len(data) != len(batch):
            raise RuntimeError(
                f"Open-Meteo returned {len(data)} locations for "
                f"{len(batch)} requested locations."
            )

        for item, (_, (lat, lon, indexes)) in zip(data, batch):
            current = item.get("current", {})

            speed = current.get("wind_speed_10m")
            direction = current.get("wind_direction_10m")

            if speed is None or direction is None:
                raise RuntimeError(
                    "Open-Meteo response did not contain current wind data."
                )

            u, v = wind_direction_to_uv(
                float(speed),
                float(direction),
            )

            key = _point_key(lat, lon)

            cache[key] = {
                "timestamp": now,
                "u": u,
                "v": v,
            }

            for index in indexes:
                result[index] = (u, v)

        # Persist after every successful batch. If a later batch fails, the
        # successful data is still available for the next attempt.
        _save_json_cache(WIND_CACHE_FILE, cache)

        if batch_start + OPEN_METEO_MAX_POINTS_PER_REQUEST < len(unique_items):
            time.sleep(OPEN_METEO_BATCH_DELAY_SECONDS)

    # Every point must now have a value.
    if any(value is None for value in result):
        raise RuntimeError("Internal error: some wind grid cells have no data.")

    return [(float(u), float(v)) for u, v in result]


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid(
    start: Point2D,
    goal: Point2D,
    cell_size: float,
    margin: int,
) -> tuple[int, int, int, int, list[tuple[float, float]]]:
    """
    Build a rectangular grid around the route.

    The grid is expressed in local metres relative to the API origin.

    Returns:
        rows, cols, start_row, start_col, goal_row, goal_col, coordinates
    """
    # Put the grid's lower-left-ish area around both endpoints.
    min_x = min(start.x, goal.x)
    max_x = max(start.x, goal.x)
    min_y = min(start.y, goal.y)
    max_y = max(start.y, goal.y)

    # Expand around the corridor.
    min_x -= margin * cell_size
    max_x += margin * cell_size
    min_y -= margin * cell_size
    max_y += margin * cell_size

    cols = max(2, math.ceil((max_x - min_x) / cell_size) + 1)
    rows = max(2, math.ceil((max_y - min_y) / cell_size) + 1)

    def nearest_col(x: float) -> int:
        return round((x - min_x) / cell_size)

    def nearest_row(y: float) -> int:
        # C++ row increases downward, while local y increases northward.
        return round((max_y - y) / cell_size)

    start_col = nearest_col(start.x)
    start_row = nearest_row(start.y)

    goal_col = nearest_col(goal.x)
    goal_row = nearest_row(goal.y)

    # Clamp in case floating-point rounding hits the boundary.
    start_col = max(0, min(cols - 1, start_col))
    start_row = max(0, min(rows - 1, start_row))
    goal_col = max(0, min(cols - 1, goal_col))
    goal_row = max(0, min(rows - 1, goal_row))

    coordinates: list[tuple[float, float]] = []

    for r in range(rows):
        y = max_y - r * cell_size
        for c in range(cols):
            x = min_x + c * cell_size

            lat, lon = local_point_to_lat_lon(
                ORIGIN_LAT_FOR_GRID,
                ORIGIN_LON_FOR_GRID,
                Point2D(x=x, y=y),
            )
            coordinates.append((lat, lon))

    return (
        rows,
        cols,
        start_row,
        start_col,
        goal_row,
        goal_col,
        coordinates,
    )


# These are set only while planning a route. Keeping them global avoids
# threading a pair of values through the low-level grid helper.
ORIGIN_LAT_FOR_GRID = 0.0
ORIGIN_LON_FOR_GRID = 0.0


# ---------------------------------------------------------------------------
# C++ execution
# ---------------------------------------------------------------------------

def run_cpp_algorithm(
    rows: int,
    cols: int,
    cell_size: float,
    start_row: int,
    start_col: int,
    goal_row: int,
    goal_col: int,
    wind: list[tuple[float, float]],
) -> tuple[float, list[tuple[int, int]]]:
    ensure_cpp_runner()

    if len(wind) != rows * cols:
        raise RuntimeError(
            f"Expected {rows * cols} wind cells, got {len(wind)}."
        )

    lines = [
        f"{rows} {cols} {cell_size} "
        f"{start_row} {start_col} {goal_row} {goal_col}"
    ]

    lines.extend(
        f"{u:.8f} {v:.8f}"
        for u, v in wind
    )

    payload = "\n".join(lines) + "\n"

    try:
        result = subprocess.run(
            [str(RUNNER_EXE)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("C++ routing algorithm timed out.") from exc

    if result.returncode != 0:
        raise RuntimeError(
            "C++ algorithm failed:\n"
            + result.stderr
        )

    output = result.stdout.strip().splitlines()

    if not output or not output[0].startswith("TIME "):
        raise RuntimeError(
            "Unexpected C++ output:\n" + result.stdout
        )

    time_value = output[0].split(maxsplit=1)[1]

    if time_value == "INF":
        return math.inf, []

    total_time = float(time_value)

    if len(output) < 2 or not output[1].startswith("PATH "):
        raise RuntimeError("C++ output does not contain a PATH line.")

    path_len = int(output[1].split()[1])

    path: list[tuple[int, int]] = []

    for line in output[2:2 + path_len]:
        r, c = map(int, line.split())
        path.append((r, c))

    if len(path) != path_len:
        raise RuntimeError("C++ returned an incomplete path.")

    return total_time, path



def _load_route_cache() -> dict:
    return _load_json_cache(ROUTE_CACHE_FILE)


def _save_route_cache(cache: dict) -> None:
    _save_json_cache(ROUTE_CACHE_FILE, cache)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/config", response_model=DroneConfig)
def get_drone_config():
    return current_config


@app.put("/api/v1/config", response_model=DroneConfig)
def update_drone_config(new_config: DroneConfig):
    global current_config
    current_config = new_config
    return current_config


@app.post("/api/v1/wind-grid")
def get_wind_grid_endpoint(payload: WindGridRequest):
    """
    Compatibility endpoint similar to main.py.

    This endpoint returns a simple square of real-time Open-Meteo wind data.
    """
    points: list[tuple[float, float]] = []

    for y_step in range(-payload.grid_steps, payload.grid_steps + 1):
        for x_step in range(-payload.grid_steps, payload.grid_steps + 1):
            point = Point2D(
                x=x_step * payload.step_deg * 111000.0,
                y=y_step * payload.step_deg * 111000.0,
            )
            points.append(
                local_point_to_lat_lon(
                    payload.center_lat,
                    payload.center_lon,
                    point,
                )
            )

    wind = fetch_wind(points)

    grid_points = []

    for (lat, lon), (u, v) in zip(points, wind):
        grid_points.append(
            {
                "latitude": lat,
                "longitude": lon,
                "u_wind_ms": round(u, 3),
                "v_wind_ms": round(v, 3),
            }
        )

    return {
        "center_lat": payload.center_lat,
        "center_lon": payload.center_lon,
        "grid_points": grid_points,
        "source": "Open-Meteo current 10 m wind",
    }


@app.delete("/api/v1/cache")
def clear_cache():
    """
    Delete cached route results and weather.

    Normally you should NOT call this during testing.
    """
    removed = []

    for path in (WIND_CACHE_FILE, ROUTE_CACHE_FILE):
        if path.exists():
            path.unlink()
            removed.append(path.name)

    return {
        "cleared": removed,
        "message": "Caches cleared.",
    }


@app.post("/api/v1/plan-route", response_model=RouteResponse)
def plan_route_endpoint(payload: RouteRequest):
    global ORIGIN_LAT_FOR_GRID, ORIGIN_LON_FOR_GRID

    # Return the complete previous result immediately.
    # This avoids BOTH the Open-Meteo request and the C++ execution.
    route_key = _route_cache_key(payload)
    route_cache = _load_route_cache()

    if not FORCE_REFRESH and route_key in route_cache:
        print(f"Route cache HIT: {route_key[:12]}")
        return RouteResponse(**route_cache[route_key])

    print(f"Route cache MISS: {route_key[:12]}")

    if not (-90 <= payload.origin_lat <= 90):
        raise HTTPException(400, "Invalid origin latitude.")

    if not (-180 <= payload.origin_lon <= 180):
        raise HTTPException(400, "Invalid origin longitude.")

    if payload.start_point.x == payload.goal_point.x and \
       payload.start_point.y == payload.goal_point.y:
        return RouteResponse(
            optimization_mode=payload.mode,
            active_drone_mass_kg=current_config.mass,
            max_ground_speed_ms=current_config.max_ground_speed,
            waypoints=[payload.start_point],
            total_energy_kj=0.0,
            total_energy_wh=0.0,
            total_time_seconds=0.0,
            formatted_time="0m 0s",
            grid_rows=1,
            grid_cols=1,
            cell_size_meters=payload.step_size,
            wind_source="Open-Meteo current 10 m wind",
        )

    ORIGIN_LAT_FOR_GRID = payload.origin_lat
    ORIGIN_LON_FOR_GRID = payload.origin_lon

    start_latlon = local_point_to_lat_lon(
        payload.origin_lat,
        payload.origin_lon,
        payload.start_point,
    )
    goal_latlon = local_point_to_lat_lon(
        payload.origin_lat,
        payload.origin_lon,
        payload.goal_point,
    )

    (
        rows,
        cols,
        start_row,
        start_col,
        goal_row,
        goal_col,
        grid_latlon,
    ) = build_grid(
        payload.start_point,
        payload.goal_point,
        payload.step_size,
        payload.grid_margin,
    )

    # Real-time weather data for every C++ cell.
    wind = fetch_wind(grid_latlon)

    total_time, path = run_cpp_algorithm(
        rows=rows,
        cols=cols,
        cell_size=payload.step_size,
        start_row=start_row,
        start_col=start_col,
        goal_row=goal_row,
        goal_col=goal_col,
        wind=wind,
    )

    if not path or not math.isfinite(total_time):
        raise HTTPException(
            status_code=422,
            detail=(
                "No physically feasible route exists for the supplied "
                "wind field and 15 m/s aircraft airspeed."
            ),
        )

    # Reconstruct the same local coordinate system used by main.py.
    min_x = min(payload.start_point.x, payload.goal_point.x) \
        - payload.grid_margin * payload.step_size
    max_x = max(payload.start_point.x, payload.goal_point.x) \
        + payload.grid_margin * payload.step_size
    min_y = min(payload.start_point.y, payload.goal_point.y) \
        - payload.grid_margin * payload.step_size
    max_y = max(payload.start_point.y, payload.goal_point.y) \
        + payload.grid_margin * payload.step_size

    waypoints: list[Point2D] = []

    for r, c in path:
        x = min_x + c * payload.step_size
        y = max_y - r * payload.step_size

        waypoints.append(Point2D(x=x, y=y))

    # The C++ algorithm optimizes TIME, not ENERGY.
    # Give the API an energy estimate only for frontend compatibility.
    estimated_energy_wh = (
        current_config.cruise_power_w * total_time / 3600.0
    )

    response = RouteResponse(
        optimization_mode="speed",
        active_drone_mass_kg=current_config.mass,
        max_ground_speed_ms=current_config.max_ground_speed,
        waypoints=waypoints,
        total_energy_kj=round(estimated_energy_wh * 3.6, 2),
        total_energy_wh=round(estimated_energy_wh, 2),
        total_time_seconds=round(total_time, 1),
        formatted_time=f"{int(total_time // 60)}m {int(total_time % 60)}s",
        grid_rows=rows,
        grid_cols=cols,
        cell_size_meters=payload.step_size,
        wind_source="Open-Meteo current 10 m wind",
    )

    # Persist the exact JSON-compatible response.
    route_cache[route_key] = response.model_dump()
    _save_route_cache(route_cache)

    print(f"Route cache SAVE: {route_key[:12]}")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_cpp:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
