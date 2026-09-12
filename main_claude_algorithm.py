import math
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from some_claude_algoritm import (
    DroneSpecs,
    find_optimal_path,
    get_optimal_edge,
    haversine,
    mock_wind_field,
)

app = FastAPI(
    title="Claude Wind-Aware Routing API",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DroneConfig(BaseModel):
    mass: float = Field(default=6.3)
    max_speed: float = Field(default=23.0)
    avionics_power_w: float = Field(default=30.0)
    max_wind_resistance: float = Field(default=15.0)


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
    waypoints: List[Point2D]
    total_energy_kj: float
    total_energy_wh: float
    total_time_seconds: float
    formatted_time: str
    straight_line_time_seconds: Optional[float] = None
    straight_line_energy_wh: Optional[float] = None
    pct_time_saved_vs_straight_line: Optional[float] = None
    pct_energy_saved_vs_straight_line: Optional[float] = None


ENGINE_SPECS = DroneSpecs()
current_config = DroneConfig()


def local_point_to_lat_lon(
    origin_lat: float,
    origin_lon: float,
    point: Point2D,
) -> Tuple[float, float]:
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
    x = (longitude - origin_lon) * 111000.0 * math.cos(math.radians(origin_lat))
    y = (latitude - origin_lat) * 111000.0
    return Point2D(x=x, y=y)


def wind_vector_to_uv(speed: float, direction_from_deg: float) -> Tuple[float, float]:
    direction_to_rad = math.radians((direction_from_deg + 180.0) % 360.0)
    return (
        speed * math.sin(direction_to_rad),
        speed * math.cos(direction_to_rad),
    )


def get_config() -> DroneConfig:
    return current_config


@app.get("/api/v1/config", response_model=DroneConfig)
def get_drone_config():
    return get_config()


@app.put("/api/v1/config", response_model=DroneConfig)
def update_drone_config(new_config: DroneConfig):
    global current_config, ENGINE_SPECS
    current_config = new_config
    ENGINE_SPECS = DroneSpecs(
        mass_kg=new_config.mass,
        max_speed_mps=new_config.max_speed,
        avionics_payload_power_w=new_config.avionics_power_w,
        max_wind_resistance_mps=new_config.max_wind_resistance,
    )
    return current_config


@app.post("/api/v1/wind-grid")
def get_wind_grid_endpoint(payload: WindGridRequest):
    grid_points = []
    for row in range(-payload.grid_steps, payload.grid_steps + 1):
        for col in range(-payload.grid_steps, payload.grid_steps + 1):
            latitude = payload.center_lat + row * payload.step_deg
            longitude = payload.center_lon + col * payload.step_deg
            speed, direction = mock_wind_field(latitude, longitude)
            u_wind, v_wind = wind_vector_to_uv(speed, direction)
            grid_points.append({
                "x_offset_m": (longitude - payload.center_lon)
                * 111000.0 * math.cos(math.radians(payload.center_lat)),
                "y_offset_m": (latitude - payload.center_lat) * 111000.0,
                "u_wind_ms": u_wind,
                "v_wind_ms": v_wind,
            })
    return {
        "center_lat": payload.center_lat,
        "center_lon": payload.center_lon,
        "grid_points": grid_points,
    }


def calculate_straight_line(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    specs: DroneSpecs,
) -> Tuple[Optional[float], Optional[float], bool]:
    distance_m, bearing = haversine(*point_a, *point_b)
    midpoint = ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)
    wind_speed, wind_direction = mock_wind_field(*midpoint)
    energy_wh, time_s = get_optimal_edge(
        distance_m,
        bearing,
        wind_speed,
        wind_direction,
        specs,
    )
    if not math.isfinite(energy_wh) or not math.isfinite(time_s):
        return None, None, False
    return energy_wh, time_s, True


@app.post("/api/v1/plan-route", response_model=RouteResponse)
def plan_route_endpoint(payload: RouteRequest):
    if payload.step_size <= 0:
        raise ValueError("step_size must be greater than zero")

    point_a = local_point_to_lat_lon(
        payload.origin_lat,
        payload.origin_lon,
        payload.start_point,
    )
    point_b = local_point_to_lat_lon(
        payload.origin_lat,
        payload.origin_lon,
        payload.goal_point,
    )

    optimal_path, total_energy_wh, total_time = find_optimal_path(
        point_a[0],
        point_a[1],
        point_b[0],
        point_b[1],
        ENGINE_SPECS,
    )
    if not optimal_path or not math.isfinite(total_energy_wh) or not math.isfinite(total_time):
        raise ValueError("The Claude route engine could not find a flyable route")

    waypoints = [
        lat_lon_to_local_point(payload.origin_lat, payload.origin_lon, latitude, longitude)
        for latitude, longitude in optimal_path
    ]
    waypoints[0] = payload.start_point
    waypoints[-1] = payload.goal_point

    straight_line_energy_wh, straight_line_time, straight_line_feasible = calculate_straight_line(
        point_a,
        point_b,
        ENGINE_SPECS,
    )

    pct_time_saved = None
    pct_energy_saved = None
    if straight_line_feasible and straight_line_time and straight_line_time > 0:
        pct_time_saved = 100.0 * (straight_line_time - total_time) / straight_line_time
    if straight_line_feasible and straight_line_energy_wh and straight_line_energy_wh > 0:
        pct_energy_saved = 100.0 * (straight_line_energy_wh - total_energy_wh) / straight_line_energy_wh

    return RouteResponse(
        optimization_mode=payload.mode,
        waypoints=waypoints,
        total_energy_kj=round(total_energy_wh * 3.6, 2),
        total_energy_wh=round(total_energy_wh, 2),
        total_time_seconds=round(total_time, 1),
        formatted_time=f"{int(total_time // 60)}m {int(total_time % 60)}s",
        straight_line_time_seconds=(
            round(straight_line_time, 1)
            if straight_line_feasible and straight_line_time is not None
            else None
        ),
        straight_line_energy_wh=(
            round(straight_line_energy_wh, 2)
            if straight_line_energy_wh is not None
            else None
        ),
        pct_time_saved_vs_straight_line=(
            round(pct_time_saved, 1) if pct_time_saved is not None else None
        ),
        pct_energy_saved_vs_straight_line=(
            round(pct_energy_saved, 1) if pct_energy_saved is not None else None
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
