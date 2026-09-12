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