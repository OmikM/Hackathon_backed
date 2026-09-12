"""
Wind-aware route energy planner for a multirotor drone.

Combines:
  - Real published specs for a DJI Matrice 300 RTK (commercial industrial
    quadcopter, source: DJI's official spec sheet, dji.com/support/product/matrice-300)
  - The Zeng, Xu & Zhang (2019) rotary-wing UAV propulsion power model
    ("Energy Minimization for Wireless Communication with Rotary-Wing UAV",
    IEEE Trans. Wireless Commun., vol 18, no 4) with its commonly-cited
    reference parameter set, since manufacturers don't publish drag
    coefficients / rotor aerodynamics.
  - Live wind + temperature + pressure data from the Open-Meteo Forecast API
    (api.open-meteo.com, free, no key, updated hourly from NOAA HRRR/GFS,
    DWD ICON, etc. depending on location).
  - Live terrain elevation from the Open-Meteo Elevation API
    (Copernicus GLO-90 DEM).

It builds a small lattice of candidate routes between two points at a FIXED
cruise altitude AGL, prices every edge in Watt-hours using real wind at that
edge, and finds the minimum-energy path with Dijkstra -- then compares it to
the straight-line route. This is the same idea as Zermelo's navigation
problem, just solved numerically on a grid with a power-cost model instead
of pure minimum-time.

Each edge's ground speed is itself optimized rather than fixed: for the wind
at that edge, the planner searches for the ground speed that minimizes
Watt-hours, bounded by the airframe's max airspeed. This lets a tailwind pay
off through BOTH lower propulsion power and less time aloft, instead of
power reduction alone -- see section 6 for why that distinction matters.

NOTE ON ASSUMPTIONS: values marked "ASSUMED" below are not published by DJI
and had to be estimated / chosen for the demo; everything else is cited.
"""

import math
import heapq
import os
import sys
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import urlencode
import json
import time

# --------------------------------------------------------------------------
# 1. DRONE SPECS -- DJI Matrice 300 RTK
# --------------------------------------------------------------------------

@dataclass
class DroneSpecs:
    name: str = "DJI Matrice 300 RTK"

    # --- Published by DJI (dji.com/support/product/matrice-300) ---
    mass_kg: float = 6.3                 # airframe + 2x TB60 batteries, no payload
    max_takeoff_weight_kg: float = 9.0
    battery_wh_each: float = 274.0       # TB60 Intelligent Flight Battery
    n_batteries: int = 2
    max_speed_mps: float = 23.0          # S-mode horizontal -- also used below as
                                          # the ceiling on airspeed (TAS) when the
                                          # optimizer picks a per-edge ground speed
    max_wind_resistance_mps: float = 15.0
    rated_hover_time_s: float = 55 * 60  # no payload, windless, ~8 m/s per DJI's test note
    diagonal_wheelbase_m: float = 0.895

    # --- ASSUMED (not published by DJI; typical values for this class) ---
    usable_energy_fraction: float = 0.80   # ASSUMED: reserve for landing/safety margin
    avionics_payload_power_w: float = 30.0 # ASSUMED: flight controller, radios, gimbal/camera hotel load
    # NOTE: there used to be a fixed `cruise_ground_speed_mps` here. It's gone --
    # ground speed is now a per-edge decision variable (see section 6) rather
    # than a constant, so there's no single cruise speed to name.

    # --- Aerodynamic/power-model parameters ---
    # Generic reference rotary-wing UAV values from Zeng, Xu & Zhang (2019),
    # widely reused in UAV trajectory-optimization literature. NOT specific
    # to the M300 RTK -- DJI does not publish blade/rotor aerodynamics --
    # but they are the standard stand-in used for exactly this kind of model.
    P0: float = 79.86     # W, blade profile power at hover
    Pi: float = 88.63     # W, induced power at hover
    U_tip: float = 120.0  # m/s, rotor blade tip speed
    v0: float = 4.03      # m/s, mean rotor induced velocity at hover
    d0: float = 0.6       # fuselage drag ratio
    s: float = 0.05       # rotor solidity
    A: float = 0.503      # m^2, rotor disc area

    @property
    def usable_energy_wh(self) -> float:
        return self.battery_wh_each * self.n_batteries * self.usable_energy_fraction


DRONE = DroneSpecs()


# --------------------------------------------------------------------------
# 2. POWER MODEL (Zeng et al. 2019, eq. for level forward flight)
# --------------------------------------------------------------------------

def propulsion_power_w(v_air_mps: float, rho: float, d: DroneSpecs) -> float:
    """Power (W) to fly at airspeed v_air (m/s) through air of density rho (kg/m^3)."""
    blade_profile = d.P0 * (1 + 3 * v_air_mps ** 2 / d.U_tip ** 2)
    induced = d.Pi * (
        math.sqrt(1 + v_air_mps ** 4 / (4 * d.v0 ** 4)) - v_air_mps ** 2 / (2 * d.v0 ** 2)
    ) ** 0.5
    parasite = 0.5 * d.d0 * rho * d.s * d.A * v_air_mps ** 3
    return blade_profile + induced + parasite


def air_density_kgm3(temp_c: float, surface_pressure_hpa: float, height_agl_m: float) -> float:
    """Air density at height_agl_m using the barometric formula from surface pressure."""
    t_k = temp_c + 273.15
    pressure_at_height_hpa = surface_pressure_hpa * (1 - 0.0065 * height_agl_m / t_k) ** 5.255
    return (pressure_at_height_hpa * 100) / (287.05 * t_k)  # ideal gas law, R_specific=287.05


# --------------------------------------------------------------------------
# 3. LOCAL CACHE -- cuts API calls to (close to) zero on repeat runs
# --------------------------------------------------------------------------
#
# Three independent tricks, all stacked:
#   1. Spatial bucketing: snap every lattice point to a coarse grid before
#      looking it up or fetching it. Weather models themselves only have
#      ~2-25 km native resolution (HRRR ~3km, ICON ~2-11km, GFS ~13-25km) --
#      asking for two points 500m apart returns interpolated near-duplicates
#      anyway, so there's no real information lost by sharing one fetch
#      across everything inside one bucket.
#   2. Disk persistence: the cache survives between runs, so restarting the
#      script 20 times while debugging doesn't cost 20x the API calls --
#      only the first run of the session pays for real fetches.
#   3. TTL per data type: wind is cached for WIND_CACHE_TTL_S (it changes
#      hour to hour), elevation is cached forever (terrain doesn't move).

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".drone_route_cache.json")
# IMPORTANT: this must stay FINER than the lattice's own lane spacing
# (offsets_km in build_lattice, default 3 km apart) -- otherwise two
# genuinely different candidate lanes round into the same bucket and the
# optimizer loses the exact wind difference it's supposed to be finding.
# 0.01 deg =~ 1.1 km, comfortably finer than a 3 km lane spacing while still
# coarser than the weather model's own resolution, so it's a safe merge.
CACHE_SPATIAL_DEG = 0.01
WIND_CACHE_TTL_S = 30 * 60        # Open-Meteo model data is refreshed hourly
NO_CACHE = "--no-cache" in sys.argv  # pass --no-cache to force fresh fetches


def _bucket(lat: float, lon: float):
    d = CACHE_SPATIAL_DEG
    return (round(lat / d) * d, round(lon / d) * d)

def _bucket_key(bucket) -> str:
    return f"{bucket[0]:.4f},{bucket[1]:.4f}"


def _load_cache() -> dict:
    if NO_CACHE or not os.path.exists(CACHE_PATH):
        return {"wind": {}, "elevation": {}}
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"wind": {}, "elevation": {}}

def _save_cache(cache: dict):
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass  # cache is a nice-to-have, never fatal


_CACHE = _load_cache()
_STATS = {"wind_hits": 0, "wind_fetched": 0, "elev_hits": 0, "elev_fetched": 0}


# --------------------------------------------------------------------------
# 4. LIVE DATA -- Open-Meteo (wind + temperature + pressure), no API key
# --------------------------------------------------------------------------

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEVATION = "https://api.open-meteo.com/v1/elevation"
CRUISE_AGL_M = 120  # Open-Meteo's 120m-AGL level ~ matches the FAA Part 107 400ft ceiling


def _get_json_with_retry(url: str, max_retries: int = 5):
    """GET url with a User-Agent header. Honors Retry-After on 429 if the
    server sends one, otherwise falls back to exponential backoff."""
    req = Request(url, headers={"User-Agent": "drone-wind-route-planner/1.0 (research demo)"})
    delay = 2.0
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after else delay
                print(f"  ... got HTTP {e.code}, waiting {wait:.0f}s before retrying "
                      f"({attempt + 1}/{max_retries})")
                time.sleep(wait)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def _fetch_wind_batch_raw(bucket_coords):
    """One HTTP request for every bucket coordinate that wasn't cached."""
    lats = ",".join(f"{la:.4f}" for la, lo in bucket_coords)
    lons = ",".join(f"{lo:.4f}" for la, lo in bucket_coords)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": f"wind_speed_{CRUISE_AGL_M}m,wind_direction_{CRUISE_AGL_M}m,"
                  f"temperature_{CRUISE_AGL_M}m,temperature_2m,surface_pressure",
        "forecast_hours": 1,
        "wind_speed_unit": "ms",
    }
    url = f"{OPEN_METEO_FORECAST}?{urlencode(params)}"
    data = _get_json_with_retry(url)
    entries = data if isinstance(data, list) else [data]

    out = {}
    for bucket, entry in zip(bucket_coords, entries):
        hourly = entry["hourly"]
        temperature_120m = hourly[f"temperature_{CRUISE_AGL_M}m"][0]
        temperature_2m = hourly["temperature_2m"][0]
        out[bucket] = {
            "wind_speed": hourly[f"wind_speed_{CRUISE_AGL_M}m"][0] or 0.0,
            "wind_dir_from_deg": hourly[f"wind_direction_{CRUISE_AGL_M}m"][0] or 0.0,
            "temp_c": temperature_120m if temperature_120m is not None else temperature_2m,
            "surface_pressure_hpa": hourly["surface_pressure"][0] or 1013.25,
            "time": hourly["time"][0],
        }
    return out


def fetch_wind_and_atmos_batch(nodes):
    """
    Cached, bucketed, batched wind/temperature/pressure lookup.
    Every node is snapped to a CACHE_SPATIAL_DEG bucket; buckets already in
    the on-disk cache (and younger than WIND_CACHE_TTL_S) are reused for
    free, and only the missing/stale buckets go out in ONE new request.
    """
    now = time.time()
    node_bucket = {node: _bucket(*node) for node in nodes}
    unique_buckets = sorted(set(node_bucket.values()))

    to_fetch = []
    for b in unique_buckets:
        entry = _CACHE["wind"].get(_bucket_key(b))
        cached_data = entry.get("data", {}) if entry else {}
        cache_has_atmosphere = (
            cached_data.get("temp_c") is not None
            and cached_data.get("surface_pressure_hpa") is not None
        )
        if entry and cache_has_atmosphere and now - entry["fetched_at"] < WIND_CACHE_TTL_S:
            _STATS["wind_hits"] += 1
        else:
            to_fetch.append(b)

    if to_fetch:
        fresh = _fetch_wind_batch_raw(to_fetch)
        _STATS["wind_fetched"] += len(to_fetch)
        for b, data in fresh.items():
            _CACHE["wind"][_bucket_key(b)] = {"fetched_at": now, "data": data}
        _save_cache(_CACHE)

    return {node: _CACHE["wind"][_bucket_key(node_bucket[node])]["data"] for node in nodes}


def fetch_elevations(coords):
    """Cached, bucketed terrain elevation (Copernicus GLO-90 DEM). Terrain
    doesn't change, so once a bucket is fetched it is cached forever."""
    node_bucket = {c: _bucket(*c) for c in coords}
    unique_buckets = sorted(set(node_bucket.values()))

    to_fetch = [b for b in unique_buckets if _bucket_key(b) not in _CACHE["elevation"]]
    _STATS["elev_hits"] += len(unique_buckets) - len(to_fetch)

    if to_fetch:
        lats = ",".join(f"{la:.4f}" for la, lo in to_fetch)
        lons = ",".join(f"{lo:.4f}" for la, lo in to_fetch)
        url = f"{OPEN_METEO_ELEVATION}?{urlencode({'latitude': lats, 'longitude': lons})}"
        data = _get_json_with_retry(url)
        _STATS["elev_fetched"] += len(to_fetch)
        for b, elev in zip(to_fetch, data["elevation"]):
            _CACHE["elevation"][_bucket_key(b)] = elev
        _save_cache(_CACHE)

    return [_CACHE["elevation"][_bucket_key(node_bucket[c])] for c in coords]


# --------------------------------------------------------------------------
# 5. GEOMETRY -- build a lattice of candidate routes between A and B
# --------------------------------------------------------------------------

R_EARTH_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(a))

def initial_bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def destination_point(lat, lon, bearing_deg, dist_km):
    br = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    ang = dist_km / R_EARTH_KM
    p2 = math.asin(math.sin(p1) * math.cos(ang) + math.cos(p1) * math.sin(ang) * math.cos(br))
    l2 = l1 + math.atan2(
        math.sin(br) * math.sin(ang) * math.cos(p1),
        math.cos(ang) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), math.degrees(l2)

def bearing_to_vec(bearing_deg, magnitude):
    r = math.radians(bearing_deg)
    return magnitude * math.sin(r), magnitude * math.cos(r)  # (east, north)

def path_length_km(path):
    return sum(haversine_km(*path[i], *path[i + 1]) for i in range(len(path) - 1))


def build_lattice(a, b, n_layers=5, offsets_km=(-6, -3, 0, 3, 6)):
    """n_layers points along the direct track; each interior layer gets lateral offsets."""
    dist = haversine_km(*a, *b)
    brg = initial_bearing_deg(*a, *b)
    layers = []
    for i in range(n_layers):
        frac = i / (n_layers - 1)
        along_lat, along_lon = destination_point(*a, brg, dist * frac)
        if i == 0 or i == n_layers - 1:
            layers.append([(along_lat, along_lon)])
        else:
            row = []
            for off in offsets_km:
                lat_o, lon_o = destination_point(along_lat, along_lon, (brg + 90) % 360, off)
                row.append((lat_o, lon_o))
            layers.append(row)
    return layers, dist, brg


# --------------------------------------------------------------------------
# 6. EDGE COST -- energy (Wh) to fly between two lattice nodes given live wind
# --------------------------------------------------------------------------
#
# UPDATE: ground speed used to be a single fixed constant for every edge.
# That made the optimizer's only lever "fly the required airspeed a fixed
# distance in a fixed time" -- a tailwind could lower propulsion power, but
# every edge still cost exactly dist/cruise_speed seconds no matter what, so
# a detour's extra distance always cost extra time (and therefore extra
# energy) that a power cut alone often can't recoup. That's why a strong
# synthetic tailwind lane could fail to move the optimizer off the straight
# line: the lattice's jump-and-back geometry pays a fixed distance penalty
# that fixed-speed edges can never pay back with time savings, only power
# savings.
#
# Now ground speed is a per-edge decision variable. For a given edge and
# wind, flying faster or slower over the ground changes BOTH the required
# airspeed (and hence propulsion power) AND the transit time -- so a
# tailwind edge can be worth taking because it's faster, cheaper in power,
# or some mix of both. Concretely: for a fixed heading (the edge's bearing),
# ground_velocity = airspeed_velocity + wind_velocity is a vector triangle;
# as ground speed g varies, required airspeed traces a convex curve with its
# minimum where the along-track ground speed matches the wind's along-track
# component (at that point you only need airspeed to cancel the crosswind).
# We search g over the range that keeps required airspeed within the
# airframe's max_speed_mps, and pick whichever g minimizes Wh for that edge
# -- a coarse grid pass followed by a short local refinement.

GROUND_SPEED_FLOOR_MPS = 0.5  # ASSUMED: floor for the per-edge speed search,
                              # keeps near-hover edges from blowing up transit time


def _airspeed_mag(bearing_deg: float, ground_speed_mps: float, wind_e: float, wind_n: float) -> float:
    """|airspeed| needed to make good `ground_speed_mps` along `bearing_deg`
    through a wind of (wind_e, wind_n): ground = airspeed + wind => airspeed = ground - wind."""
    ground_e, ground_n = bearing_to_vec(bearing_deg, ground_speed_mps)
    return math.hypot(ground_e - wind_e, ground_n - wind_n)


def _feasible_ground_speed_range(bearing_deg, wind_e, wind_n, v_air_max, g_floor=GROUND_SPEED_FLOOR_MPS):
    """
    Range of forward ground speeds g (>= g_floor) along `bearing_deg` that
    keep the required airspeed at or below v_air_max, given wind (wind_e, wind_n).

    Required airspeed as a function of g is |g*u_hat - wind|, a convex
    function of g minimized at g = u_hat . wind (the wind's along-track
    component), where the residual is just the crosswind component. Solving
    |g*u_hat - wind| = v_air_max for g gives the two endpoints of the
    feasible interval. Returns None if even the best-case g (fully
    cancelling the crosswind) would still need more airspeed than v_air_max
    -- i.e. the crosswind alone exceeds what the airframe can fly.
    """
    u_e, u_n = math.sin(math.radians(bearing_deg)), math.cos(math.radians(bearing_deg))
    along_track = u_e * wind_e + u_n * wind_n            # wind component along track (tailwind > 0)
    wind_mag_sq = wind_e ** 2 + wind_n ** 2
    crosswind_sq = max(wind_mag_sq - along_track ** 2, 0.0)  # clamp tiny fp negatives
    disc = v_air_max ** 2 - crosswind_sq
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    g_lo, g_hi = max(along_track - sqrt_disc, g_floor), along_track + sqrt_disc
    if g_lo > g_hi:
        return None
    return g_lo, g_hi


def edge_energy_wh(node_from, node_to, atmos_from, atmos_to, d: DroneSpecs, n_grid: int = 40):
    """Minimum-energy (Wh) way to fly from node_from to node_to given live wind,
    choosing the ground speed per edge rather than assuming a fixed one.
    Returns (energy_wh, time_s, ground_speed_mps)."""
    dist_km = haversine_km(*node_from, *node_to)
    if dist_km == 0:
        return 0.0, 0.0, 0.0
    dist_m = dist_km * 1000
    brg = initial_bearing_deg(*node_from, *node_to)

    # average the wind at the two endpoints for this edge
    ws = (atmos_from["wind_speed"] + atmos_to["wind_speed"]) / 2
    wind_to_bearing = ((atmos_from["wind_dir_from_deg"] + atmos_to["wind_dir_from_deg"]) / 2 + 180) % 360
    wind_e, wind_n = bearing_to_vec(wind_to_bearing, ws)

    rho = (
        air_density_kgm3(atmos_from["temp_c"], atmos_from["surface_pressure_hpa"], CRUISE_AGL_M)
        + air_density_kgm3(atmos_to["temp_c"], atmos_to["surface_pressure_hpa"], CRUISE_AGL_M)
    ) / 2

    def energy_and_time(g):
        v_air = _airspeed_mag(brg, g, wind_e, wind_n)
        power_w = propulsion_power_w(v_air, rho, d) + d.avionics_payload_power_w
        time_s = dist_m / g
        return power_w * time_s / 3600, time_s

    feasible = _feasible_ground_speed_range(brg, wind_e, wind_n, d.max_speed_mps)
    if feasible is None:
        # Crosswind alone would need more airspeed than the airframe has --
        # physically unflyable on this exact bearing at any forward speed.
        # Fall back to the max airspeed the drone can produce rather than
        # silently pretending the edge is free; this edge will almost never
        # win a min-energy search anyway.
        g_lo = g_hi = d.max_speed_mps
    else:
        g_lo, g_hi = feasible

    # Coarse grid over the feasible range...
    if g_hi <= g_lo:
        best_g = g_lo
    else:
        step = (g_hi - g_lo) / (n_grid - 1)
        best_g, best_e = g_lo, math.inf
        for i in range(n_grid):
            g = g_lo + step * i
            e_wh, _ = energy_and_time(g)
            if e_wh < best_e:
                best_e, best_g = e_wh, g
        # ...then a short local golden-section-ish refine around the winner,
        # since the grid alone can be off by up to half a grid step.
        lo, hi = max(g_lo, best_g - step), min(g_hi, best_g + step)
        for _ in range(20):
            if hi - lo < 1e-4:
                break
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            e1, _ = energy_and_time(m1)
            e2, _ = energy_and_time(m2)
            if e1 < e2:
                hi = m2
            else:
                lo = m1
        best_g = (lo + hi) / 2

    energy_wh, time_s = energy_and_time(best_g)
    return energy_wh, time_s, best_g


# --------------------------------------------------------------------------
# 7. DIJKSTRA over the lattice
# --------------------------------------------------------------------------

def solve_min_energy_route(layers, atmos_lookup, d: DroneSpecs):
    # dist[(layer, idx)] = (cumulative_energy_wh, cumulative_time_s, path)
    start = (0, 0)
    dist = {start: (0.0, 0.0, [layers[0][0]])}
    pq = [(0.0, start)]
    while pq:
        cost, (li, idx) = heapq.heappop(pq)
        if cost > dist[(li, idx)][0]:
            continue
        if li == len(layers) - 1:
            continue
        node = layers[li][idx]
        for j, nxt in enumerate(layers[li + 1]):
            e_wh, t_s, _g = edge_energy_wh(node, nxt, atmos_lookup[node], atmos_lookup[nxt], d)
            new_cost = cost + e_wh
            key = (li + 1, j)
            if key not in dist or new_cost < dist[key][0]:
                cur_e, cur_t, cur_path = dist[(li, idx)]
                dist[key] = (cur_e + e_wh, cur_t + t_s, cur_path + [nxt])
                heapq.heappush(pq, (dist[key][0], key))
    goal = (len(layers) - 1, 0)
    return dist[goal]  # (energy_wh, time_s, path)


def straight_route_energy(layers, atmos_lookup, d: DroneSpecs):
    """Energy/time for the direct path: always the zero-offset node in each layer."""
    center = {i: (len(row) // 2) for i, row in enumerate(layers)}
    total_e, total_t = 0.0, 0.0
    for i in range(len(layers) - 1):
        n1 = layers[i][center[i]]
        n2 = layers[i + 1][center[i + 1]]
        e, t, _g = edge_energy_wh(n1, n2, atmos_lookup[n1], atmos_lookup[n2], d)
        total_e += e
        total_t += t
    return total_e, total_t


# --------------------------------------------------------------------------
# 8. DEMO
# --------------------------------------------------------------------------

def main():
    point_a = (37.7749, -122.4194)  # example: San Francisco, CA
    point_b = (38.8715, -123.2730)  # example: Berkeley, CA  (~15.7 km away)

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

    straight_dist_km = dist_km  # straight route follows the direct track by construction
    opt_dist_km = path_length_km(opt_path)

    print("\n--- Straight-line route ---")
    print(f"Energy: {straight_e:.1f} Wh   Time: {straight_t/60:.1f} min   "
          f"Avg ground speed: {straight_dist_km*1000/straight_t:.1f} m/s")

    print("\n--- Wind-optimized route ---")
    print(f"Energy: {opt_e:.1f} Wh   Time: {opt_t/60:.1f} min   "
          f"Avg ground speed: {opt_dist_km*1000/opt_t:.1f} m/s "
          f"({opt_dist_km:.1f} km flown, vs {straight_dist_km:.1f} km direct)")
    print(f"Waypoints: {[(round(la,4), round(lo,4)) for la, lo in opt_path]}")

    saved_pct = 100 * (straight_e - opt_e) / straight_e if straight_e else 0
    print(f"\nEnergy saved by routing with the wind: {saved_pct:.1f}%")
    print(f"Usable battery budget: {DRONE.usable_energy_wh:.0f} Wh  "
          f"(straight route uses {100*straight_e/DRONE.usable_energy_wh:.1f}% of it, "
          f"optimized uses {100*opt_e/DRONE.usable_energy_wh:.1f}%)")


if __name__ == "__main__":
    main()