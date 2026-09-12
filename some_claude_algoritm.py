import math
import heapq
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.optimize import minimize_scalar

@dataclass
class DroneSpecs:
    mass_kg: float = 6.3                 
    max_speed_mps: float = 23.0          
    avionics_payload_power_w: float = 30.0 
    P0: float = 79.86; Pi: float = 88.63; U_tip: float = 120.0  
    v0: float = 4.03; d0: float = 0.6; s: float = 0.05; A: float = 0.503      

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    y = math.sin(dl)*math.cos(p2)
    x = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return dist, (math.degrees(math.atan2(y, x)) + 360) % 360

def calc_power(va, specs):
    p_prof = specs.P0 * (1 + 3 * (va / specs.U_tip)**2)
    ratio = va / (2 * specs.v0)
    p_ind = specs.Pi * math.sqrt(max(0, math.sqrt(1 + ratio**4) - ratio**2))
    p_par = 0.5 * specs.d0 * 1.225 * specs.s * specs.A * (va**3)
    return p_prof + p_ind + p_par + specs.avionics_payload_power_w

def simulate_segment(vg, dist, bearing, ws, wd, specs):
    w_to = math.radians((wd + 180) % 360)
    path_rad = math.radians(bearing)
    diff = w_to - path_rad
    w_par, w_per = ws * math.cos(diff), ws * math.sin(diff)
    
    va = math.sqrt((vg - w_par)**2 + w_per**2)
    if va > specs.max_speed_mps or abs(w_per) > specs.max_speed_mps: 
        return float('inf'), float('inf')
        
    time_s = dist / vg
    energy_wh = (calc_power(va, specs) * time_s) / 3600.0
    return energy_wh, time_s

def get_optimal_edge(dist, bearing, ws, wd, specs):
    """Zwraca minimalną energię i odpowiadający jej czas lotu."""
    res = minimize_scalar(
        lambda vg: simulate_segment(vg, dist, bearing, ws, wd, specs)[0],
        bounds=(2.0, specs.max_speed_mps), method='bounded'
    )
    if res.success:
        return simulate_segment(res.x, dist, bearing, ws, wd, specs)
    return float('inf'), float('inf')

def create_grid(lat_s, lon_s, lat_e, lon_e, grid_size=8):
    lats = np.linspace(lat_s, lat_e, grid_size)
    lons = np.linspace(lon_s, lon_e, grid_size)
    return lats, lons

def mock_wind_field(lat, lon):
    """Symulacja: Silny wiatr czołowy (zachodni) w centrum, słabszy na obrzeżach."""
    center_lat, center_lon = 54.43, 18.58 
    dist_from_center, _ = haversine(lat, lon, center_lat, center_lon)
    wind_speed = max(2.0, 16.0 - (dist_from_center / 1500.0)) 
    return wind_speed, 270.0 # Wiatr wiejący z Zachodu

def find_optimal_path(lat_s, lon_s, lat_e, lon_e, specs):
    grid_size = 10
    lats, lons = create_grid(lat_s, lon_s, lat_e, lon_e, grid_size)
    
    start_node, target_node = (0, 0), (grid_size-1, grid_size-1)
    
    graph = {}
    for i in range(grid_size):
        for j in range(grid_size):
            node = (i, j)
            graph[node] = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    dist, bear = haversine(lats[i], lons[j], lats[ni], lons[nj])
                    ws, wd = mock_wind_field((lats[i]+lats[ni])/2, (lons[j]+lons[nj])/2)
                    energy_cost, time_cost = get_optimal_edge(dist, bear, ws, wd, specs)
                    graph[node].append(((ni, nj), energy_cost, time_cost))

    pq = [(0.0, start_node)]
    costs = {start_node: 0.0}
    times = {start_node: 0.0}
    parents = {start_node: None}

    while pq:
        curr_cost, curr_node = heapq.heappop(pq)
        if curr_node == target_node: break
        if curr_cost > costs.get(curr_node, float('inf')): continue
            
        for neighbor, edge_energy, edge_time in graph[curr_node]:
            new_cost = curr_cost + edge_energy
            if new_cost < costs.get(neighbor, float('inf')):
                costs[neighbor] = new_cost
                times[neighbor] = times[curr_node] + edge_time
                parents[neighbor] = curr_node
                heapq.heappush(pq, (new_cost, neighbor))

    path = []
    curr = target_node
    while curr is not None:
        path.append((lats[curr[0]], lons[curr[1]]))
        curr = parents[curr]
    path.reverse()
    
    return path, costs[target_node], times[target_node]

if __name__ == "__main__":
    specs = DroneSpecs()
    lat_s, lon_s = 54.35, 18.64 # Gdańsk
    lat_e, lon_e = 54.51, 18.53 # Gdynia
    
    # 1. Trasa prosta (Naiwna, stała prędkość 12 m/s)
    dist_straight, bear_straight = haversine(lat_s, lon_s, lat_e, lon_e)
    ws_str, wd_str = mock_wind_field((lat_s+lat_e)/2, (lon_s+lon_e)/2)
    energy_str_naive, time_str_naive = simulate_segment(12.0, dist_straight, bear_straight, ws_str, wd_str, specs)
    
    # 2. Trasa prosta (Prędkość optymalizowana)
    energy_str_opt, time_str_opt = get_optimal_edge(dist_straight, bear_straight, ws_str, wd_str, specs)
    
    # 3. Trasa przestrzenna (Dijkstra - omijanie silnego wiatru)
    path, path_energy, path_time = find_optimal_path(lat_s, lon_s, lat_e, lon_e, specs)
    
    print("=== PORÓWNANIE METOD LOTU ===")
    print(f"Dystans w linii prostej: {dist_straight/1000:.2f} km")
    
    print("\n1. Linia prosta (Stała prędkość 12 m/s):")
    print(f"   Energia: {energy_str_naive:.2f} Wh")
    print(f"   Czas:    {time_str_naive/60:.2f} min")
    
    print("\n2. Linia prosta (Prędkość zoptymalizowana do wiatru):")
    print(f"   Energia: {energy_str_opt:.2f} Wh")
    print(f"   Czas:    {time_str_opt/60:.2f} min")
    
    print("\n3. Zoptymalizowana ścieżka Dijkstry (Omijanie stref wysokiego oporu):")
    print(f"   Energia: {path_energy:.2f} Wh")
    print(f"   Czas:    {path_time/60:.2f} min")
    
    print("\n=== PODSUMOWANIE ZYSKÓW (Względem prostej zoptymalizowanej) ===")
    diff_energy = ((energy_str_opt - path_energy) / energy_str_opt) * 100
    diff_time = ((time_str_opt - path_time) / time_str_opt) * 100
    
    print(f"Zysk energetyczny: {diff_energy:+.1f}%")
    print(f"Różnica w czasie:  {diff_time:+.1f}% ({'szybciej' if diff_time > 0 else 'wolniej'})")