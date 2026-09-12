#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
using namespace std;

using pii = pair<int, int>;
using pdi = pair<double, pii>;

const double INF = 1e9;
const double SQRT2 = sqrt(2.0);

// --------------------------------------------------------------------------
// DRONE / POWER MODEL -- same DJI Matrice 300 RTK figures and Zeng, Xu &
// Zhang (2019) power model used by playground.py, so this C++ solver and
// the Python lattice planner price energy the same way.
// --------------------------------------------------------------------------
const double V_AIR_MAX = 23.0;      // m/s, matches DRONE.max_speed_mps
const double AVIONICS_POWER_W = 30.0; // hotel load, matches DRONE.avionics_payload_power_w
const double RHO = 1.225;           // kg/m^3, sea-level standard air density (no elevation
                                     // grid available here, same simplification main_cpp.py
                                     // already makes for its fixed-speed estimate)
const double P0 = 79.86;     // W, blade profile power at hover
const double PI_POWER = 88.63; // W, induced power at hover (named to avoid clashing with math PI)
const double U_TIP = 120.0;  // m/s, rotor blade tip speed
const double V0_IND = 4.03;  // m/s, mean rotor induced velocity at hover
const double D0 = 0.6;       // fuselage drag ratio
const double SOLIDITY = 0.05; // rotor solidity
const double ROTOR_AREA = 0.503; // m^2, rotor disc area

// How the per-edge ground-speed search behaves.
const double GROUND_SPEED_FLOOR = 0.5; // m/s -- keeps near-hover edges from
                                        // blowing up transit time, same floor
                                        // playground.py uses
const int N_GRID = 40;      // coarse scan points across the feasible speed range,
                             // matches playground.py's edge_energy_wh(n_grid=40)
const int REFINE_ITERS = 20; // golden-section-ish refine passes around the winner

// Przesunięcia siatki: 4 boki + 4 przekątne
const int dr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
const int dc[] = {0, 0, -1, 1, -1, 1, -1, 1};

// Struktura przechowująca wektor wiatru w danej komórce (z API)
struct Wind {
    double u, v; // u = wschód-zachód, v = północ-południe (m/s)
};

struct EdgeCost {
    double energy_wh;
    double time_s;
    double ground_speed_mps;
    bool feasible;
};

// --------------------------------------------------------------------------
// Propulsion power (W) for a given airspeed, level flight -- Zeng et al.
// 2019 model. This is the SAME curve algorithm.cpp implicitly assumed was
// always evaluated at V_AIR_MAX; here it's evaluated at whatever airspeed
// the chosen ground speed for this edge actually requires.
// --------------------------------------------------------------------------
double propulsion_power_w(double v_air) {
    double blade_profile = P0 * (1.0 + 3.0 * v_air * v_air / (U_TIP * U_TIP));
    double induced = PI_POWER * sqrt(
        sqrt(1.0 + (v_air * v_air * v_air * v_air) / (4.0 * V0_IND * V0_IND * V0_IND * V0_IND))
        - (v_air * v_air) / (2.0 * V0_IND * V0_IND)
    );
    double parasite = 0.5 * D0 * RHO * SOLIDITY * ROTOR_AREA * v_air * v_air * v_air;
    return blade_profile + induced + parasite;
}

// Global minimum of propulsion_power_w over the achievable airspeed range.
// Used once, at startup, purely to build an admissible A* heuristic (see
// heuristic_wh below) -- NOT used as a per-edge cost.
double min_achievable_power_w() {
    double best = INF;
    const int SCAN = 2000;
    for (int i = 0; i <= SCAN; i++) {
        double v = V_AIR_MAX * i / SCAN;
        best = min(best, propulsion_power_w(v) + AVIONICS_POWER_W);
    }
    return best;
}

// --------------------------------------------------------------------------
// Per-edge ground-speed optimizer -- the core change from algorithm.cpp.
//
// algorithm.cpp always assumed airspeed == V_AIR_MAX and let ground speed
// fall out of the wind triangle (fastest-possible flight). That minimizes
// TIME, not energy: it always spends the most power the airframe can draw.
//
// Here, ground speed g along the edge's heading is a free variable. For a
// fixed heading, required airspeed |g*u_hat - wind| traces a convex curve
// in g; power(airspeed) is not linear, and energy = power * (L / g), so
// the energy-minimizing g is generally an interior point, not V_AIR_MAX --
// slower can win when a strong tailwind means a modest ground speed still
// needs little airspeed AND arrives quickly, or when overcoming a
// headwind costs so much power that flying at min disc-loading speed
// while accepting a longer time is cheaper. We search the whole feasible
// range instead of assuming an answer.
// --------------------------------------------------------------------------
EdgeCost calculate_edge_energy(int r1, int c1, int r2, int c2, const Wind& cell_wind, double cell_size_meters) {
    bool is_diagonal = (r1 != r2 && c1 != c2);
    double L = is_diagonal ? cell_size_meters * SQRT2 : cell_size_meters;

    double dx = (c2 - c1);
    double dy = (r1 - r2); // odwrócona oś Y w macierzy
    double len = sqrt(dx * dx + dy * dy);
    if (len == 0) return {0.0, 0.0, 0.0, true};

    double ux = dx / len;
    double uy = dy / len;

    double Wx = cell_wind.u;
    double Wy = cell_wind.v;

    double W_para = ux * Wx + uy * Wy;              // along-track wind component (tailwind > 0)
    double W_sq = Wx * Wx + Wy * Wy;
    double crosswind_sq = max(0.0, W_sq - W_para * W_para);

    // Feasible range of ground speed g (>= floor) that keeps required
    // airspeed at or below V_AIR_MAX: |g*u_hat - wind| <= V_AIR_MAX.
    double disc = V_AIR_MAX * V_AIR_MAX - crosswind_sq;
    if (disc < 0) return {0.0, 0.0, 0.0, false}; // crosswind alone exceeds the airframe

    double sqrt_disc = sqrt(disc);
    double g_lo = max(W_para - sqrt_disc, GROUND_SPEED_FLOOR);
    double g_hi = W_para + sqrt_disc;
    if (g_lo > g_hi) return {0.0, 0.0, 0.0, false};

    auto energy_and_time = [&](double g) -> pair<double, double> {
        double v_air_sq = g * g - 2.0 * g * W_para + W_sq;
        double v_air = sqrt(max(0.0, v_air_sq));
        double power_w = propulsion_power_w(v_air) + AVIONICS_POWER_W;
        double time_s = L / g;
        return {power_w * time_s / 3600.0, time_s};
    };

    double best_g;
    if (g_hi <= g_lo) {
        best_g = g_lo;
    } else {
        double step = (g_hi - g_lo) / (N_GRID - 1);
        double best_e = INF;
        best_g = g_lo;
        for (int i = 0; i < N_GRID; i++) {
            double g = g_lo + step * i;
            double e = energy_and_time(g).first;
            if (e < best_e) { best_e = e; best_g = g; }
        }
        double lo = max(g_lo, best_g - step);
        double hi = min(g_hi, best_g + step);
        for (int it = 0; it < REFINE_ITERS; it++) {
            if (hi - lo < 1e-4) break;
            double m1 = lo + (hi - lo) / 3.0;
            double m2 = hi - (hi - lo) / 3.0;
            if (energy_and_time(m1).first < energy_and_time(m2).first) hi = m2; else lo = m1;
        }
        best_g = (lo + hi) / 2.0;
    }

    auto [energy_wh, time_s] = energy_and_time(best_g);
    return {energy_wh, time_s, best_g, true};
}

// Admissible A* heuristic in Wh: no edge can need less than MIN_POWER_W of
// power, and no edge can cover ground faster than V_AIR_MAX plus the
// strongest tailwind seen anywhere in this wind field -- both are valid
// lower bounds individually, so their product lower-bounds the true
// remaining energy without ever overestimating it.
inline double heuristic_wh(int r, int c, int gr, int gc, double cell_size_meters,
                            double min_power_w, double max_ground_speed_mps) {
    double dist_m = sqrt(pow(r - gr, 2) + pow(c - gc, 2)) * cell_size_meters;
    double min_time_s = dist_m / max_ground_speed_mps;
    return min_power_w * min_time_s / 3600.0;
}

struct AStarResult {
    vector<pii> path;
    double total_energy_wh;
    double total_time_s;
    vector<double> edge_speeds_mps;
};

AStarResult a_star_energy(int R, int C, pii start, pii goal,
                           const vector<vector<Wind>>& wind_grid,
                           double cell_size_meters)
{
    double min_power_w = min_achievable_power_w();
    double max_wind_mag = 0.0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            max_wind_mag = max(max_wind_mag, hypot(wind_grid[r][c].u, wind_grid[r][c].v));
    double max_ground_speed_mps = V_AIR_MAX + max_wind_mag;

    vector<vector<double>> g(R, vector<double>(C, INF));      // cumulative energy, Wh
    vector<vector<double>> g_time(R, vector<double>(C, INF)); // cumulative time, s (for reporting)
    vector<vector<pii>> parent(R, vector<pii>(C, {-1, -1}));

    priority_queue<pdi, vector<pdi>, greater<pdi>> pq;

    g[start.first][start.second] = 0.0;
    g_time[start.first][start.second] = 0.0;
    pq.push({heuristic_wh(start.first, start.second, goal.first, goal.second, cell_size_meters,
                           min_power_w, max_ground_speed_mps), start});

    while (!pq.empty()) {
        auto [f, u] = pq.top();
        pq.pop();

        auto [r, c] = u;

        double h_here = heuristic_wh(r, c, goal.first, goal.second, cell_size_meters,
                                      min_power_w, max_ground_speed_mps);
        if (f > g[r][c] + h_here + 1e-9) continue;
        if (u == goal) break;

        for (int i = 0; i < 8; i++) {
            int nr = r + dr[i];
            int nc = c + dc[i];

            if (nr >= 0 && nr < R && nc >= 0 && nc < C) {
                EdgeCost ec = calculate_edge_energy(r, c, nr, nc, wind_grid[nr][nc], cell_size_meters);
                if (!ec.feasible) continue;

                double new_g = g[r][c] + ec.energy_wh;

                if (new_g < g[nr][nc]) {
                    g[nr][nc] = new_g;
                    g_time[nr][nc] = g_time[r][c] + ec.time_s;
                    parent[nr][nc] = u;
                    double h_next = heuristic_wh(nr, nc, goal.first, goal.second, cell_size_meters,
                                                  min_power_w, max_ground_speed_mps);
                    pq.push({new_g + h_next, {nr, nc}});
                }
            }
        }
    }

    if (g[goal.first][goal.second] == INF) return {{}, INF, INF, {}};

    vector<pii> path;
    for (pii curr = goal; curr.first != -1; curr = parent[curr.first][curr.second]) {
        path.push_back(curr);
    }
    reverse(path.begin(), path.end());

    // Recover the per-edge ground speed actually chosen along the winning
    // path (re-solving each edge is cheap, and avoids threading extra
    // state through the Dijkstra relaxation above).
    vector<double> edge_speeds;
    for (size_t i = 0; i + 1 < path.size(); i++) {
        auto [r1, c1] = path[i];
        auto [r2, c2] = path[i + 1];
        EdgeCost ec = calculate_edge_energy(r1, c1, r2, c2, wind_grid[r2][c2], cell_size_meters);
        edge_speeds.push_back(ec.ground_speed_mps);
    }

    return {path, g[goal.first][goal.second], g_time[goal.first][goal.second], edge_speeds};
}

// Append this to the bottom of your algorithm_energy.cpp
int main(int argc, char* argv[]) {
    if (argc < 7) return 1;
    int R = atoi(argv[1]);
    int C = atoi(argv[2]);
    pii start = {atoi(argv[3]), atoi(argv[4])};
    pii goal = {atoi(argv[5]), atoi(argv[6])};
    double cell_size = (argc > 7) ? atof(argv[7]) : 80.0;

    vector<vector<Wind>> wind_grid(R, vector<Wind>(C));
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            if (!(cin >> wind_grid[r][c].u >> wind_grid[r][c].v)) break;
        }
    }

    AStarResult result = a_star_energy(R, C, start, goal, wind_grid, cell_size);

    cout << "{\"total_energy_wh\":" << result.total_energy_wh
         << ",\"total_time\":" << result.total_time_s
         << ",\"path\":[";
    for (size_t i = 0; i < result.path.size(); i++) {
        cout << "[" << result.path[i].first << "," << result.path[i].second << "]"
             << (i + 1 < result.path.size() ? "," : "");
    }
    cout << "],\"edge_speeds_mps\":[";
    for (size_t i = 0; i < result.edge_speeds_mps.size(); i++) {
        cout << result.edge_speeds_mps[i] << (i + 1 < result.edge_speeds_mps.size() ? "," : "");
    }
    cout << "]}" << endl;
    return 0;
}