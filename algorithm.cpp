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
const double V_AIR = 23.0; // Matches DRONE.max_speed_mps in playground.py.

// Przesunięcia siatki: 4 boki + 4 przekątne
const int dr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
const int dc[] = {0, 0, -1, 1, -1, 1, -1, 1};

// Struktura przechowująca wektor wiatru w danej komórce (z API)
struct Wind {
    double u, v; // u = wschód-zachód, v = północ-południe (m/s)
};

// Funkcja fizyczna licząca czas przejścia krawędzi (w sekundach)
double calculate_edge_time(int r1, int c1, int r2, int c2, const Wind& cell_wind, double cell_size_meters) {
    // 1. Długość krawędzi w metrach (boki vs przekątne)
    bool is_diagonal = (r1 != r2 && c1 != c2);
    double L = is_diagonal ? cell_size_meters * SQRT2 : cell_size_meters;

    // 2. Znormalizowany wektor kierunku ruchu (u_x, u_y)
    double dx = (c2 - c1);
    double dy = (r1 - r2); // odwrócona oś Y w macierzy
    double len = sqrt(dx * dx + dy * dy);
    if (len == 0) return 0.0;
    
    double ux = dx / len;
    double uy = dy / len;

    // 3. Rozwiązanie trójkąta prędkości (Ground Speed)
    double Wx = cell_wind.u;
    double Wy = cell_wind.v;
    
    double W_para = ux * Wx + uy * Wy;
    double W_sq = Wx * Wx + Wy * Wy;
    
    double discriminant = (V_AIR * V_AIR) - (W_sq - (W_para * W_para));
    
    // Jeśli wiatr jest zbyt silny bocznie, dron nie utrzyma kursu
    if (discriminant < 0) return INF; 

    double V_gs = W_para + sqrt(discriminant);
    
    // Jeśli prędkość względem ziemi jest zerowa lub ujemna (lecimy pod huragan)
    if (V_gs <= 0) return INF;

    // Czas = Dystans / Prędkość
    return L / V_gs;
}

// --------------------------------------------------------------------------
// Straight-line reference baseline.
//
// Every route this solver produces should beat (or tie) just flying
// straight from start to goal through the same wind field. This builds
// that reference path -- the Bresenham line through the grid, using only
// the same 8-directional steps the A* search itself is allowed to take --
// and prices it with the exact same per-edge physics (calculate_edge_time),
// so the comparison is apples-to-apples rather than against some idealized
// zero-wind straight-line estimate.
// --------------------------------------------------------------------------
vector<pii> bresenham_line(pii start, pii goal) {
    vector<pii> line;
    int r0 = start.first, c0 = start.second;
    int r1 = goal.first, c1 = goal.second;
    int drow = abs(r1 - r0), dcol = abs(c1 - c0);
    int sr = (r0 < r1) ? 1 : -1;
    int sc = (c0 < c1) ? 1 : -1;
    int err = drow - dcol;
    int r = r0, c = c0;
    line.push_back({r, c});
    while (r != r1 || c != c1) {
        int e2 = 2 * err;
        if (e2 > -dcol) { err -= dcol; r += sr; }
        if (e2 < drow)  { err += drow; c += sc; }
        line.push_back({r, c});
    }
    return line;
}

struct StraightLineResult {
    vector<pii> path;
    double time_s;
    bool feasible;
};

StraightLineResult straight_line_reference(pii start, pii goal, const vector<vector<Wind>>& wind_grid, double cell_size_meters) {
    vector<pii> line = bresenham_line(start, goal);
    double total = 0.0;
    for (size_t i = 0; i + 1 < line.size(); i++) {
        auto [r1, c1] = line[i];
        auto [r2, c2] = line[i + 1];
        double t = calculate_edge_time(r1, c1, r2, c2, wind_grid[r2][c2], cell_size_meters);
        if (t == INF) return {line, INF, false}; // crosswind too strong somewhere on the direct bearing
        total += t;
    }
    return {line, total, true};
}

// Heurystyka czasowa (zakłada optymistyczny lot z maksymalną prędkością wiatru w plecy)
inline double h(int r, int c, int gr, int gc, double cell_size_meters) {
    double dist = sqrt(pow(r - gr, 2) + pow(c - gc, 2)) * cell_size_meters;
    // Minimalny teoretyczny czas (zakładając V_air + max możliwy wiatr pomocniczy, np. 20 m/s)
    return dist / (V_AIR + 20.0); 
}

pair<vector<pii>, double> a_star_wind(int R, int C, pii start, pii goal, 
                                      const vector<vector<Wind>>& wind_grid, 
                                      double cell_size_meters) 
{
    vector<vector<double>> g(R, vector<double>(C, INF));
    vector<vector<pii>> parent(R, vector<pii>(C, {-1, -1}));
    
    priority_queue<pdi, vector<pdi>, greater<pdi>> pq;

    g[start.first][start.second] = 0.0;
    pq.push({h(start.first, start.second, goal.first, goal.second, cell_size_meters), start});

    while (!pq.empty()) {
        auto [f, u] = pq.top();
        pq.pop();
        
        auto [r, c] = u;

        if (f > g[r][c] + h(r, c, goal.first, goal.second, cell_size_meters) + 1e-9) continue;
        if (u == goal) break;

        for (int i = 0; i < 8; i++) {
            int nr = r + dr[i];
            int nc = c + dc[i];

            if (nr >= 0 && nr < R && nc >= 0 && nc < C) {
                // Pobieramy wiatr z komórki, do której wchodzimy
                double edge_time = calculate_edge_time(r, c, nr, nc, wind_grid[nr][nc], cell_size_meters);
                
                if (edge_time == INF) continue; // Niedomknięty fizycznie wektor

                double new_g = g[r][c] + edge_time;
                
                if (new_g < g[nr][nc]) {
                    g[nr][nc] = new_g;
                    parent[nr][nc] = u;
                    pq.push({new_g + h(nr, nc, goal.first, goal.second, cell_size_meters), {nr, nc}});
                }
            }
        }
    }

    if (g[goal.first][goal.second] == INF) return {{}, INF};

    vector<pii> path;
    for (pii curr = goal; curr.first != -1; curr = parent[curr.first][curr.second]) {
        path.push_back(curr);
    }
    reverse(path.begin(), path.end());
    
    return {path, g[goal.first][goal.second]};
}

// Append this to the bottom of your algorithm.cpp
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

    auto [path, total_time] = a_star_wind(R, C, start, goal, wind_grid, cell_size);

    StraightLineResult straight = straight_line_reference(start, goal, wind_grid, cell_size);

    // Human-readable comparison on stderr -- shows up in whatever terminal
    // is running this binary (or in the server's console when invoked as a
    // subprocess by main_cpp.py) without corrupting the JSON on stdout.
    cerr << "--- Straight-line reference ---\n";
    if (straight.feasible) {
        double pct_faster = (straight.time_s > 0) ? 100.0 * (straight.time_s - total_time) / straight.time_s : 0.0;
        cerr << "Straight-line time:        " << straight.time_s << " s\n";
        cerr << "Wind-optimized time:       " << total_time << " s\n";
        cerr << "Improvement vs straight:   " << pct_faster << "%\n";
    } else {
        cerr << "Straight-line path is not flyable (crosswind exceeds drone airspeed on at least one segment).\n";
    }

    cout << "{\"total_time\":" << total_time
         << ",\"straight_line_time\":" << (straight.feasible ? straight.time_s : -1.0)
         << ",\"straight_line_feasible\":" << (straight.feasible ? "true" : "false")
         << ",\"straight_line_path\":[";
    for (size_t i = 0; i < straight.path.size(); i++) {
        cout << "[" << straight.path[i].first << "," << straight.path[i].second << "]" << (i + 1 < straight.path.size() ? "," : "");
    }
    cout << "],\"path\":[";
    for (size_t i = 0; i < path.size(); i++) {
        cout << "[" << path[i].first << "," << path[i].second << "]" << (i + 1 < path.size() ? "," : "");
    }
    cout << "]}" << endl;
    return 0;
}