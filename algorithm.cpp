#include <bits/stdc++.h>
using namespace std;

using pii = pair<int, int>;
using pdi = pair<double, pii>;

const double INF = 1e9;
const double SQRT2 = sqrt(2.0);
const double V_AIR = 15.0; // Stała prędkość własna drona (m/s)

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