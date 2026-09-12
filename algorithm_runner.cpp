#include <bits/stdc++.h>
using namespace std;

// The actual routing algorithm lives in algorithm.cpp.
#include "algorithm.cpp"

/*
Input:
  R C cell_size start_r start_c goal_r goal_c
  followed by R*C lines:
    u v

Output:
  TIME <seconds>
  PATH <number_of_points>
  <row> <col>
  ...
*/

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int R, C, sr, sc, gr, gc;
    double cell_size;

    if (!(cin >> R >> C >> cell_size >> sr >> sc >> gr >> gc)) {
        cerr << "Invalid header input\n";
        return 2;
    }

    if (R <= 0 || C <= 0 ||
        sr < 0 || sr >= R || sc < 0 || sc >= C ||
        gr < 0 || gr >= R || gc < 0 || gc >= C) {
        cerr << "Invalid grid/start/goal\n";
        return 3;
    }

    vector<vector<Wind>> wind_grid(R, vector<Wind>(C));

    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            if (!(cin >> wind_grid[r][c].u >> wind_grid[r][c].v)) {
                cerr << "Missing wind data at " << r << "," << c << "\n";
                return 4;
            }
        }
    }

    auto [path, total_time] = a_star_wind(
        R, C,
        {sr, sc},
        {gr, gc},
        wind_grid,
        cell_size
    );

    if (path.empty()) {
        cout << "TIME INF\n";
        cout << "PATH 0\n";
        return 0;
    }

    cout << setprecision(12);
    cout << "TIME " << total_time << "\n";
    cout << "PATH " << path.size() << "\n";

    for (auto [r, c] : path) {
        cout << r << ' ' << c << "\n";
    }

    return 0;
}
