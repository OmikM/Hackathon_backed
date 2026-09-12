# C++ Wind Routing API

This is a Python/FastAPI wrapper around `algorithm.cpp`.

## Files

```text
algorithm.cpp          # your actual A* routing algorithm
algorithm_runner.cpp   # tiny stdin/stdout bridge to algorithm.cpp
main_cpp.py            # FastAPI + Open-Meteo + C++ process launcher
```

## What happens on `/api/v1/plan-route`

1. FastAPI receives the route in the same local metre coordinate system as `main.py`.
2. Python creates a rectangular grid around the route.
3. Python converts each grid cell to latitude/longitude.
4. Python fetches current 10 m wind from Open-Meteo.
5. Meteorological wind direction is converted into the `u/east` and `v/north`
   vector expected by `algorithm.cpp`.
6. Python starts the C++ executable.
7. The entire wind grid is sent to C++ over stdin.
8. `algorithm.cpp` runs its A* search.
9. C++ returns grid-cell waypoints and total travel time.
10. Python converts those cells back to local `x/y` points and returns JSON.

## Important difference from the old `main.py`

Your C++ algorithm is a **time optimizer**, not an energy optimizer.

It uses:

```cpp
const double V_AIR = 15.0;
```

and calculates the physically achievable ground speed for each edge.

Therefore `total_time_seconds` comes directly from the C++ algorithm.

The `total_energy_wh` returned by this wrapper is only a simple compatibility
estimate:

```text
energy = cruise_power * time
```

It is NOT calculated by `algorithm.cpp`.

## Install

```bash
pip install fastapi uvicorn requests pydantic
```

You also need GCC/MinGW:

```bash
g++ --version
```

## Run

Put all three files in the same directory:

```text
algorithm.cpp
algorithm_runner.cpp
main_cpp.py
```

Then:

```bash
python main_cpp.py
```

The first request automatically compiles:

```text
algorithm_runner.cpp + algorithm.cpp
```

into `algorithm_runner` / `algorithm_runner.exe`.

## Example request

```bash
curl -X POST http://localhost:8000/api/v1/plan-route \
  -H "Content-Type: application/json" \
  -d '{
    "origin_lat": 54.7,
    "origin_lon": 18.5,
    "start_point": {
      "x": 0,
      "y": 0
    },
    "goal_point": {
      "x": 4000,
      "y": 2500
    },
    "mode": "speed",
    "step_size": 80,
    "grid_margin": 4
  }'
```

## Notes

Open-Meteo's wind direction describes the direction the wind is coming FROM.
The C++ algorithm needs the direction the air is actually moving TOWARD.

The wrapper therefore performs:

```text
u = -speed * sin(direction)
v = -speed * cos(direction)
```

which produces:

```text
u > 0  -> air moving east
u < 0  -> air moving west
v > 0  -> air moving north
v < 0  -> air moving south
```

The C++ code uses the wind of the cell being entered, exactly as your
`algorithm.cpp` specifies.
