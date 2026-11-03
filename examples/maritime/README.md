# Maritime Environmental Presets & Models (`examples/maritime`)

This directory contains representative maritime environmental presets, sea route definitions, and tidal current models consolidated for the Hakoniwa simulation ecosystem (supporting vessels, marine drones, and engineering viewers).

## Contents

- `sea_areas/`: Sea area geometry and environment definition files (JSON):
  - `uraga_suido.json`: Tokyo Bay Uraga Channel (URAGA SUIDO) Route definition based on the Maritime Traffic Safety Act Cabinet Order. Includes fairways, northbound/southbound traffic lanes, speed limits, and M2 semidiurnal tidal currents.
  - `bisan_seto.json`: Seto Inland Sea Bisan Seto Route definition, including strong tidal stream settings (peak > 2.5 kt).
- `sea_area.py`: Python geometric utility library to:
  - Convert navigational landmarks and bearing/distance definitions into local ENU polygons.
  - Calculate fairway centerline, lane boundaries (TSS: Traffic Separation Scheme), and widths.
  - Calculate dynamic 2D tidal current vectors $(v_e, v_n)$ at arbitrary position $(x, y)$ and time $t$.

## Shared Modules

### Wave Spectrum & Deterministic Wave Generator
- **Python**: `hakoniwa_envsim.wave.WaveSpectrumEngine`, `WaveParams` (`src/hakoniwa_envsim/wave/`)
- **C++**: `include/hakoniwa_envsim/environment/wave_spectrum.hpp`

Deterministic JONSWAP spectrum & directional Gerstner wave generation compatible across physical simulation (buoyancy) and rendering viewers (Unreal Engine 5 / Godot).

### Disturbance PDU Interface
- **Python**: `hakoniwa_envsim.asset.ShipIO` (`src/hakoniwa_envsim/asset/ship_io.py`)
- **PDU Contract**: Encodes/decodes 3D wind, tidal current, water depth $H$, tidal elevation, visibility, rain rate, and wave parameters into `hako_msgs/Disturbance`.
