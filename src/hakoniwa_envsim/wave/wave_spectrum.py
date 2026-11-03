"""wave_spectrum.py - JONSWAPスペクトル・多成分Gerstner波のPython決定論的生成モジュール。

C++版 (include/ship/environment/wave_spectrum.hpp) と同一アルゴリズム・同一PRNG (Xorshift32) を持ち、
テスト・検証・データ生成およびゲームエンジンシェーダ用定数バッファ出力を行う。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

PI = math.pi
GRAVITY = 9.80665


class DeterministicRNG:
    """Xorshift32 決定論的擬似乱数生成器 (C++版とビット完全互換)。"""

    def __init__(self, seed: int):
        self.state = 1 if seed == 0 else (seed & 0xFFFFFFFF)

    def next_uint32(self) -> int:
        x = self.state
        x = (x ^ ((x << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
        x = (x ^ (x >> 17)) & 0xFFFFFFFF
        x = (x ^ ((x << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
        self.state = x
        return x

    def next_double(self) -> float:
        return float(self.next_uint32()) / 4294967296.0


@dataclass
class WaveParams:
    hs_m: float = 2.5
    tp_s: float = 8.0
    wave_dir_rad: float = 0.0
    gamma: float = 3.3
    seed: int = 42
    water_depth_m: float = 100.0
    tide_level_m: float = 0.0
    num_components: int = 32


@dataclass
class WaveComponent:
    omega: float
    k: float
    kx: float
    ky: float
    amp: float
    phase: float
    dir_rad: float
    choppiness: float


class WaveSpectrumEngine:
    """JONSWAP / Gerstner 波面生成エンジン (Python実装)。"""

    def __init__(self, params: WaveParams = None):
        self.params = params or WaveParams()
        self.components: List[WaveComponent] = []
        if params is not None:
            self.initialize(self.params)

    def initialize(self, params: WaveParams) -> None:
        self.params = params
        self.components = []
        rng = DeterministicRNG(params.seed)

        omega_min = 0.20
        omega_max = 3.20
        d_omega = (omega_max - omega_min) / params.num_components
        omega_p = (2.0 * PI) / max(1.0, params.tp_s)
        target_m0 = (params.hs_m / 4.0) ** 2

        omegas = []
        s_raw = []
        integral_s = 0.0

        for i in range(params.num_components):
            w = omega_min + (i + 0.5) * d_omega
            omegas.append(w)

            sigma = 0.07 if w <= omega_p else 0.09
            r = math.exp(-((w - omega_p) ** 2) / (2.0 * (sigma * omega_p) ** 2))
            pm = (w ** -5.0) * math.exp(-1.25 * ((omega_p / w) ** 4.0))
            s_val = pm * (max(1.0, params.gamma) ** r)

            s_raw.append(s_val)
            integral_s += s_val * d_omega

        alpha_norm = (target_m0 / integral_s) if integral_s > 1e-9 else 1.0

        total_ka = 0.0
        for i in range(params.num_components):
            w = omegas[i]
            s_norm = s_raw[i] * alpha_norm
            amp = math.sqrt(2.0 * s_norm * d_omega)

            k = self.solve_dispersion_relation(w, params.water_depth_m)
            phase = rng.next_double() * 2.0 * PI
            spread_offset = (rng.next_double() - 0.5) * (PI / 2.0)
            dir_rad = params.wave_dir_rad + spread_offset

            kx = k * math.cos(dir_rad)
            ky = k * math.sin(dir_rad)

            total_ka += k * amp
            self.components.append(
                WaveComponent(
                    omega=w, k=k, kx=kx, ky=ky, amp=amp, phase=phase, dir_rad=dir_rad, choppiness=0.0
                )
            )

        max_choppiness = 0.85
        for c in self.components:
            if total_ka > 1e-6:
                c.choppiness = min(max_choppiness, max_choppiness / total_ka)
            else:
                c.choppiness = 0.0

    def get_elevation(self, x: float, y: float, t: float) -> float:
        """任意位置 (x, y) および時刻 t における波面標高 eta(x, y, t) [m]。"""
        eta = self.params.tide_level_m
        for c in self.components:
            phase_angle = c.kx * x + c.ky * y - c.omega * t + c.phase
            eta += c.amp * math.cos(phase_angle)
        return eta

    def get_surface(self, x: float, y: float, t: float) -> Dict[str, Any]:
        """波面標高、勾配、単位法線、Gerstner変位を一括算出。"""
        elevation = self.params.tide_level_m
        slope_x = 0.0
        slope_y = 0.0
        disp_x = 0.0
        disp_y = 0.0

        for c in self.components:
            phase_angle = c.kx * x + c.ky * y - c.omega * t + c.phase
            cos_val = math.cos(phase_angle)
            sin_val = math.sin(phase_angle)

            elevation += c.amp * cos_val
            slope_x -= c.amp * c.kx * sin_val
            slope_y -= c.amp * c.ky * sin_val

            if c.k > 1e-9:
                disp_x -= c.choppiness * c.amp * (c.kx / c.k) * sin_val
                disp_y -= c.choppiness * c.amp * (c.ky / c.k) * sin_val

        nx = -slope_x
        ny = -slope_y
        nz = 1.0
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)

        return {
            "elevation": elevation,
            "slope_x": slope_x,
            "slope_y": slope_y,
            "normal": [nx / norm, ny / norm, nz / norm],
            "disp_x": disp_x,
            "disp_y": disp_y,
        }

    def get_water_velocity(self, x: float, y: float, z: float, t: float) -> Tuple[float, float, float]:
        """3D水粒子速度ベクトル (u, v, w) [m/s]。"""
        eta = self.get_elevation(x, y, t)
        if z > eta:
            return (0.0, 0.0, 0.0)

        h = max(1.0, self.params.water_depth_m)
        u = 0.0
        v = 0.0
        w = 0.0

        for c in self.components:
            phase_angle = c.kx * x + c.ky * y - c.omega * t + c.phase
            cos_val = math.cos(phase_angle)
            sin_val = math.sin(phase_angle)

            kh_clamped = min(20.0, c.k * h)
            kzh_clamped = min(20.0, c.k * (z + h))
            sinh_kh = math.sinh(kh_clamped)

            depth_decay_h = math.cosh(kzh_clamped) / sinh_kh
            depth_decay_v = math.sinh(kzh_clamped) / sinh_kh

            if c.k > 1e-9:
                u += c.omega * c.amp * (c.kx / c.k) * depth_decay_h * cos_val
                v += c.omega * c.amp * (c.ky / c.k) * depth_decay_h * cos_val
                w += c.omega * c.amp * depth_decay_v * sin_val

        return (u, v, w)

    @staticmethod
    def solve_dispersion_relation(omega: float, depth_h: float) -> float:
        """有限水深分散関係 w^2 = g * k * tanh(k * h) を解いて波数 k [rad/m] を求める。"""
        k0 = (omega * omega) / GRAVITY
        if depth_h >= 200.0 or (k0 * depth_h) > 5.0:
            return k0

        k = k0
        for _ in range(10):
            kh = k * depth_h
            tanh_kh = math.tanh(kh)
            f = GRAVITY * k * tanh_kh - (omega * omega)
            df = GRAVITY * (tanh_kh + kh * (1.0 - tanh_kh * tanh_kh))
            dk = f / df
            k -= dk
            if abs(dk) < 1e-7:
                break
        return max(1e-4, k)
