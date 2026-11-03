"""test_wave_spectrum.py - 波浪スペクトル・決定論的波面生成の単体テストスイート (hakoniwa-simenv-data)。"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from hakoniwa_envsim.wave import DeterministicRNG, WaveParams, WaveSpectrumEngine


def test_deterministic_rng_reproducibility():
    """同一シードで全く同一の乱数列が再現されることの検証。"""
    rng1 = DeterministicRNG(12345)
    rng2 = DeterministicRNG(12345)

    seq1 = [rng1.next_double() for _ in range(100)]
    seq2 = [rng2.next_double() for _ in range(100)]

    assert seq1 == seq2
    # 異なるシードでは一致しないこと
    rng3 = DeterministicRNG(54321)
    seq3 = [rng3.next_double() for _ in range(100)]
    assert seq1 != seq3


def test_dispersion_relation():
    """深水・浅水における分散関係の物理的妥当性検証。"""
    g = 9.80665

    # 1. 深水域 (h = 500m, w = 1.0 rad/s -> k ≈ w^2 / g ≈ 0.10197)
    k_deep = WaveSpectrumEngine.solve_dispersion_relation(1.0, 500.0)
    expected_deep = (1.0 ** 2) / g
    assert math.isclose(k_deep, expected_deep, rel_tol=1e-3)

    # 2. 浅水域 (h = 5m, w = 0.2 rad/s -> 長波波速 c ≈ sqrt(g*h) ≈ 7.0m/s -> k ≈ w / c)
    h_shallow = 5.0
    w_shallow = 0.2
    k_shallow = WaveSpectrumEngine.solve_dispersion_relation(w_shallow, h_shallow)
    c_longwave = math.sqrt(g * h_shallow)
    expected_shallow = w_shallow / c_longwave
    assert math.isclose(k_shallow, expected_shallow, rel_tol=0.05)


def test_jonswap_spectrum_energy_normalization():
    """JONSWAPスペクトルの0次モーメント m0 が (Hs / 4)^2 と厳密に一致することの検証。"""
    hs = 3.5
    params = WaveParams(hs_m=hs, tp_s=9.0, gamma=3.3, num_components=64)
    engine = WaveSpectrumEngine(params)

    # 成分波振幅 a_i から全分散 sum(0.5 * a_i^2) を計算
    # 線形不規則波理論: m0 = sum(0.5 * a_i^2) = (Hs / 4)^2
    total_m0 = sum(0.5 * (c.amp ** 2) for c in engine.components)
    target_m0 = (hs / 4.0) ** 2

    assert math.isclose(total_m0, target_m0, rel_tol=1e-4)


def test_wave_elevation_deterministic():
    """空間・時間座標での波高計算の決定論的再現性検証。"""
    params = WaveParams(hs_m=2.0, tp_s=7.0, seed=999)
    eng1 = WaveSpectrumEngine(params)
    eng2 = WaveSpectrumEngine(params)

    for x in [-50.0, 0.0, 50.0]:
        for y in [-30.0, 30.0]:
            for t in [0.0, 1.0, 10.0]:
                eta1 = eng1.get_elevation(x, y, t)
                eta2 = eng2.get_elevation(x, y, t)
                assert math.isclose(eta1, eta2, abs_tol=1e-12)
