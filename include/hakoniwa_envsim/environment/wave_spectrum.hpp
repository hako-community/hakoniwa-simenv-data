/**
 * @file wave_spectrum.hpp
 * @brief JONSWAPスペクトルおよび多成分Gerstner波の決定論的波面生成ライブラリ (C++17 Header-only)
 *
 * 物理エンジン (MuJoCo ShipHydroPlugin)、Python環境 (テスト・シナリオ)、
 * および3Dビューア (Unreal Engine 5.8 / Godot 4.6 シェーダ) の3者で、
 * 同一のシードと環境パラメータからビットレベルで同一の波面標高・法線・粒子速度を
 * 決定論的に展開・共有するための共通数学コア。
 */

#ifndef SHIP_ENVIRONMENT_WAVE_SPECTRUM_HPP_
#define SHIP_ENVIRONMENT_WAVE_SPECTRUM_HPP_

#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace ship {
namespace environment {

constexpr double PI = 3.14159265358979323846;
constexpr double GRAVITY = 9.80665;

/**
 * @brief 海上環境外乱 PDU (hako_msgs/Disturbance) から供給される波浪パラメータ
 */
struct WaveParams {
    double hs_m = 2.5;            ///< 有義波高 [m]
    double tp_s = 8.0;            ///< ピーク周期 [s]
    double wave_dir_rad = 0.0;    ///< 主波向 [rad] (ENU: 0=東, pi/2=北)
    double gamma = 3.3;           ///< JONSWAP 尖鋭度パラメータ (1.0..7.0)
    uint32_t seed = 42;           ///< 決定論的乱数シード
    double water_depth_m = 100.0; ///< 局所水深 [m] (浅水・有限水深補正)
    double tide_level_m = 0.0;    ///< 潮位・平均海面オフセット [m]
    int num_components = 32;      ///< 成分波の数 (16..64)
};

/**
 * @brief 単一の成分波 (Component Wave) パラメータ
 */
struct WaveComponent {
    double omega = 0.0;       ///< 角周波数 [rad/s]
    double k = 0.0;           ///< 波数 (Wave Number) |k| = 2*pi / L_w [rad/m]
    double kx = 0.0;          ///< 波数 X成分 (ENU 東)
    double ky = 0.0;          ///< 波数 Y成分 (ENU 北)
    double amp = 0.0;         ///< 振幅 [m] (a_i = sqrt(2 * S(w) * dw))
    double phase = 0.0;       ///< 初期位相 phi_i [0, 2*pi)
    double dir_rad = 0.0;     ///< 進行方向 [rad]
    double choppiness = 0.0;  ///< Gerstner水平急峻度 Q_i
};

/**
 * @brief 波面評価結果
 */
struct WaveSurface {
    double elevation = 0.0;       ///< 標高 z = eta(x, y, t) [m]
    double slope_x = 0.0;         ///< 勾配 d(eta)/dx
    double slope_y = 0.0;         ///< 勾配 d(eta)/dy
    double normal[3] = {0, 0, 1}; ///< 単位法線ベクトル (nx, ny, nz)
    double disp_x = 0.0;          ///< Gerstner水平変位 dx [m]
    double disp_y = 0.0;          ///< Gerstner水平変位 dy [m]
};

/**
 * @brief 3D水粒子速度ベクトル
 */
struct WaterVelocity {
    double u = 0.0; ///< 東方向流速 [m/s]
    double v = 0.0; ///< 北方向流速 [m/s]
    double w = 0.0; ///< 上方向鉛直流速 [m/s]
};

/**
 * @brief 決定論的 PRNG (Xorshift32: C++, Python, HLSL, GLSL 完全互換)
 */
class DeterministicRNG {
public:
    explicit DeterministicRNG(uint32_t seed) {
        state_ = (seed == 0) ? 1u : seed;
    }

    uint32_t next_uint32() {
        uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        return x;
    }

    /// [0.0, 1.0) の一様実数
    double next_double() {
        return static_cast<double>(next_uint32()) / 4294967296.0;
    }

private:
    uint32_t state_;
};

/**
 * @brief JONSWAP / Gerstner 波面生成エンジン
 */
class WaveSpectrumEngine {
public:
    WaveSpectrumEngine() = default;

    /**
     * @brief 初期化・成分波テーブルの確定的一括生成
     */
    void initialize(const WaveParams& params) {
        params_ = params;
        components_.clear();
        components_.reserve(params.num_components);

        DeterministicRNG rng(params.seed);

        // 1. 周波数レンジの離散化 (omega_min = 0.2, omega_max = 3.2 rad/s)
        const double omega_min = 0.20;
        const double omega_max = 3.20;
        const double d_omega = (omega_max - omega_min) / params.num_components;
        const double omega_p = (2.0 * PI) / std::max(1.0, params.tp_s);
        const double target_m0 = std::pow(params.hs_m / 4.0, 2);

        // 2. JONSWAP 生スペクトル S_raw(omega) の算出と数値積分
        std::vector<double> omegas(params.num_components);
        std::vector<double> s_raw(params.num_components);
        double integral_s = 0.0;

        for (int i = 0; i < params.num_components; ++i) {
            double w = omega_min + (i + 0.5) * d_omega;
            omegas[i] = w;

            // ピーク形状パラメータ sigma
            double sigma = (w <= omega_p) ? 0.07 : 0.09;
            double r = std::exp(-std::pow(w - omega_p, 2) / (2.0 * std::pow(sigma * omega_p, 2)));

            // Pierson-Moskowitz 基礎スペクトル
            double pm = std::pow(w, -5.0) * std::exp(-1.25 * std::pow(omega_p / w, 4.0));
            // JONSWAP ピーク強調 (gamma^r)
            double s_val = pm * std::pow(std::max(1.0, params.gamma), r);

            s_raw[i] = s_val;
            integral_s += s_val * d_omega;
        }

        // 3. 有義波高 Hs に一致させる正規化係数 alpha
        double alpha_norm = (integral_s > 1e-9) ? (target_m0 / integral_s) : 1.0;

        // 4. 方向分散および有限水深分散関係の計算
        double total_ka = 0.0;

        for (int i = 0; i < params.num_components; ++i) {
            WaveComponent comp;
            comp.omega = omegas[i];
            double s_norm = s_raw[i] * alpha_norm;
            comp.amp = std::sqrt(2.0 * s_norm * d_omega);

            // 有限水深分散関係: w^2 = g * k * tanh(k * h)
            comp.k = solve_dispersion_relation(comp.omega, params.water_depth_m);

            // 決定論的初期位相 phi_i in [0, 2*pi)
            comp.phase = rng.next_double() * 2.0 * PI;

            // 方向分散 (主波向を中心に ±45度で拡散)
            double spread_offset = (rng.next_double() - 0.5) * (PI / 2.0);
            comp.dir_rad = params.wave_dir_rad + spread_offset;

            comp.kx = comp.k * std::cos(comp.dir_rad);
            comp.ky = comp.k * std::sin(comp.dir_rad);

            total_ka += comp.k * comp.amp;
            components_.push_back(comp);
        }

        // 5. 波の自己交差を防ぐ Gerstner チョッピーネス正規化 (sum(Q_i * k_i * a_i) <= 1.0)
        double max_choppiness = 0.85; // 破波を防ぐ上限
        for (auto& c : components_) {
            if (total_ka > 1e-6) {
                c.choppiness = std::min(max_choppiness, max_choppiness / total_ka);
            } else {
                c.choppiness = 0.0;
            }
        }
    }

    /**
     * @brief 任意位置 (x, y) および時刻 t における波面標高 eta(x, y, t) [m]
     */
    double get_elevation(double x, double y, double t) const {
        double eta = params_.tide_level_m;
        for (const auto& c : components_) {
            double phase_angle = c.kx * x + c.ky * y - c.omega * t + c.phase;
            eta += c.amp * std::cos(phase_angle);
        }
        return eta;
    }

    /**
     * @brief 波面標高・勾配・法線・Gerstner変位の一括評価
     */
    WaveSurface get_surface(double x, double y, double t) const {
        WaveSurface surf;
        surf.elevation = params_.tide_level_m;

        for (const auto& c : components_) {
            double phase_angle = c.kx * x + c.ky * y - c.omega * t + c.phase;
            double cos_val = std::cos(phase_angle);
            double sin_val = std::sin(phase_angle);

            // 標高
            surf.elevation += c.amp * cos_val;

            // 勾配 d(eta)/dx, d(eta)/dy
            surf.slope_x -= c.amp * c.kx * sin_val;
            surf.slope_y -= c.amp * c.ky * sin_val;

            // Gerstner 水平変位
            if (c.k > 1e-9) {
                surf.disp_x -= c.choppiness * c.amp * (c.kx / c.k) * sin_val;
                surf.disp_y -= c.choppiness * c.amp * (c.ky / c.k) * sin_val;
            }
        }

        // 単位法線ベクトル n = (-d_eta/dx, -d_eta/dy, 1) / norm
        double nx = -surf.slope_x;
        double ny = -surf.slope_y;
        double nz = 1.0;
        double norm = std::sqrt(nx * nx + ny * ny + nz * nz);

        surf.normal[0] = nx / norm;
        surf.normal[1] = ny / norm;
        surf.normal[2] = nz / norm;

        return surf;
    }

    /**
     * @brief 任意点 (x, y, z) における3D水粒子速度ベクトル (u, v, w) [m/s]
     */
    WaterVelocity get_water_velocity(double x, double y, double z, double t) const {
        WaterVelocity vel;
        // 水面より上 (空中) では水流速は 0
        double eta = get_elevation(x, y, t);
        if (z > eta) {
            return vel;
        }

        double h = std::max(1.0, params_.water_depth_m);

        for (const auto& c : components_) {
            double phase_angle = c.kx * x + c.ky * y - c.omega * t + c.phase;
            double cos_val = std::cos(phase_angle);
            double sin_val = std::sin(phase_angle);

            // 有限水深における鉛直減衰係数
            // 水底 (z = -h) で鉛直流速 w = 0
            double depth_decay_h = std::cosh(std::min(20.0, c.k * (z + h))) / std::sinh(std::min(20.0, c.k * h));
            double depth_decay_v = std::sinh(std::min(20.0, c.k * (z + h))) / std::sinh(std::min(20.0, c.k * h));

            if (c.k > 1e-9) {
                vel.u += c.omega * c.amp * (c.kx / c.k) * depth_decay_h * cos_val;
                vel.v += c.omega * c.amp * (c.ky / c.k) * depth_decay_h * cos_val;
                vel.w += c.omega * c.amp * depth_decay_v * sin_val;
            }
        }
        return vel;
    }

    const std::vector<WaveComponent>& get_components() const {
        return components_;
    }

    const WaveParams& get_params() const {
        return params_;
    }

private:
    /**
     * @brief 有限水深分散関係 w^2 = g * k * tanh(k * h) を解いて波数 k [rad/m] を求める
     */
    static double solve_dispersion_relation(double omega, double depth_h) {
        // 深水初期推定 k0 = w^2 / g
        double k0 = (omega * omega) / GRAVITY;
        if (depth_h >= 200.0 || (k0 * depth_h) > 5.0) {
            return k0; // 深水域
        }

        // ニュートン・ラフソン法による高精度求解 (収束 3〜4反復)
        double k = k0;
        for (int iter = 0; iter < 10; ++iter) {
            double kh = k * depth_h;
            double tanh_kh = std::tanh(kh);
            double f = GRAVITY * k * tanh_kh - (omega * omega);
            double df = GRAVITY * (tanh_kh + kh * (1.0 - tanh_kh * tanh_kh));
            double dk = f / df;
            k -= dk;
            if (std::abs(dk) < 1e-7) {
                break;
            }
        }
        return std::max(1e-4, k);
    }

    WaveParams params_;
    std::vector<WaveComponent> components_;
};

} // namespace environment
} // namespace ship

#endif // SHIP_ENVIRONMENT_WAVE_SPECTRUM_HPP_
