// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <array>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
// 头部（observations.h 或对应 .cpp 顶部）补充：
#include <numbers>     // ✅ 用 std::numbers::pi_v<float>

#include <iostream>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <sstream>
#include <limits>


#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/depth/depth_utils.h"   // ✅ 新增：forward_depth 依赖

#include "/home/tianhup/Desktop/unitree_rl_lab/deploy/thirdparty/cnpy/cnpy.h"
// #include <cnpy.h>
#include <spdlog/spdlog.h>
#include <eigen3/Eigen/Dense>  // ✅ 用到了 Eigen::VectorXf，就需要这个
namespace {
inline std::vector<float> to_vec(const std::vector<float>& v) { return v; }
inline std::vector<float> to_vec(const Eigen::VectorXf& v) {
    std::vector<float> out(static_cast<size_t>(v.size()));
    for (int i = 0; i < v.size(); ++i) out[static_cast<size_t>(i)] = v[i];
    return out;
}

// joint_pos_rel 的首帧“基线”（进入 Velocity 时采样到的 q）
inline bool& jp_baseline_set() { static bool b = false; return b; }
inline std::vector<float>& jp_baseline() { static std::vector<float> v; return v; }
} // anonymous


#include <filesystem>
namespace fs = std::filesystem;

namespace {
inline fs::file_time_type& demo_last_mtime() { 
    static fs::file_time_type t{}; 
    return t; 
}
inline std::string& demo_motion_path() { 
    static std::string p; 
    return p; 
}
inline std::uintmax_t& demo_last_bytes() {
     static std::uintmax_t b = 0; 
     return b; 
}

}
// ==== 可选：提供一个可从别处调用的“重置基线”函数（放在命名空间里）====
namespace isaaclab { namespace mdp {
inline void reset_joint_pos_rel_baseline() {
    jp_baseline().clear();
    jp_baseline_set() = false;
}
}} // namespace

namespace isaaclab {
namespace mdp {

// ------------ 普通 observation，保持不变 ------------
REGISTER_OBSERVATION(base_ang_vel)
{
    auto& asset = env->robot;
    auto& data  = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto& asset = env->robot;
    auto& data  = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}



REGISTER_OBSERVATION(joint_pos_rel)
{
    auto& asset = env->robot;
    std::vector<float> data;

    auto cfg = env->cfg["observations"]["joint_pos_rel"];

    // 统一拿到要用的关节索引 joint_ids
    std::vector<int> joint_ids;
    if (cfg["params"]["asset_cfg"]["joint_ids"].IsDefined()) {
        joint_ids = cfg["params"]["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } else {
        joint_ids.resize(asset->data.joint_pos.size());
        std::iota(joint_ids.begin(), joint_ids.end(), 0);
    }

    // 计算相对关节角： q - q_default
    data.resize(joint_ids.size());
    for (size_t i = 0; i < joint_ids.size(); ++i) {
        int j = joint_ids[i];
        data[i] = asset->data.joint_pos[j] - asset->data.default_joint_pos[j];
    }
    // ===== DEBUG PRINT END =====

    return data;
}



REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto & data = asset->data.joint_vel;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto& joystick = env->robot->data.joystick;

    auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];
    obs[0] = std::clamp( joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    return obs;
}
REGISTER_OBSERVATION(forward_depth)
{
    // 直接从 depth ring 读取（已是 [0,1]、长度 out_h*out_w=48*64）
    return isaaclab::mdp::obs_forward_depth();
}

REGISTER_OBSERVATION(gait_phase)
{
    float period      = env->cfg["observations"]["gait_phase"]["params"]["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase  = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2.f * static_cast<float>(M_PI));
    obs[1] = std::cos(env->global_phase * 2.f * static_cast<float>(M_PI));
    return obs;
}

// ------------ demo 序列：加载与状态机 ------------
namespace {

// 共享缓冲：每帧 14 维
inline std::vector<std::vector<float>>& demo_frames() {
    static std::vector<std::vector<float>> f;
    return f;
}

// 状态标志
inline bool& demo_loaded()         { static bool b = false; return b; }
inline int&  demo_index()          { static int  i = 0;     return i; }
inline bool& demo_curr_inited()    { static bool b = false; return b; }
inline bool& demo_next_inited()    { static bool b = false; return b; }
inline bool& demo_first_compute()  { static bool b = true;  return b; }
inline size_t& demo_last_size() { static size_t n = 0; return n; }
inline bool&   demo_last_inited() { static bool inited = false; return inited; }  // 可选

// 如需在状态切换时复位（可在你的状态机进入 Velocity 时调用）
inline void reset_demo_seq()
{
    demo_index()         = 0;
    demo_curr_inited()   = false;
    demo_next_inited()   = false;
    demo_first_compute() = true;
}

// 加载 .npy 并构建子集
inline void load_motion_if_needed(isaaclab::ManagerBasedRLEnv* env)
{
    if (demo_loaded()) return;

    auto mcfg = env->cfg["motion_reference"];
    if (!mcfg) throw std::runtime_error("motion_reference block missing in deploy.yaml");

    const std::string path  = mcfg["path"].as<std::string>();
    const auto urdf_to_usd  = mcfg["urdf_to_usd"].as<std::vector<int>>();
    const auto subset       = mcfg["subset"].as<std::vector<int>>();   // 14 个 USD 索引
    const int   decimation  = mcfg["decimation"].as<int>(4);

    if (urdf_to_usd.size() != 29) throw std::runtime_error("urdf_to_usd size must be 29");
    if (subset.size()      != 14) throw std::runtime_error("subset size must be 14");

    auto vec2str = [](const std::vector<int>& v){
        std::string s; s.reserve(v.size()*3);
        for (size_t i=0;i<v.size();++i) { s += std::to_string(v[i]); if (i+1<v.size()) s += ", "; }
        return s;
    };

    cnpy::NpyArray arr = cnpy::npy_load(path.c_str());
    if (arr.shape.size() != 2) throw std::runtime_error("npy must be 2D (T x D)");
    const size_t T = arr.shape[0], D = arr.shape[1];
    if (D < 29) throw std::runtime_error("npy second dim must be >= 29");

    spdlog::info("Loaded motion data: ({}, {}) -> reindexed to: ({}, {})", T, D, T, D);
    const size_t T_ds = (T + decimation - 1) / decimation;
    spdlog::info("Downsampled motion data (every {} frames): ({}, 29)", decimation, T_ds);
    spdlog::info("Using DOF subset: [{}]", vec2str(subset));

    auto& frames = demo_frames();
    frames.clear();
    frames.reserve(T_ds);

    const bool use_double = (arr.word_size == sizeof(double));
    for (size_t t=0; t<T; t += std::max(1, decimation)) {
        std::array<float,29> urdf29{};
        if (use_double) {
            const double* row = arr.data<double>() + t*D;
            for (int i=0;i<29;++i) urdf29[i] = static_cast<float>(row[i]);
        } else {
            const float* row = arr.data<float>() + t*D;
            for (int i=0;i<29;++i) urdf29[i] = row[i];
        }
        std::array<float,29> usd29{};
        for (int i=0;i<29;++i) usd29[i] = urdf29[ urdf_to_usd[i] ];

        std::vector<float> sub14; sub14.reserve(14);
        for (int j : subset) {
            if (j<0 || j>=29) throw std::runtime_error("subset index out of range");
            sub14.push_back(usd29[j]);
        }
        frames.push_back(std::move(sub14));
    }

    const size_t N = frames.size();
    spdlog::info("Final motion data subset shape: ({}, 14)", N);
    const float step_dt = env->cfg["step_dt"].as<float>(env->step_dt);
    spdlog::info("Motion data duration: {:.2f} seconds at {:.0f} Hz", N*step_dt, 1.0f/step_dt);

    for (size_t k=0; k<std::min<size_t>(3, N); ++k){
        std::string row; row.reserve(14*8);
        for (int j=0;j<14;++j){ row += std::to_string(frames[k][j]); if(j+1<14) row += ", "; }
        spdlog::info("demo_subset[{}] = [{}]", k, row);
    }

    demo_loaded() = true;
    demo_last_size() = demo_frames().size();  // 记录当前帧数
    reset_demo_seq();    
    // 记录路径 & 当前 mtime
    demo_motion_path() = path;
    std::error_code ec;
    auto p = fs::path(path);
    demo_last_mtime() = fs::last_write_time(p, ec);
    demo_last_bytes() = fs::file_size(p, ec);
}

} // anonymous namespace




inline void reload_if_file_changed(isaaclab::ManagerBasedRLEnv* env)
{
    if (!demo_loaded()) return;
    const std::string& path = demo_motion_path();
    if (path.empty()) return;

    std::error_code ec;
    auto p = fs::path(path);
    auto now_mtime = fs::last_write_time(p, ec);
    auto now_bytes = fs::file_size(p, ec);
    if (ec) return;

    if (now_mtime != demo_last_mtime()|| now_bytes != demo_last_bytes()) {
        // —— 保存现场 ——
        const int  old_idx           = demo_index();
        const bool old_curr_inited   = demo_curr_inited();
        const bool old_next_inited   = demo_next_inited();
        const bool old_first_compute = demo_first_compute();
        const size_t old_N           = demo_last_size();

        // —— 执行重载（不 reset）——
        demo_loaded() = false;

        load_motion_if_needed(env);

        // 更新 mtime 记录
        demo_last_mtime() = now_mtime;
        demo_last_bytes() = now_bytes;

        // —— 恢复状态（保留进度）——
        const size_t new_N = demo_frames().size();
        demo_last_size()   = new_N;

        // 如果确实是“追加”而非“替换”，old prefix 与新数据兼容，此时直接保留 idx
        // Clip 一下，防止越界（比如替换成更短的）
        demo_index()          = std::min<int>(old_idx, int(new_N) - 1);
        demo_curr_inited()    = old_curr_inited;
        demo_next_inited()    = old_next_inited;
        demo_first_compute()  = old_first_compute;  // 若你已进入 steady，则继续 steady

        spdlog::info("[demo] motion_reference npy changed: {} -> {} frames, idx={} (preserved).",
                     old_N, new_N, demo_index());
    }
}
// ------------ 两个 demo observation：按“demo 逻辑”对齐 Python ------------
REGISTER_OBSERVATION(curr_demo_dof_pos)
{   
    reload_if_file_changed(env);
    load_motion_if_needed(env);
    auto& frames = demo_frames();
    int   idx    = std::min(demo_index(), (int)frames.size()-1);

    // 第一次被调用（reset 阶段）：返回 frame0 填充历史
    if (!demo_curr_inited()) {
        demo_curr_inited() = true;
        return frames[idx]; // idx==0 => frame0
    }

    // 第一次 compute：返回 frame1，不推进 idx
    if (demo_first_compute() && idx + 1 < (int)frames.size()) {
        return frames[idx + 1];
    }

    // 之后：返回当前 idx（与 next_demo 同步）
    return frames[idx];
}

REGISTER_OBSERVATION(next_demo_dof_pos)
{   
    reload_if_file_changed(env);
    load_motion_if_needed(env);
    auto& frames = demo_frames();
    int&  idx    = demo_index();
    int   cur    = std::min(idx, (int)frames.size()-1);

    // 第一次被调用（reset 阶段）：返回 frame0 填充历史
    if (!demo_next_inited()) {
        demo_next_inited() = true;
        return frames[cur]; // cur==0 => frame0
    }

    // 第一次 compute：返回 frame1，并把 idx 推进到 1
    if (demo_first_compute()) {
        demo_first_compute() = false;
        if (cur + 1 < (int)frames.size()) {
            ++idx;         // 0 -> 1
            return frames[idx]; // frame1
        }
        return frames[cur]; // 只有一帧的兜底
    }

    // 之后：返回当前，再推进（保证两者在同一 tick 相等）
    auto out = frames[cur];
    if (idx + 1 < (int)frames.size()) ++idx;
    return out;
}

} // namespace mdp
} // namespace isaaclab
