#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <yaml-cpp/yaml.h>

// 现在先用绝对路径，保证能编过；之后可以换成 #include "cnpy.h" + 正确的 CMake include
#include "/home/tianhup/Desktop/unitree_rl_lab/deploy/thirdparty/cnpy/cnpy.h"

#include "isaaclab/envs/mdp/observations.h"   // 里面会包含 observation_manager.h，提供 REGISTER_OBSERVATION 宏

namespace isaaclab {

// ------------ 可按需修改的常量（和你 Python 脚本保持一致） ------------
static const char* kNPY_PATH =
    "/home/tianhup/Desktop/unitree_rl_lab/deploy/robots/g1/config/policy/motion_tracking/motion_reference/G1_sim_retargeted_result.npy";

// URDF -> USD 重排（29）
static const int kURDF2USD[29] = {
    0,6,12, 1,7,13, 2,8,14, 3,9,15, 22, 4,10,16,23, 5,11,17,24, 18,25, 19,26, 20,27, 21,28
};

// 从 USD 里挑的 14 个 DOF（和你 Python 里 dof_ids_subset 一样）
static const int kSUBSET[14] = {
    11,12, 15,16, 19,20,21,22,23,24,25,26,27,28
};

// 下采样：200 Hz -> 50 Hz（每 4 帧取 1 帧）
static const int kDECIMATION = 4;
// --------------------------------------------------------------------

// 缓存帧序列（每帧 14 维）
static std::vector<std::vector<float>>& frames()
{
    static bool loaded = false;
    static std::vector<std::vector<float>> g_frames;

    if (!loaded) {
        cnpy::NpyArray arr = cnpy::npy_load(kNPY_PATH);
        if (arr.shape.size() != 2) {
            throw std::runtime_error("DemoNPYObs: npy must be 2D (T x D)");
        }
        const size_t T = arr.shape[0];
        const size_t D = arr.shape[1];
        if (D < 29) {
            throw std::runtime_error("DemoNPYObs: second dim must be >= 29");
        }

        g_frames.reserve((T + kDECIMATION - 1) / kDECIMATION);

        if (arr.word_size == sizeof(double)) {
            const double* base = arr.data<double>();
            for (size_t t = 0; t < T; t += kDECIMATION) {
                const double* row = base + t * D;
                // URDF -> USD（29）
                float usd29[29];
                for (int i = 0; i < 29; ++i) {
                    usd29[i] = static_cast<float>(row[kURDF2USD[i]]);
                }
                // 取 subset 14
                std::vector<float> sub14;
                sub14.reserve(14);
                for (int j = 0; j < 14; ++j) {
                    sub14.push_back(usd29[kSUBSET[j]]);
                }
                g_frames.push_back(std::move(sub14));
            }
        }
        else if (arr.word_size == sizeof(float)) {
            const float* base = arr.data<float>();
            for (size_t t = 0; t < T; t += kDECIMATION) {
                const float* row = base + t * D;
                float usd29[29];
                for (int i = 0; i < 29; ++i) {
                    usd29[i] = row[kURDF2USD[i]];
                }
                std::vector<float> sub14;
                sub14.reserve(14);
                for (int j = 0; j < 14; ++j) {
                    sub14.push_back(usd29[kSUBSET[j]]);
                }
                g_frames.push_back(std::move(sub14));
            }
        }
        else {
            throw std::runtime_error("DemoNPYObs: unsupported dtype (word_size not 4 or 8).");
        }

        if (g_frames.empty()) {
            throw std::runtime_error("DemoNPYObs: no frames after decimation.");
        }
        loaded = true;
    }
    return g_frames;
}

// 全局索引：只在 curr_demo_dof_pos 里递增
static int& frame_idx()
{
    static int idx = -1;  // 第一次调用先 ++ 变 0
    return idx;
}

// 注册：curr_demo_dof_pos（返回当前帧，并把索引 +1）
REGISTER_OBSERVATION(curr_demo_dof_pos)
{
    auto& f = frames();
    auto& idx = frame_idx();
    if (idx + 1 < static_cast<int>(f.size())) {
        ++idx;
    } else {
        idx = static_cast<int>(f.size()) - 1;  // 到尾帧就停住
    }
    return f[idx];  // 14 维
}

// 注册：next_demo_dof_pos（不修改索引，返回 idx+1）
REGISTER_OBSERVATION(next_demo_dof_pos)
{
    auto& f = frames();
    int idx = frame_idx();
    int idx_next = std::min(idx + 1, static_cast<int>(f.size()) - 1);
    return f[idx_next];  // 14 维
}

} // namespace isaaclab