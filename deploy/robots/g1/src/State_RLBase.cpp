#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

// 深度桥接 & 预处理
#include "ipc/depth_shm_reader.hpp"
#include "isaaclab/envs/mdp/depth/depth_utils.h"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <cstring>             // memcpy
#include <cmath>
#include <algorithm>

// ===== 构造 =====
State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    spdlog::info("Initializing State_{}...", state_string);
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile((policy_dir / "params" / "deploy.yaml").string()),
        std::make_shared<unitree::BaseArticulation<g1::subscription::LowState::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>((policy_dir / "exported" / "policy.onnx").string());

    // === 深度共享内存读取器 ===
    depth_reader_ = std::make_unique<depth_ipc::DepthShmReader>(); // 默认 /depth_shm0
    spdlog::info("DepthShmReader ready (shm='{}')", depth_ipc::kDefaultShmName);

    // === 初始化 depth_utils 预处理环形缓存 ===
    // 你可以把这些参数改成从 YAML 读取；此处给一套合理默认
    isaaclab::mdp::DepthPreprocCfg dcfg;
    dcfg.data_type = "distance_to_camera";          // 我们的 SHM 是 r（到相机距离，单位米）
    dcfg.convert_perspective_to_orthogonal = false; // 若策略期望正交 z，置 true 并设置好内参
    // ⚠️ 拿真实相机内参替换（从 /camera/depth/camera_info 读取）
    dcfg.fx = 600.f;  dcfg.fy = 600.f;
    dcfg.cx = 424.f;  dcfg.cy = 240.f;              // 848x480 时大致中心；请改为标定值

    // 视野裁剪（按需调整）
    dcfg.crop_top = 0; dcfg.crop_bottom = 0;
    dcfg.crop_left = 0; dcfg.crop_right = 0;

    // 输出尺寸（与你训练时一致）
    dcfg.out_h = 48; dcfg.out_w = 64;
    dcfg.resize_mode = isaaclab::mdp::ResizeMode::Bilinear;

    // 剪裁与归一化范围（米）
    dcfg.clip_min = 0.2f;
    dcfg.clip_max = 5.0f;

    // 预处理刷新/延迟：控制周期 env->step_dt；希望 30Hz 刷新，延迟 2 帧
    const double control_dt = env->step_dt;   // 例如 0.01
    const double refresh_hz = 30.0;
    const double delay_s    = 2.0 / refresh_hz;

    isaaclab::mdp::init_forward_depth(control_dt, refresh_hz, delay_s, dcfg);
    spdlog::info("Depth preproc ring initialized: out={}x{}, clip=[{:.2f},{:.2f}]",
                 dcfg.out_h, dcfg.out_w, dcfg.clip_min, dcfg.clip_max);
}

// ===== 每步运行 =====
void State_RLBase::run()
{
    // === 1) 从共享内存取“最新整帧”（米）并送入 depth_utils 预处理环 ===
    if (depth_reader_) {
        const float* d = nullptr; uint32_t W = 0, H = 0; uint64_t tns = 0, seq = 0;
        if (depth_reader_->latest(d, W, H, tns, seq) && d && W > 0 && H > 0) {
            // （A）统计性缓存：中心像素、中心 ROI 均值（可用于日志/安全检测）
            if (seq != depth_seq_) {
                depth_w_ = W; depth_h_ = H;
                depth_seq_ = seq; depth_stamp_ns_ = tns;
                depth_center_m_   = d[(H/2) * W + (W/2)];
                depth_roi_mean_m_ = roi_mean_center(d, W, H, /*half=*/10);
                // spdlog::debug("Depth seq={} center={:.3f}m roi_mean={:.3f}m",
                //               (unsigned long long)seq, depth_center_m_, depth_roi_mean_m_);
            }

            // （B）整帧复制到 vector 并泵入预处理（NaN→clip、裁剪、重采样、归一化）
            static std::vector<float> frame_m;                   // 复用缓冲避免频繁分配
            frame_m.resize(size_t(W) * H);
            std::memcpy(frame_m.data(), d, frame_m.size() * sizeof(float));
            isaaclab::mdp::pump_depth_each_step(frame_m, int(H), int(W));
        }
        else {
            // 没帧到也要推进步计数，保持与控制相位一致（pump_depth_each_step 内部会 step）
            // 这里不额外处理即可（我们只在拿到帧时调用 pump）
        }
    }

    // === 2) 你的原有控制逻辑（不变） ===
    auto action = env->action_manager->processed_actions();
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

// ===== 小工具：中心 ROI 均值 =====
float State_RLBase::roi_mean_center(const float* d, uint32_t W, uint32_t H, int half) {
    if (!d || W == 0 || H == 0) return NAN;
    const int cx = int(W) / 2, cy = int(H) / 2;
    const int x0 = std::max(0, cx - half), x1 = std::min<int>(W - 1, cx + half);
    const int y0 = std::max(0, cy - half), y1 = std::min<int>(H - 1, cy + half);

    double sum = 0.0; size_t cnt = 0;
    for (int y = y0; y <= y1; ++y) {
        const size_t row = size_t(y) * W;
        for (int x = x0; x <= x1; ++x) {
            const float v = d[row + x];
            if (std::isfinite(v) && v > 0.f) { sum += v; cnt++; }
        }
    }
    return (cnt ? float(sum / cnt) : NAN);
}
