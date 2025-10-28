#pragma once
#include <vector>
#include <string>

namespace isaaclab::mdp {

// 和 cpp 一致的插值模式
enum class ResizeMode { Nearest, Bilinear };

// 预处理配置：严格对齐 depth_utils.cpp 里用到的所有字段
struct DepthPreprocCfg {
  // 输入类型
  std::string data_type = "depth";              // "depth" 或 "distance_to_camera"
  bool convert_perspective_to_orthogonal = true;

  // 裁剪 & 输出
  int crop_top = 0, crop_bottom = 0, crop_left = 0, crop_right = 0;
  int out_h = 48, out_w = 64;
  ResizeMode resize_mode = ResizeMode::Nearest;

  // 数值域
  float clip_min = 0.12f, clip_max = 2.0f;

  // 相机内参（仅当 data_type=="distance_to_camera" 且需要正交化时使用）
  float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;
};

// 预处理：裁剪→(可选正交化)→resize→clip→归一化到[0,1]，返回长度 out_h*out_w
std::vector<float> preprocessDepth(
    const std::vector<float>& depth_in_m, // H*W，单位米
    int H, int W,
    const DepthPreprocCfg& cfg);

// 初始化 ring：用控制步长/相机刷新率/固定延迟计算帧间隔与延迟帧数
void init_forward_depth(double control_dt,
                        double refresh_hz,
                        double delay_s,
                        const DepthPreprocCfg& cfg);

// 每步：喂入一帧“原始深度(米)”（H×W）。内部按刷新频率决定是否写入。
void pump_depth_each_step(const std::vector<float>& depth_m, int H, int W);

// 给 REGISTER_OBSERVATION(forward_depth) 用：返回 [0,1] 扁平 48×64
std::vector<float> obs_forward_depth();

} // namespace isaaclab::mdp
