#include "isaaclab/envs/mdp/depth/depth_utils.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace { // —— 内部小工具 ——

// 射线距离 r → 正交深度 z（只在 data_type=="distance_to_camera" 且需要正交化时使用）
inline float rangeToOrthoZ(float r, int u, int v, float fx, float fy, float cx, float cy) {
  const float dx = (u - cx) / fx;
  const float dy = (v - cy) / fy;
  const float denom = std::sqrt(1.f + dx*dx + dy*dy);
  return (denom > 1e-9f) ? (r / denom) : r;
}

// 最近邻 / 双线性 resize 到 (H2, W2)
enum class _ResizeMode { Nearest, Bilinear };

inline void resizeDepth(const std::vector<float>& src, int H1, int W1,
                        std::vector<float>& dst, int H2, int W2, _ResizeMode mode)
{
  dst.resize(H2 * W2);
  if (mode == _ResizeMode::Nearest) {
    for (int y = 0; y < H2; ++y) {
      int yy = int(std::lround((y + 0.5) * H1 / double(H2) - 0.5));
      yy = std::clamp(yy, 0, H1 - 1);
      for (int x = 0; x < W2; ++x) {
        int xx = int(std::lround((x + 0.5) * W1 / double(W2) - 0.5));
        xx = std::clamp(xx, 0, W1 - 1);
        dst[y * W2 + x] = src[yy * W1 + xx];
      }
    }
  } else {
    for (int y = 0; y < H2; ++y) {
      double sy = (y + 0.5) * H1 / double(H2) - 0.5;
      int y0 = std::clamp(int(std::floor(sy)), 0, H1 - 1);
      int y1 = std::clamp(y0 + 1, 0, H1 - 1);
      double wy = sy - y0;
      for (int x = 0; x < W2; ++x) {
        double sx = (x + 0.5) * W1 / double(W2) - 0.5;
        int x0 = std::clamp(int(std::floor(sx)), 0, W1 - 1);
        int x1 = std::clamp(x0 + 1, 0, W1 - 1);
        double wx = sx - x0;
        float v00 = src[y0 * W1 + x0];
        float v01 = src[y0 * W1 + x1];
        float v10 = src[y1 * W1 + x0];
        float v11 = src[y1 * W1 + x1];
        dst[y * W2 + x] =
            float((1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11));
      }
    }
  }
}

} // anon

namespace isaaclab::mdp {

// ---------------- preprocessDepth ----------------
std::vector<float> preprocessDepth(const std::vector<float>& depth_in_m, int H, int W,
                                   const DepthPreprocCfg& C)
{
  if (int(depth_in_m.size()) != H * W) {
    throw std::runtime_error("preprocessDepth: input size mismatch");
  }

  // 1) 数值净化：nan/inf/<=0 → clip_max
  std::vector<float> tmp(depth_in_m);
  for (auto& v : tmp) {
    if (!std::isfinite(v) || v <= 0.f) v = C.clip_max;
  }

  // 2) 可选正交化（distance_to_camera → z）
  if (C.data_type == "distance_to_camera" && C.convert_perspective_to_orthogonal) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        float r = tmp[y * W + x];
        tmp[y * W + x] = rangeToOrthoZ(r, x, y, C.fx, C.fy, C.cx, C.cy);
      }
    }
  }

  // 3) 裁剪
  const int top = C.crop_top, bot = C.crop_bottom, left = C.crop_left, right = C.crop_right;
  const int Hc = H - top - bot;
  const int Wc = W - left - right;
  if (Hc <= 0 || Wc <= 0) throw std::runtime_error("preprocessDepth: invalid crop region");
  std::vector<float> cropped; cropped.reserve(Hc * Wc);
  for (int y = 0; y < Hc; ++y) {
    const float* row = tmp.data() + (top + y) * W + left;
    cropped.insert(cropped.end(), row, row + Wc);
  }

  // 4) resize 到 (out_h, out_w)
  std::vector<float> resized;
  _ResizeMode rm = (C.resize_mode == ResizeMode::Nearest) ? _ResizeMode::Nearest : _ResizeMode::Bilinear;
  resizeDepth(cropped, Hc, Wc, resized, C.out_h, C.out_w, rm);

  // 5) clip + 归一化到 [0,1]
  const float scale = std::max(1e-6f, C.clip_max - C.clip_min);
  for (auto& v : resized) {
    if (v < C.clip_min) v = C.clip_min;
    if (v > C.clip_max) v = C.clip_max;
    v = (v - C.clip_min) / scale;
  }
  return resized; // 长度 out_h*out_w
}

// ---------------- DepthRing 实现 ----------------
class DepthRing {
public:
  DepthRing(int frames_per_refresh, int delay_frames, int out_len)
  : fpr_(std::max(1, frames_per_refresh)),
    delay_(std::max(1, delay_frames)),
    buf_(std::max(delay_ + 2, fpr_ + 2), std::vector<float>(out_len, 1.0f)) {}

  void maybeWrite(const std::vector<float>& flat_img_norm) {
    if ((step_ - last_refresh_) >= fpr_) {
      buf_[w_ % int(buf_.size())] = flat_img_norm;
      w_ = (w_ + 1) % int(buf_.size());
      last_refresh_ = step_;
    }
  }

  const std::vector<float>& readDelayed() const {
    int read = w_ - delay_;
    read %= int(buf_.size());
    if (read < 0) read += int(buf_.size());
    return buf_[read];
  }

  void step() { ++step_; }

private:
  int fpr_;
  int delay_;
  mutable int w_ = 0;
  int step_ = 0;
  int last_refresh_ = -1000000;
  std::vector<std::vector<float>> buf_;
};

// ---------------- 单相机全局句柄 ----------------
namespace {
  std::unique_ptr<DepthRing> g_ring;
  DepthPreprocCfg g_cfg;
  int g_out_len = 48 * 64;
  int g_frames_per_refresh = 5;
  int g_delay_frames = 2;
} // anon

void init_forward_depth(double control_dt,
                        double refresh_hz,
                        double delay_s,
                        const DepthPreprocCfg& cfg)
{
  g_cfg = cfg;
  g_out_len = cfg.out_h * cfg.out_w;
  g_frames_per_refresh = std::max(1, int(std::lround(1.0 / (refresh_hz * control_dt))));
  g_delay_frames       = std::max(1, int(std::lround(delay_s * refresh_hz)));
  g_ring = std::make_unique<DepthRing>(g_frames_per_refresh, g_delay_frames, g_out_len);
}

void pump_depth_each_step(const std::vector<float>& depth_m, int H, int W) {
  if (!g_ring) return;
  auto flat01 = preprocessDepth(depth_m, H, W, g_cfg);
  g_ring->maybeWrite(flat01); // 只有到刷新步才写
  g_ring->step();             // 每个控制步都 step
}

std::vector<float> obs_forward_depth() {
  if (!g_ring) return std::vector<float>(g_out_len, 1.0f);
  return g_ring->readDelayed(); // [0,1]，长度 out_h*out_w
}

} // namespace isaaclab::mdp
