#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <thread>
#include "ipc/depth_shm_reader.hpp"

int main() {
  try {
    depth_ipc::DepthShmReader reader;

    uint64_t last_seq = 0;
    auto last_t = std::chrono::steady_clock::now();

    for (int i = 0; i < 100; ++i) {
      const float* d = nullptr; uint32_t W = 0, H = 0; uint64_t tns = 0, seq = 0;

      if (reader.latest(d, W, H, tns, seq) && d && W > 0 && H > 0) {
        // 中心像素
        const size_t ci = (size_t(H)/2) * W + (size_t(W)/2);
        const float center = d[ci];

        // 粗略有效像素比例（网格抽样，避免 O(W*H) 全量遍历）
        size_t valid = 0, total = 0;
        const uint32_t stepY = std::max(1u, H / 32); // 约 32×32 采样
        const uint32_t stepX = std::max(1u, W / 32);
        for (uint32_t y = 0; y < H; y += stepY) {
          const size_t row = size_t(y) * W;
          for (uint32_t x = 0; x < W; x += stepX) {
            const float v = d[row + x];
            total++;
            if (std::isfinite(v) && v > 0.f) valid++;
          }
        }
        const float valid_ratio = total ? float(valid) / float(total) : 0.f;

        // 是否在更新（seq 是否前进）
        const bool stalled = (seq == last_seq);
        auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - last_t).count();
        const double fps = (seq > last_seq && dt > 1e-6) ? (seq - last_seq) / dt : 0.0;

        printf("seq=%llu%s  %ux%u  center=%.3f m  valid~%.0f%%  fps=%.1f\n",
               (unsigned long long)seq,
               stalled ? " (stalled)" : "",
               W, H,
               center,
               valid_ratio * 100.0f,
               fps);

        last_seq = seq;
        last_t = now;
      } else {
        printf("no depth yet\n");
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100 ms
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERR: %s\n", e.what());
    return 1;
  }
  return 0;
}
