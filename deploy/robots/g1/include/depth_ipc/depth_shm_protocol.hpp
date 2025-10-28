#pragma once
#include <cstdint>
#include <cstring>

namespace depth_ipc {

struct DepthHeader {
  uint32_t width, height, stride, ready_idx;
  uint64_t seq, stamp_ns;
  char     encoding[16];
  char     _pad[64 - 4*4 - 2*8 - 16];
};

static constexpr const char* kDefaultShmName = "/depth_shm0";
static constexpr uint32_t kMaxW = 1280;
static constexpr uint32_t kMaxH = 720;

inline size_t shm_size_bytes() {
  const size_t header = sizeof(DepthHeader);
  const size_t plane  = size_t(kMaxW) * size_t(kMaxH) * sizeof(float);
  return header + 2 * plane; // 双缓冲
}

} // namespace depth_ipc
