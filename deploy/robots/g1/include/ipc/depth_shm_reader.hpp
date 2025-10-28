#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>
#include <cstdint>
#include "depth_ipc/depth_shm_protocol.hpp"

namespace depth_ipc {

class DepthShmReader {
public:
  explicit DepthShmReader(const char* name = kDefaultShmName) {
    fd_ = shm_open(name, O_RDWR, 0666);
    if (fd_ < 0) throw std::runtime_error("shm_open failed");
    size_ = shm_size_bytes();
    base_ = static_cast<uint8_t*>(mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    hdr_ = reinterpret_cast<DepthHeader*>(base_);
  }
  ~DepthShmReader() {
    if (base_) munmap(base_, size_);
    if (fd_>=0) close(fd_);
  }

  // 返回最新帧的指针与尺寸（指针指向共享内存，不要 free）
  bool latest(const float*& ptr, uint32_t& w, uint32_t& h, uint64_t& stamp_ns, uint64_t& seq) const {
    if (!hdr_) return false;
    w = hdr_->width; h = hdr_->height;
    stamp_ns = hdr_->stamp_ns; seq = hdr_->seq;
    const uint32_t idx = hdr_->ready_idx & 1;
    ptr = reinterpret_cast<const float*>(
      base_ + sizeof(DepthHeader) + size_t(idx) * size_t(kMaxW) * size_t(kMaxH) * sizeof(float));
    return (w>0 && h>0);
  }

private:
  int fd_{-1};
  uint8_t* base_{nullptr};
  DepthHeader* hdr_{nullptr};
  size_t size_{0};
};

} // namespace depth_ipc
