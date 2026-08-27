// The wire's write side as a stream operation, so a frame costs the departure thread one
// enqueue instead of an interpreter round trip.
//
// The measured case for this file: a queued frame spent 0.95 ms of the pool's serial path
// against 0.16 ms of arithmetic, and the difference was interpreter time -- the launches, the
// event machinery and the sender thread all take turns under one lock. Here the copy is a
// cudaMemcpyAsync into a pinned slab on the CURRENT stream and the socket write runs in a
// cudaLaunchHostFunc callback on the driver's thread: after the enqueue returns, no Python
// runs on this frame's behalf at all.
//
// Rules the callback lives by:
//   - no CUDA calls inside it (the runtime forbids them), so tensors are NOT retained there;
//     the caching allocator's stream ordering is what keeps the source memory valid until the
//     copy has run, which is the same guarantee every kernel launch relies on
//   - the per-fd mutex serialises it against every other writer of that socket, including
//     Python's send_frame, which takes the same mutex through acquire_fd/release_fd
//   - a blocking send() delays later callbacks on the stream by the write's length; frames
//     here are tens of kilobytes against a link measured at 670 MB/s, so that is ~100 us,
//     which is the price of ordering and is paid off the interpreter

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kSlabs = 16;
constexpr size_t kSlabBytes = 1 << 20;  // 1 MiB: the largest frame is windows at 8 rows

struct Sender {
  int fd;
  std::mutex wire;  // one writer at a time, C callback and Python alike
  void* slabs[kSlabs] = {};
  std::atomic<bool> busy[kSlabs] = {};
  std::mutex slab_mx;
  std::condition_variable slab_cv;

  explicit Sender(int fd_) : fd(fd_) {
    for (int i = 0; i < kSlabs; i++) {
      TORCH_CHECK(cudaHostAlloc(&slabs[i], kSlabBytes, cudaHostAllocDefault) == cudaSuccess,
                  "pinned slab allocation failed");
    }
  }

  int try_take_slab() {
    for (int i = 0; i < kSlabs; i++) {
      bool expected = false;
      if (busy[i].compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return i;
      }
    }
    return -1;
  }

  int take_slab() {
    std::unique_lock<std::mutex> lk(slab_mx);
    for (;;) {
      int got = try_take_slab();
      if (got >= 0) return got;
      slab_cv.wait_for(lk, std::chrono::milliseconds(50));
    }
  }

  void give_back(int i) {
    busy[i].store(false, std::memory_order_release);
    slab_cv.notify_one();
  }
};

std::mutex g_registry_mx;
std::unordered_map<int, std::unique_ptr<Sender>> g_senders;

Sender& sender_of(int fd) {
  std::lock_guard<std::mutex> g(g_registry_mx);
  auto it = g_senders.find(fd);
  if (it == g_senders.end()) {
    it = g_senders.emplace(fd, std::make_unique<Sender>(fd)).first;
  }
  return *it->second;
}

struct Pending {
  Sender* owner;
  int slab;
  std::string header;
  size_t payload;
};

void write_all(int fd, const char* p, size_t n) {
  while (n) {
    ssize_t wrote = ::send(fd, p, n, MSG_NOSIGNAL);
    if (wrote < 0) {
      if (errno == EINTR) continue;
      // the reader side notices the broken frame; nothing to raise from here
      std::fprintf(stderr, "afd stream sender: send failed errno=%d\n", errno);
      return;
    }
    p += wrote;
    n -= static_cast<size_t>(wrote);
  }
}

void CUDART_CB on_copied(void* ud) {
  std::unique_ptr<Pending> p(static_cast<Pending*>(ud));
  {
    std::lock_guard<std::mutex> g(p->owner->wire);
    write_all(p->owner->fd, p->header.data(), p->header.size());
    write_all(p->owner->fd, static_cast<const char*>(p->owner->slabs[p->slab]), p->payload);
  }
  p->owner->give_back(p->slab);
}

void send_frame_streamed(int64_t fd, const std::string& header,
                         const std::vector<torch::Tensor>& tensors) {
  Sender& s = sender_of(static_cast<int>(fd));
  size_t total = 0;
  for (const auto& t : tensors) {
    TORCH_CHECK(t.is_cuda(), "the streamed path stages from the GPU; got a CPU tensor");
    TORCH_CHECK(t.is_contiguous(), "the wire copies flat bytes; make the tensor contiguous");
    total += static_cast<size_t>(t.nbytes());
  }
  TORCH_CHECK(total <= kSlabBytes, "frame of ", total, " bytes exceeds the ", kSlabBytes,
              "-byte slab");
  // The common case takes a slab with three atomic reads and never touches the GIL. The
  // first build released it around the wait unconditionally, and REACQUIRING it is a queue
  // behind whichever thread holds it -- measured at 1.7 ms a send against 0.17 for the queued
  // path it replaced. Only an actually-exhausted pool is worth that price.
  int slab = s.try_take_slab();
  if (slab < 0) {
    pybind11::gil_scoped_release release;
    slab = s.take_slab();
  }
  char* base = static_cast<char*>(s.slabs[slab]);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  size_t off = 0;
  for (const auto& t : tensors) {
    TORCH_CHECK(cudaMemcpyAsync(base + off, t.data_ptr(), t.nbytes(),
                                cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                "staging copy failed to enqueue");
    off += static_cast<size_t>(t.nbytes());
  }
  auto* pending = new Pending{&s, slab, header, off};
  TORCH_CHECK(cudaLaunchHostFunc(stream, on_copied, pending) == cudaSuccess,
              "host function failed to enqueue");
}

void acquire_fd(int64_t fd) {
  Sender& s = sender_of(static_cast<int>(fd));
  pybind11::gil_scoped_release release;
  s.wire.lock();
}

void release_fd(int64_t fd) { sender_of(static_cast<int>(fd)).wire.unlock(); }

bool knows_fd(int64_t fd) {
  std::lock_guard<std::mutex> g(g_registry_mx);
  return g_senders.count(static_cast<int>(fd)) > 0;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("send_frame_streamed", &send_frame_streamed,
        "Stage GPU tensors on the current stream and write the frame from a driver thread");
  m.def("acquire_fd", &acquire_fd, "Take the fd's wire mutex (for a Python-side writer)");
  m.def("release_fd", &release_fd, "Release the fd's wire mutex");
  m.def("knows_fd", &knows_fd, "Whether a streamed sender exists for this fd");
}
