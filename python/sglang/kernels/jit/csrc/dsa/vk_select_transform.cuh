// VestigeKV's selection, forked from dsa/kpool_topk_transform.cuh (whose radix
// core is adapted from tilelang's deepseek_v32 topk_selector).
//
// Same kernel, one rule changed. DSA picks the bin holding the K-th largest
// group score -- a RANK threshold, which cannot be known until every score has
// been histogrammed, which is why that kernel reads the scores two or three
// times and runs one CTA per row. VestigeKV's threshold comes from a REFERENCE
// set instead: the q-quantile of the groups the spectral tier already keeps.
// That is known before any other score is read, so the selection is a single
// compare pass with no histogram and no refinement.
//
// The emitted set is kept ∪ fired ∪ tail, written in DSA's own
// dst_token_indices layout so DSA's attention kernel consumes it unchanged.
//
// OVERFLOW, in three tiers. A rank threshold admits exactly K groups by
// construction; a quantile admits however many clear it, so the budget can be
// exceeded where DSA's rule cannot. The excess is bounded in practice -- firing
// is sparse -- so the candidates are collected first and only then cut:
//
//   n_cand <= budget   emit them all. The common path, one compare pass.
//   n_cand <= kCandMax BITONIC sort the candidates by score in shared memory
//                      and keep the best `budget`. Exact, score-ordered, and it
//                      KEEPS THE KEPT SEED -- which is the whole point, because
//                      tier 1 holds rows that score low and still matter, and
//                      handing the step to DSA's top-K would silently delete
//                      them. No re-read of the scores: the candidates are
//                      already in shared memory from the compare pass.
//   otherwise          DSA's own radix top-k, bit-identical to DSA. Only a
//                      threshold far looser than the quantile can get here.
//
// A radix search is already bisection in base 256 -- one histogram pass resolves
// eight bits of threshold -- so narrowing the threshold in base 2 would be
// strictly worse than either of the above. ovf_counts records which tier ran: a
// silent row drop is the failure mode that once had attention reading the trash
// slot for ten hours while every byte and latency counter reproduced.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#ifndef USE_ROCM
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#endif

namespace sglang {
namespace {

#ifndef C10_LIKELY
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#endif

#ifndef SGL_GROUP_TOPK
#define SGL_GROUP_TOPK 256
#endif

inline constexpr int kGroupTopK = SGL_GROUP_TOPK;
inline constexpr int kThreadsPerBlock = 1024;

inline constexpr std::size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)

// Candidates staged for the bitonic tier, as (float, int) pairs in the dynamic
// shared memory above: 2048 * 8 B = 16 KB of the 32 KB, so it shares the block
// with the radix core's buffers without growing the request. A power of two
// because the bitonic stages require it. Firing is sparse -- about 115 groups of
// 32768 a layer-step -- so 2048 is four times the whole budget and reaching it
// means the threshold is wrong, not merely loose, which is what tier 3 is for.
inline constexpr int kCandMax = 2048;

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B] or nullptr
  int32_t* __restrict__ indices;           // unused here (kept for layout parity)
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int K>
__device__ void
fast_topk_cuda_tl_impl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // We assume length > K here, or it will crash
  int topk = K;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[K - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ int32_t transform_kpool_token(
    int32_t raw_token,
    const int32_t* __restrict__ page_table_entry,
    const int32_t* __restrict__ topk_indices_offset,
    int32_t offset) {
  if (page_table_entry != nullptr) {
    return page_table_entry[raw_token];
  }
  if (topk_indices_offset != nullptr) {
    return raw_token + offset;
  }
  return raw_token;
}

template <int K>
__global__ __launch_bounds__(kThreadsPerBlock) void vk_select_transform_kernel(
    const __grid_constant__ FastTopKParams params,
    int32_t* __restrict__ dst_token_indices,
    const int64_t dst_stride,
    const int32_t pool_size,
    const int32_t token_topk,
    const int32_t out_cols,
    const int32_t* __restrict__ page_table,
    const int64_t page_table_stride,
    const int32_t* __restrict__ page_table_row_index,
    const int32_t* __restrict__ topk_indices_offset,
    const int32_t* __restrict__ seq_lens,
    const float* __restrict__ thresholds,
    const int32_t* __restrict__ kept_groups,
    const int64_t kept_stride,
    const int32_t* __restrict__ kept_lens,
    const int8_t* __restrict__ kept_mask,
    const int64_t kept_mask_stride,
    int32_t* __restrict__ ovf_counts) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto score = input + bid * input_stride;
  const auto dst = dst_token_indices + bid * dst_stride;
  const auto page_table_row = page_table_row_index == nullptr ? bid : static_cast<uint64_t>(page_table_row_index[bid]);
  const auto page_table_entry = page_table == nullptr ? nullptr : page_table + page_table_row * page_table_stride;
  const auto offset = topk_indices_offset == nullptr ? 0 : topk_indices_offset[bid];
  const bool append_tail = seq_lens != nullptr;
  const auto full_pool_token_len = length * pool_size;
  const auto history_len = full_pool_token_len < token_topk ? full_pool_token_len : token_topk;
  const auto tail_count = append_tail ? seq_lens[bid] % pool_size : 0;

  if (length <= K) {
    for (int col = tid; col < out_cols; col += kThreadsPerBlock) {
      if (col < history_len) {
        const auto group_rank = col / pool_size;
        const auto slot = col % pool_size;
        const auto raw_token = group_rank * pool_size + slot;
        dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
      } else if (append_tail && col < history_len + tail_count) {
        const auto raw_token = length * pool_size + (col - history_len);
        dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
      } else {
        dst[col] = -1;
      }
    }
    return;
  }

  __shared__ int s_indices[K];
  __shared__ int s_nsel;    // groups committed to s_indices (the kept seed)
  __shared__ int s_ncand;   // fired candidates collected
  __shared__ int s_spill;   // candidates past kCandMax: the radix tier
  if (tid == 0) {
    s_nsel = 0;
    s_ncand = 0;
    s_spill = 0;
  }
  __syncthreads();

  // The candidate staging area aliases the dynamic shared memory the radix core
  // uses for its own refinement buffers. Safe because the two are never live at
  // once: the radix tier runs only after the candidates are abandoned, with a
  // barrier between.
  extern __shared__ int s_dynamic[];
  float* cand_score = reinterpret_cast<float*>(s_dynamic);
  int* cand_idx = reinterpret_cast<int*>(s_dynamic) + kCandMax;

  // Seed with the groups the spectral tier keeps. These are attended whether or
  // not they clear the threshold -- the threshold is a quantile OF them, so by
  // construction most do not, and dropping them would silently delete tier 1.
  const auto klen = kept_lens == nullptr ? 0 : kept_lens[bid];
  const auto kept_row = kept_groups + bid * kept_stride;
  for (int i = tid; i < klen; i += kThreadsPerBlock) {
    const auto g = kept_row[i];
    if (g >= 0 && g < length) {
      const auto pos = ::atomicAdd(&s_nsel, 1);
      if (pos < K) s_indices[pos] = g;
    }
  }
  __syncthreads();
  const auto n_kept = s_nsel < K ? s_nsel : K;
  const auto budget = K - n_kept;

  // One compare pass. No histogram, no cumsum, no refinement: the threshold is
  // a float and the comparison is exact, where DSA's bin-edge threshold has to
  // be refined inside the tying bin. kept_mask excludes the seeded groups so a
  // kept group that also clears the threshold is not collected twice.
  const auto thr = thresholds[bid];
  const auto kmask = kept_mask == nullptr ? nullptr : kept_mask + bid * kept_mask_stride;
  for (int idx = tid; idx < length; idx += kThreadsPerBlock) {
    if (kmask != nullptr && kmask[idx] != 0) continue;
    const auto sc = score[idx + row_start];
    if (sc > thr) {
      const auto pos = ::atomicAdd(&s_ncand, 1);
      if (pos < kCandMax) {
        cand_score[pos] = sc;
        cand_idx[pos] = idx;
      } else {
        ::atomicAdd(&s_spill, 1);
      }
    }
  }
  __syncthreads();

  const auto n_cand = s_ncand;
  const bool fits = n_cand <= budget;
  const bool sortable = !fits && s_spill == 0;

  if (!fits && s_spill != 0) {
    // Tier 3: more candidates than the staging area holds. Be DSA exactly.
    if (tid == 0 && ovf_counts != nullptr) ::atomicAdd(ovf_counts + bid, n_cand - budget);
    __syncthreads();
    fast_topk_cuda_tl_impl<K>(score, s_indices, row_start, length);
    __syncthreads();
    if (tid == 0) s_nsel = K;
    __syncthreads();
  } else {
    if (sortable) {
      // Tier 2: bitonic sort the candidates by score, descending, padding the
      // tail with -inf so the invalid entries sink. kCandMax is a power of two,
      // which is what lets the stage/substage loops be unguarded.
      if (tid == 0 && ovf_counts != nullptr) ::atomicAdd(ovf_counts + bid, n_cand - budget);
      for (int i = n_cand + tid; i < kCandMax; i += kThreadsPerBlock) {
        cand_score[i] = -INFINITY;
        cand_idx[i] = -1;
      }
      __syncthreads();
      for (int k = 2; k <= kCandMax; k <<= 1) {
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
          for (int i = tid; i < kCandMax; i += kThreadsPerBlock) {
            const int ixj = i ^ jj;
            if (ixj > i) {
              // Descending overall: the direction flips with bit k of i.
              const bool down = (i & k) != 0;
              const bool swap = down ? (cand_score[i] > cand_score[ixj])
                                     : (cand_score[i] < cand_score[ixj]);
              if (swap) {
                const auto fs = cand_score[i];
                cand_score[i] = cand_score[ixj];
                cand_score[ixj] = fs;
                const auto is_ = cand_idx[i];
                cand_idx[i] = cand_idx[ixj];
                cand_idx[ixj] = is_;
              }
            }
          }
          __syncthreads();
        }
      }
    }
    // Commit: everything when it fits, the best `budget` when it was sorted.
    const auto take = fits ? n_cand : budget;
    for (int i = tid; i < take; i += kThreadsPerBlock) {
      s_indices[n_kept + i] = cand_idx[i];
    }
    __syncthreads();
    if (tid == 0) s_nsel = n_kept + take;
    __syncthreads();
  }

  // What the emit loop below treats as "history": DSA's fixed min(length*pool,
  // token_topk) becomes the selected count, so a step that fires little writes
  // little and the rest of the row is -1.
  const auto n_sel = s_nsel < K ? s_nsel : K;
  const auto sel_len = n_sel * pool_size;
  const auto hist_len = sel_len < history_len ? sel_len : history_len;

  for (int col = tid; col < out_cols; col += kThreadsPerBlock) {
    if (col < hist_len) {
      const auto group_rank = col / pool_size;
      const auto group_id = s_indices[group_rank];
      const auto slot = col % pool_size;
      const auto raw_token = group_id * pool_size + slot;
      dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
    } else if (append_tail && col < hist_len + tail_count) {
      const auto raw_token = length * pool_size + (col - hist_len);
      dst[col] = transform_kpool_token(raw_token, page_table_entry, topk_indices_offset, offset);
    } else {
      dst[col] = -1;
    }
  }
}

template <auto* f, std::size_t kMaxDynamicSMEM>
void setup_kernel_smem_once(host::DebugInfo where = {}) {
  [[maybe_unused]]
  static const auto result = [] {
    const auto fptr = std::bit_cast<const void*>(f);
    return ::cudaFuncSetAttribute(fptr, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
  }();
  host::RuntimeDeviceCheck(result, where);
}

template <typename T>
const T* optional_data_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& opt) {
  if (!opt.has_value()) {
    return nullptr;
  }
  return static_cast<const T*>(opt.value().data_ptr());
}

struct VkSelectTransformKernel {
  static constexpr auto kernel = vk_select_transform_kernel<kGroupTopK>;

  static void transform(
      const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView dst_token_indices,
      const int64_t pool_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> topk_indices_offset_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_starts_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> seq_lens_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table_row_index_opt,
      const tvm::ffi::TensorView thresholds,
      const tvm::ffi::Optional<tvm::ffi::TensorView> kept_groups_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> kept_lens_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> kept_mask_opt,
      const tvm::ffi::Optional<tvm::ffi::TensorView> ovf_counts_opt) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto out_cols_sym = SymbolicSize{"out_cols"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1}).with_strides({S, 1}).with_dtype<float>().with_device(device).verify(score);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B, out_cols_sym}).with_dtype<int32_t>().with_device(device).verify(dst_token_indices);

    RuntimeCheck(pool_size > 1, "pool_size must be > 1, got ", pool_size);
    RuntimeCheck(
        !(page_table_opt.has_value() && topk_indices_offset_opt.has_value()),
        "page_table and topk_indices_offset are mutually exclusive");
    RuntimeCheck(
        !page_table_row_index_opt.has_value() || page_table_opt.has_value(),
        "page_table_row_index requires page_table");

    const auto out_cols = static_cast<int32_t>(out_cols_sym.unwrap());
    const auto tail_cols = seq_lens_opt.has_value() ? static_cast<int32_t>(pool_size) - 1 : 0;
    RuntimeCheck(out_cols > tail_cols, "dst_token_indices columns ", out_cols, " must exceed tail ", tail_cols);
    const auto token_topk = out_cols - tail_cols;
    RuntimeCheck(token_topk % static_cast<int32_t>(pool_size) == 0, "token_topk must be a multiple of pool_size");
    RuntimeCheck(
        token_topk / static_cast<int32_t>(pool_size) == kGroupTopK,
        "this module is built for group_topk=",
        kGroupTopK,
        " but got ",
        token_topk / static_cast<int32_t>(pool_size));

    const auto batch_size = static_cast<uint32_t>(B.unwrap());

    int64_t page_table_stride = 0;
    const int32_t* page_table_ptr = nullptr;
    if (page_table_opt.has_value()) {
      auto P = SymbolicSize{"page_table_stride"};
      if (page_table_row_index_opt.has_value()) {
        auto page_table_rows = SymbolicSize{"page_table_rows"};
        TensorMatcher({page_table_rows, -1})
            .with_strides({P, 1})
            .with_dtype<int32_t>()
            .with_device(device)
            .verify(page_table_opt.value());
      } else {
        TensorMatcher({B, -1}).with_strides({P, 1}).with_dtype<int32_t>().with_device(device).verify(
            page_table_opt.value());
      }
      page_table_ptr = static_cast<const int32_t*>(page_table_opt.value().data_ptr());
      page_table_stride = static_cast<int64_t>(P.unwrap());
    }

    if (topk_indices_offset_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(topk_indices_offset_opt.value());
    }
    if (row_starts_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(row_starts_opt.value());
    }
    if (seq_lens_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(seq_lens_opt.value());
    }
    if (page_table_row_index_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(page_table_row_index_opt.value());
    }

    TensorMatcher({B}).with_dtype<float>().with_device(device).verify(thresholds);
    int64_t kept_stride = 0;
    const int32_t* kept_ptr = nullptr;
    if (kept_groups_opt.has_value()) {
      auto KS = SymbolicSize{"kept_stride"};
      TensorMatcher({B, -1}).with_strides({KS, 1}).with_dtype<int32_t>().with_device(device).verify(
          kept_groups_opt.value());
      kept_ptr = static_cast<const int32_t*>(kept_groups_opt.value().data_ptr());
      kept_stride = static_cast<int64_t>(KS.unwrap());
      RuntimeCheck(
          kept_lens_opt.has_value(), "kept_groups requires kept_lens: the seed count is per row");
      // A seed wider than the budget would drop kept groups with no witness,
      // which is the one thing this kernel must never do quietly.
      RuntimeCheck(
          static_cast<int>(kept_groups_opt.value().shape()[1]) <= kGroupTopK,
          "kept_groups width ",
          kept_groups_opt.value().shape()[1],
          " exceeds the group budget ",
          kGroupTopK);
    }
    int64_t kept_mask_stride = 0;
    const int8_t* kept_mask_ptr = nullptr;
    if (kept_mask_opt.has_value()) {
      auto KM = SymbolicSize{"kept_mask_stride"};
      TensorMatcher({B, -1}).with_strides({KM, 1}).with_dtype<int8_t>().with_device(device).verify(
          kept_mask_opt.value());
      kept_mask_ptr = static_cast<const int8_t*>(kept_mask_opt.value().data_ptr());
      kept_mask_stride = static_cast<int64_t>(KM.unwrap());
    }
    if (kept_lens_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(kept_lens_opt.value());
    }
    if (ovf_counts_opt.has_value()) {
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(ovf_counts_opt.value());
    }

    const auto params = FastTopKParams{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = optional_data_ptr<int32_t>(row_starts_opt),
        .indices = nullptr,
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .input_stride = static_cast<int64_t>(S.unwrap()),
    };

    setup_kernel_smem_once<kernel, kSmem>();
    LaunchKernel(batch_size, kThreadsPerBlock, device.unwrap(), kSmem)(
        kernel,
        params,
        static_cast<int32_t*>(dst_token_indices.data_ptr()),
        static_cast<int64_t>(dst_token_indices.strides()[0]),
        static_cast<int32_t>(pool_size),
        token_topk,
        out_cols,
        page_table_ptr,
        page_table_stride,
        optional_data_ptr<int32_t>(page_table_row_index_opt),
        optional_data_ptr<int32_t>(topk_indices_offset_opt),
        optional_data_ptr<int32_t>(seq_lens_opt),
        static_cast<const float*>(thresholds.data_ptr()),
        kept_ptr,
        kept_stride,
        optional_data_ptr<int32_t>(kept_lens_opt),
        kept_mask_ptr,
        kept_mask_stride,
        ovf_counts_opt.has_value()
            ? static_cast<int32_t*>(ovf_counts_opt.value().data_ptr())
            : nullptr);
  }
};

}  // namespace

}  // namespace sglang
