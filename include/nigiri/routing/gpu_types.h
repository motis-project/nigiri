#pragma once

#include <cassert>
#include <chrono>
#include <type_traits>
#include <cuda/std/chrono>
#include <cuda/std/utility>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <algorithm>

#include "cista/containers/ptr.h"
#include "cista/containers/vector.h"
#include "cista/reflection/comparable.h"
#include "cista/reflection/printable.h"
#include "cista/verify.h"
#include "geo/latlng.h"

template <typename T, typename Tag>
struct gpu_strong {
  using value_t = T;
  gpu_strong() = default;
  __host__ __device__ explicit gpu_strong(T const& v) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
      : v_{v} {}
  __host__ __device__ explicit gpu_strong(T&& v) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : v_{std::move(v)} {}
  template <typename X>
  __host__ __device__ explicit gpu_strong(X&& x) : v_{static_cast<T>(x)} {
  }

  gpu_strong(gpu_strong&& o) noexcept(
      std::is_nothrow_move_constructible_v<T>) = default;
  gpu_strong& operator=(gpu_strong&& o) noexcept(
      std::is_nothrow_move_constructible_v<T>) = default;

  gpu_strong(gpu_strong const& o) = default;
  gpu_strong& operator=(gpu_strong const& o) = default;

  __host__ __device__ static gpu_strong invalid() {
    return gpu_strong{cuda::std::numeric_limits<T>::max()};
  }

  __host__ __device__ gpu_strong& operator++() {
    ++v_;
    return *this;
  }

  __host__ __device__ gpu_strong operator++(int) {
    auto cpy = *this;
    ++v_;
    return cpy;
  }

  __host__ __device__ gpu_strong& operator--() {
    --v_;
    return *this;
  }

  __host__ __device__ const gpu_strong operator--(int) {
    auto cpy = *this;
    --v_;
    return cpy;
  }

  __host__ __device__ gpu_strong operator+(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ + s.v_)};
  }
  __host__ __device__ gpu_strong operator-(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ - s.v_)};
  }
  __host__ __device__ gpu_strong operator*(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ * s.v_)};
  }
  __host__ __device__ gpu_strong operator/(gpu_strong const& s) const {
    return gpu_strong{static_cast<value_t>(v_ / s.v_)};
  }
  __host__ __device__ gpu_strong operator+(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ + i)};
  }
  __host__ __device__ gpu_strong operator-(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ - i)};
  }
  __host__ __device__ gpu_strong operator*(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ * i)};
  }
  __host__ __device__ gpu_strong operator/(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ / i)};
  }

  __host__ __device__ gpu_strong& operator+=(T const& i) {
    v_ += i;
    return *this;
  }
  __host__ __device__ gpu_strong& operator-=(T const& i) {
    v_ -= i;
    return *this;
  }

  __host__ __device__ gpu_strong operator>>(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ >> i)};
  }
  __host__ __device__ gpu_strong operator<<(T const& i) const {
    return gpu_strong{static_cast<value_t>(v_ << i)};
  }
  __host__ __device__ gpu_strong operator>>(gpu_strong const& o) const { return v_ >> o.v_; }
  __host__ __device__ gpu_strong operator<<(gpu_strong const& o) const { return v_ << o.v_; }

  __host__ __device__ gpu_strong& operator|=(gpu_strong const& o) {
    v_ |= o.v_;
    return *this;
  }
  __host__ __device__ gpu_strong& operator&=(gpu_strong const& o) {
    v_ &= o.v_;
    return *this;
  }

  __host__ __device__ bool operator==(gpu_strong const& o) const { return v_ == o.v_; }
  __host__ __device__ bool operator!=(gpu_strong const& o) const { return v_ != o.v_; }
  __host__ __device__ bool operator<=(gpu_strong const& o) const { return v_ <= o.v_; }
  __host__ __device__ bool operator>=(gpu_strong const& o) const { return v_ >= o.v_; }
  __host__ __device__ bool operator<(gpu_strong const& o) const { return v_ < o.v_; }
  __host__ __device__ bool operator>(gpu_strong const& o) const { return v_ > o.v_; }

  __host__ __device__ bool operator==(T const& o) const { return v_ == o; }
  __host__ __device__ bool operator!=(T const& o) const { return v_ != o; }
  __host__ __device__ bool operator<=(T const& o) const { return v_ <= o; }
  __host__ __device__ bool operator>=(T const& o) const { return v_ >= o; }
  __host__ __device__ bool operator<(T const& o) const { return v_ < o; }
  __host__ __device__ bool operator>(T const& o) const { return v_ > o; }

  __host__ __device__ explicit operator T const&() const& noexcept { return v_; }

  __host__ __device__ friend std::ostream& operator<<(std::ostream& o, gpu_strong const& t) {
    return o << t.v_;
  }

  T v_;
};

namespace cista {

template <std::size_t Size>
struct gpu_bitset {
  using block_t = std::uint64_t;
  static constexpr auto const bits_per_block = sizeof(block_t) * 8U;
  static constexpr auto const num_blocks =
      Size / bits_per_block + (Size % bits_per_block == 0U ? 0U : 1U);

  constexpr gpu_bitset() noexcept = default;
  __host__ __device__ static constexpr gpu_bitset max() {
    gpu_bitset ret;
    for (auto& b : ret.blocks_) {
      b = cuda::std::numeric_limits<block_t>::max();
    }
    return ret;
  }

  __host__ __device__ auto cista_members() noexcept { return std::tie(blocks_); }

  __host__ __device__ constexpr void set(std::size_t const i, bool const val = true) noexcept {
    assert((i / bits_per_block) < num_blocks);
    auto& block = blocks_[i / bits_per_block];
    auto const bit = i % bits_per_block;
    auto const mask = block_t{1U} << bit;
    if (val) {
      block |= mask;
    } else {
      block &= (~block_t{0U} ^ mask);
    }
  }

  void reset() noexcept { blocks_ = {}; }

  bool operator[](std::size_t const i) const noexcept { return test(i); }

  __host__ __device__ std::size_t count() const noexcept {
    std::size_t sum = 0U;
    for (std::size_t i = 0U; i != num_blocks - 1U; ++i) {
      sum += popcount(blocks_[i]);
    }
    return sum + popcount(sanitized_last_block());
  }

  __host__ __device__ bool test(std::size_t const i) const noexcept {
    if (i >= Size) {
      return false;
    }
    auto const block = blocks_[i / bits_per_block];
    auto const bit = (i % bits_per_block);
    return (block & (block_t{1U} << bit)) != 0U;
  }

  __host__ __device__ std::size_t size() const noexcept { return Size; }

  __host__ __device__ bool any() const noexcept {
    for (std::size_t i = 0U; i != num_blocks - 1U; ++i) {
      if (blocks_[i] != 0U) {
        return true;
      }
    }
    return sanitized_last_block() != 0U;
  }

  __host__ __device__ bool none() const noexcept { return !any(); }

  __host__ __device__ block_t sanitized_last_block() const noexcept {
    if constexpr ((Size % bits_per_block) != 0U) {
      return blocks_[num_blocks - 1U] &
             ~((~block_t{0U}) << (Size % bits_per_block));
    } else {
      return blocks_[num_blocks - 1U];
    }
  }

  __host__ __device__ friend bool operator==(gpu_bitset const& a, gpu_bitset const& b) noexcept {
    for (std::size_t i = 0U; i != num_blocks - 1U; ++i) {
      if (a.blocks_[i] != b.blocks_[i]) {
        return false;
      }
    }
    return a.sanitized_last_block() == b.sanitized_last_block();
  }

  __host__ __device__ friend bool operator<(gpu_bitset const& a, gpu_bitset const& b) noexcept {
    auto const a_last = a.sanitized_last_block();
    auto const b_last = b.sanitized_last_block();
    if (a_last < b_last) {
      return true;
    }
    if (b_last < a_last) {
      return false;
    }

    for (int i = num_blocks - 2; i != -1; --i) {
      auto const x = a.blocks_[i];
      auto const y = b.blocks_[i];
      if (x < y) {
        return true;
      }
      if (y < x) {
        return false;
      }
    }

    return false;
  }
  __host__ __device__ friend bool operator!=(gpu_bitset const& a, gpu_bitset const& b) noexcept {
    return !(a == b);
  }

  __host__ __device__ friend bool operator>(gpu_bitset const& a, gpu_bitset const& b) noexcept {
    return b < a;
  }

  __host__ __device__ friend bool operator<=(gpu_bitset const& a, gpu_bitset const& b) noexcept {
    return !(a > b);
  }

  __host__ __device__ friend bool operator>=(gpu_bitset const& a, gpu_bitset const& b) noexcept {
    return !(a < b);
  }

  __host__ __device__ gpu_bitset& operator&=(gpu_bitset const& o) noexcept {
    for (auto i = 0U; i < num_blocks; ++i) {
      blocks_[i] &= o.blocks_[i];
    }
    return *this;
  }

  __host__ __device__ gpu_bitset& operator|=(gpu_bitset const& o) noexcept {
    for (auto i = 0U; i < num_blocks; ++i) {
      blocks_[i] |= o.blocks_[i];
    }
    return *this;
  }

  __host__ __device__ gpu_bitset& operator^=(gpu_bitset const& o) noexcept {
    for (auto i = 0U; i < num_blocks; ++i) {
      blocks_[i] ^= o.blocks_[i];
    }
    return *this;
  }

  __host__ __device__ gpu_bitset operator~() const noexcept {
    auto copy = *this;
    for (auto& b : copy.blocks_) {
      b = ~b;
    }
    return copy;
  }

  __host__ __device__ friend gpu_bitset operator&(gpu_bitset const& lhs, gpu_bitset const& rhs) noexcept {
    auto copy = lhs;
    copy &= rhs;
    return copy;
  }

  __host__ __device__ friend gpu_bitset operator|(gpu_bitset const& lhs, gpu_bitset const& rhs) noexcept {
    auto copy = lhs;
    copy |= rhs;
    return copy;
  }

  __host__ __device__ friend gpu_bitset operator^(gpu_bitset const& lhs, gpu_bitset const& rhs) noexcept {
    auto copy = lhs;
    copy ^= rhs;
    return copy;
  }

  __host__ __device__ gpu_bitset& operator>>=(std::size_t const shift) noexcept {
    if (shift >= Size) {
      reset();
      return *this;
    }

    if constexpr ((Size % bits_per_block) != 0U) {
      blocks_[num_blocks - 1U] = sanitized_last_block();
    }

    if constexpr (num_blocks == 1U) {
      blocks_[0U] >>= shift;
      return *this;
    } else {
      if (shift == 0U) {
        return *this;
      }

      auto const shift_blocks = shift / bits_per_block;
      auto const shift_bits = shift % bits_per_block;
      auto const border = num_blocks - shift_blocks - 1U;

      if (shift_bits == 0U) {
        for (std::size_t i = 0U; i <= border; ++i) {
          blocks_[i] = blocks_[i + shift_blocks];
        }
      } else {
        for (std::size_t i = 0U; i < border; ++i) {
          blocks_[i] =
              (blocks_[i + shift_blocks] >> shift_bits) |
              (blocks_[i + shift_blocks + 1] << (bits_per_block - shift_bits));
        }
        blocks_[border] = (blocks_[num_blocks - 1] >> shift_bits);
      }

      for (auto i = border + 1U; i != num_blocks; ++i) {
        blocks_[i] = 0U;
      }

      return *this;
    }
  }

  __host__ __device__ gpu_bitset& operator<<=(std::size_t const shift) noexcept {
    if (shift >= Size) {
      reset();
      return *this;
    }

    if constexpr (num_blocks == 1U) {
      blocks_[0U] <<= shift;
      return *this;
    } else {
      if (shift == 0U) {
        return *this;
      }

      auto const shift_blocks = shift / bits_per_block;
      auto const shift_bits = shift % bits_per_block;

      if (shift_bits == 0U) {
        for (auto i = std::size_t{num_blocks - 1}; i >= shift_blocks; --i) {
          blocks_[i] = blocks_[i - shift_blocks];
        }
      } else {
        for (auto i = std::size_t{num_blocks - 1}; i != shift_blocks; --i) {
          blocks_[i] =
              (blocks_[i - shift_blocks] << shift_bits) |
              (blocks_[i - shift_blocks - 1U] >> (bits_per_block - shift_bits));
        }
        blocks_[shift_blocks] = blocks_[0U] << shift_bits;
      }

      for (auto i = 0U; i != shift_blocks; ++i) {
        blocks_[i] = 0U;
      }

      return *this;
    }
  }

  __host__ __device__ gpu_bitset operator>>(std::size_t const i) const noexcept {
    auto copy = *this;
    copy >>= i;
    return copy;
  }

  __host__ __device__ gpu_bitset operator<<(std::size_t const i) const noexcept {
    auto copy = *this;
    copy <<= i;
    return copy;
  }

  cuda::std::array<block_t, num_blocks> blocks_{};
};

}  // namespace cista


template <typename T>
struct gpu_base_type {
  using type = T;
};
template <typename T, typename Tag>
struct gpu_base_type<gpu_strong<T, Tag>> {
  using type = T;
};
template <typename T>
using gpu_base_t = typename gpu_base_type<T>::type;

template <typename T, typename Tag>
__host__ __device__ inline const typename gpu_strong<T, Tag>::value_t  gpu_to_idx(const gpu_strong<T, Tag>& s) {
  return s.v_;
}
template <typename T, typename Tag>
__host__ __device__ inline typename gpu_strong<T, Tag>::value_t  gpu_to_idx(gpu_strong<T, Tag>& s) {
  return s.v_;
}
template <typename T>
__host__ __device__ T gpu_to_idx(T const& t) {
  return t;
}

using gpu_delta_t = int16_t;
using gpu_clasz_mask_t = std::uint16_t;
using gpu_location_idx_t = gpu_strong<std::uint32_t, struct _location_idx>;
using gpu_value_type = gpu_location_idx_t::value_t;
using gpu_bitfield_idx_t = gpu_strong<std::uint32_t, struct _bitfield_idx>;
using gpu_route_idx_t = gpu_strong<std::uint32_t, struct _route_idx>;
using gpu_transport_idx_t = gpu_strong<std::uint32_t, struct _transport_idx>;
using gpu_day_idx_t = gpu_strong<std::uint16_t, struct _day_idx>;
template <size_t Size>
using gpu_bitset = cista::gpu_bitset<Size>;
constexpr auto const kMaxDays = 512;
using gpu_bitfield = gpu_bitset<kMaxDays>;
using gpu_clasz_mask_t = std::uint16_t;
using gpu_profile_idx_t = std::uint8_t;
using gpu_stop_idx_t = std::uint16_t;
using i16_minutes = cuda::std::chrono::duration<std::int16_t, cuda::std::ratio<60>>;
using gpu_duration_t = i16_minutes;
using gpu_minutes_after_midnight_t = gpu_duration_t;

enum class gpu_clasz : std::uint8_t {
  kAir = 0,
  kHighSpeed = 1,
  kLongDistance = 2,
  kCoach = 3,
  kNight = 4,
  kRegionalFast = 5,
  kRegional = 6,
  kMetro = 7,
  kSubway = 8,
  kTram = 9,
  kBus = 10,
  kShip = 11,
  kOther = 12,
  kNumClasses
};

template <typename R1, typename R2>
using gpu_ratio_multiply = decltype(cuda::std::ratio_multiply<R1, R2>{});
using gpu_days = cuda::std::chrono::duration<int, gpu_ratio_multiply<cuda::std::ratio<24>, cuda::std::chrono::hours::period>>;
enum class gpu_event_type { kArr, kDep };
enum class gpu_direction { kForward, kBackward };

template <typename T>
using ptr = T*;
using gpu_i32_minutes = cuda::std::chrono::duration<int32_t, cuda::std::ratio<60>>;
using gpu_u8_minutes = cuda::std::chrono::duration<std::uint8_t, cuda::std::ratio<60>>;

template <typename D>
using gpu_sys_time = cuda::std::chrono::time_point<cuda::std::chrono::system_clock, D>;
using gpu_unixtime_t = gpu_sys_time<gpu_i32_minutes>;
using gpu_sys_days    = gpu_sys_time<gpu_days>;

__host__ __device__ inline std::ostream& operator<<(std::ostream& out, const gpu_duration_t& duration) {
  return out << duration.count() << " minute(s)";
}
struct gpu_delta{
  std::uint16_t days_ : 5;
  std::uint16_t mam_ : 11;
  __host__ __device__ bool operator== (gpu_delta const& a) const{
    return (a.days_== this->days_ && a.mam_==this->mam_);
  }
  __host__ __device__ bool operator!= (gpu_delta const& a) const{
    return !(operator==(a));
  }
#ifdef NIGIRI_CUDA

  __host__ __device__ std::int16_t count() const { return days_ * 1440U + mam_; }
#endif
};
template<typename T>
__host__ __device__ T gpu_clamp(T value, T low, T high) {
  return (value < low) ? low : (value > high) ? high : value;
}
#ifdef NIGIRI_CUDA
template <typename T>
__host__ __device__ inline gpu_delta_t gpu_clamp(T t) {
  return static_cast<gpu_delta_t>(
      gpu_clamp(t, static_cast<int>(cuda::std::numeric_limits<gpu_delta_t>::min()),
                 static_cast<int>(cuda::std::numeric_limits<gpu_delta_t>::max())));
}

__host__ __device__ inline gpu_delta_t unix_to_gpu_delta(gpu_sys_days const base, gpu_unixtime_t const t) {
  return gpu_clamp(
      (t - cuda::std::chrono::time_point_cast<gpu_unixtime_t::duration>(base)).count());
}
__host__ __device__ inline cuda::std::pair<gpu_day_idx_t, gpu_minutes_after_midnight_t> gpu_split_day_mam(
    gpu_day_idx_t const base, gpu_delta_t const x) {
  assert(x != cuda::std::numeric_limits<gpu_delta_t>::min());
  assert(x != cuda::std::numeric_limits<gpu_delta_t>::max());
  if (x < 0) {
    auto const t = -x / 1440 + 1;
    auto const min = x + (t * 1440);
    return {static_cast<gpu_day_idx_t>(static_cast<int>(gpu_to_idx(base)) - t),
            gpu_minutes_after_midnight_t{min}};
  } else {
    return {static_cast<gpu_day_idx_t>(static_cast<int>(gpu_to_idx(base)) + x / 1440),
            gpu_minutes_after_midnight_t{x % 1440}};
  }
}
#else

template <typename T>
inline gpu_delta_t gpu_clamp(T t) {
  return static_cast<gpu_delta_t>(
      gpu_clamp(t, static_cast<int>(std::numeric_limits<gpu_delta_t>::min()),
                 static_cast<int>(std::numeric_limits<gpu_delta_t>::max())));
}
inline gpu_delta_t unix_to_gpu_delta(gpu_sys_days const base, gpu_unixtime_t const t) {
  return gpu_clamp(
      (t - cuda::std::chrono::time_point_cast<gpu_unixtime_t::duration>(base)).count());
}
#endif
template <gpu_direction SearchDir>
inline constexpr auto const kInvalidGpuDelta =
    SearchDir == gpu_direction::kForward ? std::numeric_limits<gpu_delta_t>::max()
                                         : std::numeric_limits<gpu_delta_t>::min();
inline gpu_unixtime_t gpu_delta_to_unix(gpu_sys_days const base, gpu_delta_t const d) {
  return cuda::std::chrono::time_point_cast<gpu_unixtime_t::duration>(base) +
         d * gpu_unixtime_t::duration{1};
}

namespace cista {
template <typename Key, typename DataVec, typename IndexVec>
struct basic_gpu_vecvec {
  using data_value_type = typename DataVec::value_type;
  using index_value_type = typename IndexVec::value_type;
#ifdef NIGIRI_CUDA
  struct bucket final {
    using value_type = data_value_type;
    using iterator = typename DataVec::iterator;
    using const_iterator = typename DataVec::iterator;

    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::add_pointer_t<value_type>;
    using reference = bucket;

    __host__ __device__ bucket(basic_gpu_vecvec* map, index_value_type const i)
        : map_{map}, i_{gpu_to_idx(i)} {}

    __host__ __device__ friend data_value_type* data(bucket b) { return &b[0]; }
    __host__ __device__ friend index_value_type size(bucket b) {
      return b.size();
    }

    __host__ __device__ data_value_type const* data() const {
      return empty() ? nullptr : &front();
    }

    template <typename T = std::decay_t<data_value_type>,
              typename = std::enable_if_t<std::is_same_v<T, char>>>
    __host__ __device__ std::string_view view() const {
      return std::string_view{begin(), size()};
    }

    __host__ __device__ const value_type& front() const{
      assert(!empty());
      return operator[](0);
    }

    __host__ __device__ value_type& back() {
      assert(!empty());
      return operator[](size() - 1U);
    }

    __host__ __device__ bool empty() const { return begin() == end(); }

    template <typename Args>
    __host__ __device__ void push_back(Args&& args) {
      map_->data_.insert(std::next(std::begin(map_->data_), bucket_end_idx()),
                         std::forward<Args>(args));
      for (auto i = i_ + 1; i != map_->bucket_starts_.size(); ++i) {
        ++map_->bucket_starts_[i];
      }
    }

    __host__ __device__ value_type& operator[](std::size_t const i) {
      assert(is_inside_bucket(i));
      return map_->data_[gpu_to_idx(map_->bucket_starts_[i_] + i)];
    }

    __host__ __device__ value_type const& operator[](
        std::size_t const i) const {
      assert(is_inside_bucket(i));
      return map_->data_[gpu_to_idx(map_->bucket_starts_[i_] + i)];
    }

    __host__ __device__ value_type const& at(std::size_t const i) const {
      verify(i < size(), "bucket::at: index out of range");
      return *(begin() + i);
    }

    __host__ __device__ value_type& at(std::size_t const i) {
      verify(i < size(), "bucket::at: index out of range");
      return *(begin() + i);
    }

    __host__ __device__ std::size_t size() const {
      return bucket_end_idx() - bucket_begin_idx();
    }
    __host__ __device__ iterator begin() {
      return map_->data_.begin() + bucket_begin_idx();
    }
    __host__ __device__ iterator end() {
      return map_->data_.begin() + bucket_end_idx();
    }
    __host__ __device__ const_iterator begin() const {
      return map_->data_.begin() + bucket_begin_idx();
    }
    __host__ __device__ const_iterator end() const {
      return map_->data_.begin() + bucket_end_idx();
    }
    __host__ __device__ friend iterator begin(bucket const& b) {
      return b.begin();
    }
    __host__ __device__ friend iterator end(bucket const& b) { return b.end(); }
    __host__ __device__ friend iterator begin(bucket& b) { return b.begin(); }
    __host__ __device__ friend iterator end(bucket& b) { return b.end(); }

    __host__ __device__ friend bool operator==(bucket const& a,
                                               bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ == b.i_;
    }
    __host__ __device__ friend bool operator!=(bucket const& a,
                                               bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ != b.i_;
    }
    __host__ __device__ bucket& operator++() {
      ++i_;
      return *this;
    }
    __host__ __device__ bucket& operator--() {
      --i_;
      return *this;
    }
    __host__ __device__ bucket operator*() const { return *this; }
    __host__ __device__ bucket& operator+=(difference_type const n) {
      i_ += n;
      return *this;
    }
    __host__ __device__ bucket& operator-=(difference_type const n) {
      i_ -= n;
      return *this;
    }
    __host__ __device__ bucket operator+(difference_type const n) const {
      auto tmp = *this;
      tmp += n;
      return tmp;
    }
    __host__ __device__ bucket operator-(difference_type const n) const {
      auto tmp = *this;
      tmp -= n;
      return tmp;
    }
    __host__ __device__ friend difference_type operator-(bucket const& a,
                                                         bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ - b.i_;
    }

  private:
    __host__ __device__ index_value_type bucket_begin_idx() const {
      return gpu_to_idx(map_->bucket_starts_[i_]);
    }
    __host__ __device__ index_value_type bucket_end_idx() const {
      return gpu_to_idx(map_->bucket_starts_[i_ + 1U]);
    }
    __host__ __device__ bool is_inside_bucket(std::size_t const i) const {
      return bucket_begin_idx() + i < bucket_end_idx();
    }

    basic_gpu_vecvec* map_;
    index_value_type i_;
  };

  struct const_bucket final {
    using value_type = data_value_type;
    using iterator = typename DataVec::const_iterator;
    using const_iterator = typename DataVec::const_iterator;

    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using pointer = std::add_pointer_t<value_type>;
    using reference = std::add_lvalue_reference<value_type>;

    __host__ __device__ const_bucket(basic_gpu_vecvec const* map,
                                     index_value_type const i)
        : map_{map}, i_{gpu_to_idx(i)} {}

    __host__ __device__ friend data_value_type const* data(const_bucket b) {
      return b.data();
    }
    __host__ __device__ friend index_value_type size(const_bucket b) {
      return b.size();
    }

    __host__ __device__ data_value_type const* data() const {
      return empty() ? nullptr : &front();
    }

    template <typename T = std::decay_t<data_value_type>,
              typename = std::enable_if_t<std::is_same_v<T, char>>>
    __host__ __device__ std::string_view view() const {
      return std::string_view{begin(), size()};
    }

    __host__ __device__ value_type const& front() const {
      assert(!empty());
      return operator[](0);
    }

    __host__ __device__ value_type const& back() const {
      assert(!empty());
      return operator[](size() - 1U);
    }

    __host__ __device__ bool empty() const { return begin() == end(); }

    __host__ __device__ value_type const& at(std::size_t const i) const {
      verify(i < size(), "bucket::at: index out of range");
      return *(begin() + i);
    }

    __host__ __device__ value_type const& operator[](
        std::size_t const i) const {
      assert(is_inside_bucket(i));
      return map_->data_[map_->bucket_starts_[i_] + i];
    }

    __host__ __device__ index_value_type size() const {
      return bucket_end_idx() - bucket_begin_idx();
    }
    __host__ __device__ const_iterator begin() const {
      return map_->data_.begin() + bucket_begin_idx();
    }
    __host__ __device__ const_iterator end() const {
      return map_->data_.begin() + bucket_end_idx();
    }
    __host__ __device__ friend const_iterator begin(const_bucket const& b) {
      return b.begin();
    }
    __host__ __device__ friend const_iterator end(const_bucket const& b) {
      return b.end();
    }

    __host__ __device__ friend bool operator==(const_bucket const& a,
                                               const_bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ == b.i_;
    }
    __host__ __device__ friend bool operator!=(const_bucket const& a,
                                               const_bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ != b.i_;
    }
    __host__ __device__ const_bucket& operator++() {
      ++i_;
      return *this;
    }
    __host__ __device__ const_bucket& operator--() {
      --i_;
      return *this;
    }
    __host__ __device__ const_bucket operator*() const { return *this; }
    __host__ __device__ const_bucket& operator+=(difference_type const n) {
      i_ += n;
      return *this;
    }
    __host__ __device__ const_bucket& operator-=(difference_type const n) {
      i_ -= n;
      return *this;
    }
    __host__ __device__ const_bucket operator+(difference_type const n) const {
      auto tmp = *this;
      tmp += n;
      return tmp;
    }
    __host__ __device__ const_bucket operator-(difference_type const n) const {
      auto tmp = *this;
      tmp -= n;
      return tmp;
    }
    __host__ __device__ friend difference_type operator-(
        const_bucket const& a, const_bucket const& b) {
      assert(a.map_ == b.map_);
      return a.i_ - b.i_;
    }

  private:
    __host__ __device__ std::size_t bucket_begin_idx() const {
      return gpu_to_idx(map_->bucket_starts_[i_]);
    }
    __host__ __device__ std::size_t bucket_end_idx() const {
      return gpu_to_idx(map_->bucket_starts_[i_ + 1]);
    }
    __host__ __device__ bool is_inside_bucket(std::size_t const i) const {
      return bucket_begin_idx() + i < bucket_end_idx();
    }

    std::size_t i_;
    basic_gpu_vecvec const* map_;
  };

  using value_type = bucket;
  using iterator = bucket;
  using const_iterator = const_bucket;

  __host__ __device__ bucket operator[](Key const i) {
    return {this, gpu_to_idx(i)};
  }
  __host__ __device__ const_bucket operator[](Key const i) const {
    return {this, gpu_to_idx(i)};
  }

  __host__ __device__ const_bucket at(Key const i) const {
    verify(gpu_to_idx(i) < bucket_starts_.size(),
           "basic_gpu_vecvec::at: index out of range");
    return {this, gpu_to_idx(i)};
  }

  __host__ __device__ bucket at(Key const i) {
    verify(gpu_to_idx(i) < bucket_starts_.size(),
           "basic_gpu_vecvec::at: index out of range");
    return {this, gpu_to_idx(i)};
  }

  __host__ __device__ bucket front() { return at(Key{0}); }
  __host__ __device__ bucket back() { return at(Key{size() - 1}); }

  __host__ __device__ const_bucket front() const { return at(Key{0}); }
  __host__ __device__ const_bucket back() const { return at(Key{size() - 1}); }

  __host__ __device__ index_value_type size() const {
    return empty() ? 0U : bucket_starts_.size() - 1;
  }
  __host__ __device__ bool empty() const { return bucket_starts_.empty(); }

  template <typename Container,
            typename = std::enable_if_t<std::is_convertible_v<
                decltype(*std::declval<Container>().begin()),
                data_value_type>>>
  __host__ __device__ void emplace_back(Container&& bucket) {
    if (bucket_starts_.empty()) {
      bucket_starts_.emplace_back(index_value_type{0U});
    }
    bucket_starts_.emplace_back(
        static_cast<index_value_type>(data_.size() + bucket.size()));
    data_.insert(std::end(data_),  //
                 std::make_move_iterator(std::begin(bucket)),
                 std::make_move_iterator(std::end(bucket)));
  }

  __host__ __device__ bucket add_back_sized(std::size_t const bucket_size) {
    if (bucket_starts_.empty()) {
      bucket_starts_.emplace_back(index_value_type{0U});
    }
    data_.resize(data_.size() + bucket_size);
    bucket_starts_.emplace_back(static_cast<index_value_type>(data_.size()));
    return at(Key{size() - 1U});
  }

  template <typename X>
  std::enable_if_t<std::is_convertible_v<std::decay_t<X>, data_value_type>>
      __host__ __device__ emplace_back(std::initializer_list<X>&& x) {
    if (bucket_starts_.empty()) {
      bucket_starts_.emplace_back(index_value_type{0U});
    }
    bucket_starts_.emplace_back(
        static_cast<index_value_type>(data_.size() + x.size()));
    data_.insert(std::end(data_),  //
                 std::make_move_iterator(std::begin(x)),
                 std::make_move_iterator(std::end(x)));
  }

  template <typename T = data_value_type,
            typename = std::enable_if_t<std::is_convertible_v<T, char const>>>
  __host__ __device__ void emplace_back(char const* s) {
    return emplace_back(std::string_view{s});
  }

  __host__ __device__ void resize(std::size_t const new_size) {
    auto const old_size = bucket_starts_.size();
    bucket_starts_.resize(
        static_cast<typename IndexVec::size_type>(new_size + 1U));
    for (auto i = old_size; i < new_size + 1U; ++i) {
      bucket_starts_[i] = data_.size();
    }
  }

  __host__ __device__ bucket begin() { return bucket{this, 0U}; }
  __host__ __device__ bucket end() { return bucket{this, size()}; }
  __host__ __device__ const_bucket begin() const {
    return const_bucket{this, 0U};
  }
  __host__ __device__ const_bucket end() const {
    return const_bucket{this, size()};
  }

  __host__ __device__ friend bucket begin(basic_gpu_vecvec& m) {
    return m.begin();
  }
  __host__ __device__ friend bucket end(basic_gpu_vecvec& m) { return m.end(); }
  __host__ __device__ friend const_bucket begin(basic_gpu_vecvec const& m) {
    return m.begin();
  }
  __host__ __device__ friend const_bucket end(basic_gpu_vecvec const& m) {
    return m.end();
  }
#endif
  DataVec data_;
  IndexVec bucket_starts_;
};

} //namespace cista

struct gpu_transport {
  CISTA_FRIEND_COMPARABLE(gpu_transport)
  CISTA_PRINTABLE(gpu_transport, "idx", "day")

  __host__ __device__ static gpu_transport invalid() noexcept {
    return gpu_transport{};
  }
  __host__ __device__ bool is_valid() const {
    return day_ != gpu_day_idx_t::invalid();
  }
  gpu_transport_idx_t t_idx_{gpu_transport_idx_t::invalid()};
  gpu_day_idx_t day_{gpu_day_idx_t::invalid()};
};


namespace cista {

template <typename T, template <typename> typename Ptr,
          bool IndexPointers = false, typename TemplateSizeType = std::uint32_t,
          class Allocator = allocator<T, Ptr>>
struct gpu_basic_vector {
  using size_type = gpu_base_t<TemplateSizeType>;
  using difference_type = std::ptrdiff_t;
  using access_type = TemplateSizeType;
  using reference = T&;
  using const_reference = T const&;
  using pointer = Ptr<T>;
  using const_pointer = Ptr<T const>;
  using value_type = T;
  using iterator = T*;
  using const_iterator = T const*;
  using allocator_type = Allocator;

  __host__ __device__ explicit gpu_basic_vector(allocator_type const&) noexcept {}
  gpu_basic_vector() noexcept = default;

  __host__ __device__ explicit gpu_basic_vector(size_type const size, T init = T{},
                        Allocator const& alloc = Allocator{}) {
    CISTA_UNUSED_PARAM(alloc)
    resize(size, std::move(init));
  }

  __host__ __device__ gpu_basic_vector(std::initializer_list<T> init,
               Allocator const& alloc = Allocator{}) {
    CISTA_UNUSED_PARAM(alloc)
    set(init.begin(), init.end());
  }

  template <typename It>
  __host__ __device__ gpu_basic_vector(It begin_it, It end_it) {
    set(begin_it, end_it);
  }

  __host__ __device__ gpu_basic_vector(gpu_basic_vector&& o, Allocator const& alloc = Allocator{}) noexcept
      : el_(o.el_),
        used_size_(o.used_size_),
        allocated_size_(o.allocated_size_),
        self_allocated_(o.self_allocated_) {
    CISTA_UNUSED_PARAM(alloc)
    o.reset();
  }

  __host__ __device__ gpu_basic_vector(gpu_basic_vector const& o, Allocator const& alloc = Allocator{}) {
    CISTA_UNUSED_PARAM(alloc)
    set(o);
  }

  __host__ __device__ gpu_basic_vector& operator=(gpu_basic_vector&& arr) noexcept {
    deallocate();

    el_ = arr.el_;
    used_size_ = arr.used_size_;
    self_allocated_ = arr.self_allocated_;
    allocated_size_ = arr.allocated_size_;

    arr.reset();
    return *this;
  }

  __host__ __device__ gpu_basic_vector& operator=(gpu_basic_vector const& arr) {
    if (&arr != this) {
      set(arr);
    }
    return *this;
  }

  __host__ __device__ ~gpu_basic_vector() { deallocate(); }

  __host__ __device__ void deallocate() {
    if (!self_allocated_ || el_ == nullptr) {
      return;
    }

    for (auto& el : *this) {
      el.~T();
    }

    std::free(el_);  // NOLINT
    reset();
  }

  __host__ __device__ allocator_type get_allocator() const noexcept { return {}; }

  __host__ __device__ T const* data() const noexcept { return begin(); }
  __host__ __device__ T* data() noexcept { return begin(); }
  __host__ __device__ T const* begin() const noexcept { return el_; }
  __host__ __device__ T const* end() const noexcept { return el_ + used_size_; }  // NOLINT
  __host__ __device__ T const* cbegin() const noexcept { return el_; }
  __host__ __device__ T const* cend() const noexcept { return el_ + used_size_; }  // NOLINT
  __host__ __device__ T* begin() noexcept { return el_; }
  __host__ __device__ T* end() noexcept { return el_ + used_size_; }  // NOLINT

  __host__ __device__ std::reverse_iterator<T const*> rbegin() const {
    return std::reverse_iterator<T*>(el_ + size());  // NOLINT
  }
  __host__ __device__ std::reverse_iterator<T const*> rend() const {
    return std::reverse_iterator<T*>(el_);
  }
  __host__ __device__ std::reverse_iterator<T*> rbegin() {
    return std::reverse_iterator<T*>(el_ + size());  // NOLINT
  }
  __host__ __device__ std::reverse_iterator<T*> rend() { return std::reverse_iterator<T*>(el_); }

  __host__ __device__ friend T const* begin(gpu_basic_vector const& a) noexcept { return a.begin(); }
  __host__ __device__ friend T const* end(gpu_basic_vector const& a) noexcept { return a.end(); }

  __host__ __device__ friend T* begin(gpu_basic_vector& a) noexcept { return a.begin(); }
  __host__ __device__ friend T* end(gpu_basic_vector& a) noexcept { return a.end(); }

  __host__ __device__ T const& operator[](access_type const index) const noexcept {
    assert(el_ != nullptr && index < used_size_);
    return el_[gpu_to_idx(index)];
  }

  __host__ __device__ T& operator[](access_type const index) noexcept {
    assert(el_ != nullptr);
    assert(index < used_size_);
    return el_[gpu_to_idx(index)];
  }

  T& at(access_type const index) {
    if (index >= used_size_) {
      throw std::out_of_range{"vector::at(): invalid index"};
    }
    return (*this)[index];
  }

  __host__ __device__ T const& at(access_type const index) const {
    return const_cast<gpu_basic_vector*>(this)->at(index);
  }

  __host__ __device__ T const& back() const noexcept { return ptr_cast(el_)[used_size_ - 1]; }
  __host__ __device__ T& back() noexcept { return ptr_cast(el_)[used_size_ - 1]; }

  __host__ __device__ T& front() noexcept { return ptr_cast(el_)[0]; }
  __host__ __device__ T const& front() const noexcept { return ptr_cast(el_)[0]; }

  __host__ __device__ size_type size() const noexcept { return used_size_; }
  __host__ __device__ bool empty() const noexcept { return size() == 0U; }

  template <typename It>
  __host__ __device__ void set(It begin_it, It end_it) {
    auto const range_size = std::distance(begin_it, end_it);
#ifndef __CUDA_ARCH__
    verify(
        range_size >= 0 && range_size <= std::numeric_limits<size_type>::max(),
        "cista::vector::set: invalid range");
#endif
    reserve(static_cast<size_type>(range_size));

    auto copy_source = begin_it;
    auto copy_target = el_;
    for (; copy_source != end_it; ++copy_source, ++copy_target) {
      new (copy_target) T{std::forward<decltype(*copy_source)>(*copy_source)};
    }

    used_size_ = static_cast<size_type>(range_size);
  }

  __host__ __device__ void set(gpu_basic_vector const& arr) {
    if constexpr (std::is_trivially_copyable_v<T>) {
      if (arr.used_size_ != 0U) {
        reserve(arr.used_size_);
        #ifdef __CUDA_ARCH__
          cudaMemcpy(data(), arr.data(), arr.used_size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        #else
          std::memcpy(data(), arr.data(), arr.used_size_ * sizeof(T));
        #endif
      }
      used_size_ = arr.used_size_;
    } else {
      set(std::begin(arr), std::end(arr));
    }
  }

  __host__ __device__ friend std::ostream& operator<<(std::ostream& out, gpu_basic_vector const& v) {
    out << "[\n  ";
    auto first = true;
    for (auto const& e : v) {
      if (!first) {
        out << ",\n  ";
      }
      out << e;
      first = false;
    }
    return out << "\n]";
  }


  template <class It>
  __host__ __device__ T* insert(T* pos, It first, It last) {
    return insert(pos, first, last,
                  typename std::iterator_traits<It>::iterator_category());
  }

  __host__ __device__ void pop_back() noexcept(noexcept(std::declval<T>().~T())) {
    --used_size_;
    el_[used_size_].~T();
  }

  __host__ __device__ void clear() {
    for (auto& el : *this) {
      el.~T();
    }
    used_size_ = 0;
  }
  template <typename TemplateSizeType>
  __host__ __device__ TemplateSizeType gpu_next_power_of_two(TemplateSizeType n) noexcept {
    --n;
    n |= n >> 1U;
    n |= n >> 2U;
    n |= n >> 4U;
    n |= n >> 8U;
    n |= n >> 16U;
    if constexpr (sizeof(TemplateSizeType) > 32U) {
      n |= n >> 32U;
    }
    ++n;
    return n;
  }

  __host__ __device__ void reserve(size_type new_size) {
    new_size = std::max(allocated_size_, new_size);

    if (allocated_size_ >= new_size) {
      return;
    }

    auto next_size = gpu_next_power_of_two(new_size);
    auto num_bytes = static_cast<std::size_t>(next_size) * sizeof(T);
    T* mem_buf;
#ifdef __CUDA_ARCH__
    cudaError_t err = cudaMalloc(&mem_buf, num_bytes);
    if (err != cudaSuccess) {
      printf("GPU allocation failed\n");
      return;
    }
#else
    mem_buf = static_cast<T*>(std::malloc(num_bytes));  // NOLINT
    if (mem_buf == nullptr) {
      throw std::bad_alloc();
    }
#endif
    if (size() != 0) {
      T* move_target = mem_buf;
      for (auto& el : *this) {
        new (move_target++) T(std::move(el));
        el.~T();
      }
    }

    T* free_me = el_;
    el_ = mem_buf;

    if (self_allocated_) {
    #ifdef __CUDA_ARCH__
          cudaFree(free_me);
    #else
          std::free(free_me);  // NOLINT
    #endif
    }

    self_allocated_ = true;
    allocated_size_ = next_size;
  }

  __host__ __device__ T* erase(T* first, T* last) {
    if (first != last) {
      auto const new_end = std::move(last, end(), first);
      for (auto it = new_end; it != end(); ++it) {
        it->~T();
      }
      used_size_ -= static_cast<size_type>(std::distance(new_end, end()));
    }
    return end();
  }

  __host__ __device__ bool contains(T const* el) const noexcept {
    return el >= begin() && el < end();
  }

  __host__ __device__ std::size_t index_of(T const* el) const noexcept {
    assert(contains(el));
    return std::distance(begin(), el);
  }

  __host__ __device__ friend bool operator==(gpu_basic_vector const& a,
                         gpu_basic_vector const& b) noexcept {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  }
  __host__ __device__ friend bool operator!=(gpu_basic_vector const& a,
                         gpu_basic_vector const& b) noexcept {
    return !(a == b);
  }
  __host__ __device__ friend bool operator<(gpu_basic_vector const& a, gpu_basic_vector const& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
  }
  __host__ __device__ friend bool operator>(gpu_basic_vector const& a, gpu_basic_vector const& b) noexcept {
    return b < a;
  }
  __host__ __device__ friend bool operator<=(gpu_basic_vector const& a,
                         gpu_basic_vector const& b) noexcept {
    return !(a > b);
  }
  __host__ __device__ friend bool operator>=(gpu_basic_vector const& a,
                         gpu_basic_vector const& b) noexcept {
    return !(a < b);
  }

  __host__ __device__ void reset() noexcept {
    el_ = nullptr;
    used_size_ = {};
    allocated_size_ = {};
    self_allocated_ = false;
  }
  Ptr<T> el_{nullptr};
  size_type used_size_{0U};
  size_type allocated_size_{0U};
  bool self_allocated_{false};
  std::uint8_t __fill_0__{0U};
  std::uint16_t __fill_1__{0U};
  std::uint32_t __fill_2__{0U};
};

namespace raw {

template <typename T>
using gpu_vector = gpu_basic_vector<T, ptr>;
template <typename K, typename V, typename SizeType = gpu_base_t<K>>
using gpu_vecvec = basic_gpu_vecvec<K, gpu_vector<V>, gpu_vector<SizeType>>;

template <typename Key, typename Value>
using gpu_vector_map = gpu_basic_vector<Value, ptr, false, Key>;

}  // namespace raw
#undef CISTA_TO_VEC

}  // namespace cista

template <typename T>
using gpu_vector = cista::gpu_basic_vector<T, cista::raw::ptr>;

template <typename K, typename V>
using gpu_vector_map = cista::raw::gpu_vector_map<K, V>;

namespace nigiri {

template <typename T>
struct gpu_interval {
#ifdef NIGIRI_CUDA
  struct iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = std::add_pointer_t<value_type>;
    using reference = value_type;
    CISTA_FRIEND_COMPARABLE(iterator);
    __host__ __device__ value_type operator*() const { return t_; }
    __host__ __device__ iterator& operator++() {
      ++t_;
      return *this;
    }
    __host__ __device__ iterator& operator--() {
      --t_;
      return *this;
    }
    __host__ __device__ iterator& operator+=(difference_type const x) {
      t_ += x;
      return *this;
    }
    __host__ __device__ iterator& operator-=(difference_type const x) {
      t_ -= x;
      return *this;
    }
    __host__ __device__ iterator operator+(difference_type const x) const { return *this += x; }
    __host__ __device__ iterator operator-(difference_type const x) const { return *this -= x; }
    __host__ __device__ friend difference_type operator-(iterator const& a, iterator const& b) {
      return static_cast<difference_type>(gpu_to_idx(a.t_) -
                                          gpu_to_idx(b.t_));
    }
    T t_;
  };

  template <typename X>
  __host__ __device__ gpu_interval operator+(X const& x) const {
    return {from_ + x, to_ + x};
  }

  template <typename X>
  __host__ __device__ gpu_interval operator-(X const& x) const {
    return {from_ - x, to_ - x};
  }

  template <typename X>
    requires std::is_convertible_v<T, X>
  __host__ __device__ operator gpu_interval<X>() {
    return {from_, to_};
  }

  __host__ __device__ T clamp(T const x) const { return gpu_clamp(x, from_, to_); }

  __host__ __device__ bool contains(T const t) const { return t >= from_ && t < to_; }

  __host__ __device__ bool overlaps(gpu_interval const& o) const {
    return from_ < o.to_ && to_ > o.from_;
  }

  __host__ __device__ iterator begin() const { return {from_}; }
  __host__ __device__ iterator end() const { return {to_}; }
  __host__ __device__ friend iterator begin(gpu_interval const& r) { return r.begin(); }
  __host__ __device__ friend iterator end(gpu_interval const& r) { return r.end(); }

  __host__ __device__ cuda::std::reverse_iterator<iterator> rbegin() const {
    return cuda::std::reverse_iterator<iterator>{iterator{to_}};
  }
  __host__ __device__ cuda::std::reverse_iterator<iterator> rend() const {
    return cuda::std::reverse_iterator<iterator>{iterator{from_}};
  }
  __host__ __device__ friend cuda::std::reverse_iterator<iterator> rbegin(gpu_interval const& r) {
    return r.begin();
  }
  __host__ __device__ friend cuda::std::reverse_iterator<iterator> rend(gpu_interval const& r) {
    return r.end();
  }

  __host__ __device__ auto size() const { return to_ - from_; }

  __host__ __device__ T operator[](std::size_t const i) const {
    assert(contains(from_ + static_cast<T>(i)));
    return from_ + static_cast<T>(i);
  }

  __host__ __device__ friend std::ostream& operator<<(std::ostream& out, gpu_interval const& i) {
    return out << "[" << i.from_ << ", " << i.to_ << "[";
  }
#else
  struct iterator {
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = std::add_pointer_t<value_type>;
    using reference = value_type;
    CISTA_FRIEND_COMPARABLE(iterator);
    value_type operator*() const { return t_; }
    iterator& operator++() {
      ++t_;
      return *this;
    }
    iterator& operator--() {
      --t_;
      return *this;
    }
    iterator& operator+=(difference_type const x) {
      t_ += x;
      return *this;
    }
    iterator& operator-=(difference_type const x) {
      t_ -= x;
      return *this;
    }
    iterator operator+(difference_type const x) const { return *this += x; }
    iterator operator-(difference_type const x) const { return *this -= x; }
    friend difference_type operator-(iterator const& a, iterator const& b) {
      return static_cast<difference_type>(cista::to_idx(a.t_) -
                                          cista::to_idx(b.t_));
    }
    T t_;
  };

  template <typename X>
  gpu_interval operator+(X const& x) const {
    return {from_ + x, to_ + x};
  }

  template <typename X>
  gpu_interval operator-(X const& x) const {
    return {from_ - x, to_ - x};
  }

  template <typename X>
    requires std::is_convertible_v<T, X>
  operator gpu_interval<X>() {
    return {from_, to_};
  }

  T clamp(T const x) const { return gpu_clamp(x, from_, to_); }

  bool contains(T const t) const { return t >= from_ && t < to_; }

  bool overlaps(gpu_interval const& o) const {
    return from_ < o.to_ && to_ > o.from_;
  }

  iterator begin() const { return {from_}; }
  iterator end() const { return {to_}; }
  friend iterator begin(gpu_interval const& r) { return r.begin(); }
  friend iterator end(gpu_interval const& r) { return r.end(); }

  cuda::std::reverse_iterator<iterator> rbegin() const {
    return cuda::std::reverse_iterator<iterator>{iterator{to_}};
  }
  cuda::std::reverse_iterator<iterator> rend() const {
    return cuda::std::reverse_iterator<iterator>{iterator{from_}};
  }
  friend cuda::std::reverse_iterator<iterator> rbegin(gpu_interval const& r) {
    return r.begin();
  }
  friend cuda::std::reverse_iterator<iterator> rend(gpu_interval const& r) {
    return r.end();
  }

  auto size() const { return to_ - from_; }

  T operator[](std::size_t const i) const {
    assert(contains(from_ + static_cast<T>(i)));
    return from_ + static_cast<T>(i);
  }

  friend std::ostream& operator<<(std::ostream& out, gpu_interval const& i) {
    return out << "[" << i.from_ << ", " << i.to_ << "[";
  }
#endif
  T from_{}, to_{};
};

template <typename T, typename T1, typename = std::common_type_t<T1, T>>
gpu_interval(T, T1) -> gpu_interval<std::common_type_t<T, T1>>;

}  // namespace nigiri

template <typename T>
using gpu_interval = nigiri::gpu_interval<T>;

template <typename T>
struct fmt::formatter<nigiri::gpu_interval<T>> : ostream_formatter {};

template <typename T>
using gpu_interval = nigiri::gpu_interval<T>;


template <typename K, typename V, typename SizeType = gpu_base_t<K>>
using gpu_vecvec = cista::raw::gpu_vecvec<K, V, SizeType>;

namespace nigiri{

struct gpu_footpath {
  using value_type = gpu_location_idx_t::value_t;
  static constexpr auto const kTotalBits = 8 * sizeof(value_type);
  static constexpr auto const kTargetBits = 22U;
  static constexpr auto const kDurationBits = kTotalBits - kTargetBits;
  static constexpr auto const kMaxDuration = gpu_duration_t{
      cuda::std::numeric_limits<gpu_location_idx_t::value_t>::max()>> kTargetBits};
  
  gpu_footpath() = default;

  __host__ __device__ gpu_footpath(gpu_location_idx_t::value_t const val) {
    std::memcpy(this, &val, sizeof(value_type));
  }

  __host__ __device__ gpu_footpath(gpu_location_idx_t const target, gpu_duration_t const duration)
      : target_{target},
        duration_{static_cast<value_type>(
            (duration > kMaxDuration ? kMaxDuration : duration).count())} {
  }

  __host__ __device__ gpu_location_idx_t target() const { return gpu_location_idx_t{target_}; }
  __host__ __device__ gpu_duration_t duration() const { return gpu_duration_t{duration_}; }

  __host__ __device__ gpu_location_idx_t::value_t value() const {
    return *reinterpret_cast<gpu_location_idx_t::value_t const*>(this);
  }
  __host__ __device__ friend std::ostream& operator<<(std::ostream& out, gpu_footpath const& fp) {
    return out << "(" << fp.target() << ", " << fp.duration() << ")";
  }

  __host__ __device__ friend bool operator==(gpu_footpath const& a, gpu_footpath const& b) {
    return a.value() == b.value();
  }

  __host__ __device__ friend bool operator<(gpu_footpath const& a, gpu_footpath const& b) {
    return a.value() < b.value();
  }

  gpu_location_idx_t::value_t target_ : kTargetBits;
  gpu_location_idx_t::value_t duration_ : kDurationBits;
};

template <typename Ctx>
inline void serialize(Ctx&, gpu_footpath const*, cista::offset_t const) {}

template <typename Ctx>
inline void deserialize(Ctx const&, gpu_footpath*) {}

struct gpu_locations_device {
  gpu_vector_map<gpu_location_idx_t, gpu_u8_minutes>* transfer_time_;
  gpu_vecvec<gpu_location_idx_t, gpu_footpath>* gpu_footpaths_out_;
  gpu_vecvec<gpu_location_idx_t, gpu_footpath>* gpu_footpaths_in_;
};
}//namespace: nigiri
using gpu_locations = nigiri::gpu_locations_device;

gpu_duration_t constexpr operator""_gpu_days(unsigned long long n) {
  return gpu_duration_t{n * 1440U};
}
constexpr auto const gpu_kMaxTransfers = std::uint8_t{7U};
constexpr auto const gpu_kMaxTravelTime = 1_gpu_days;

enum class gpu_special_station : gpu_location_idx_t::value_t {
  kStart,
  kEnd,
  kVia0,
  kVia1,
  kVia2,
  kVia3,
  kVia4,
  kVia5,
  kVia6,
  kSpecialStationsSize
};

inline bool is_special(gpu_location_idx_t const l) {
  constexpr auto const max =
      static_cast<std::underlying_type_t<gpu_special_station>>(
          gpu_special_station::kSpecialStationsSize);
  return gpu_to_idx(l) < max;
}

auto const gpu_special_stations_names =
    cuda::std::array<std::string_view,
                 static_cast<std::underlying_type_t<gpu_special_station>>(
                     gpu_special_station::kSpecialStationsSize)>{
        "START", "END", "VIA0", "VIA1", "VIA2", "VIA3", "VIA4", "VIA5", "VIA6"};

inline gpu_location_idx_t const get_gpu_special_station(gpu_special_station const x) {
  return gpu_location_idx_t{
      static_cast<std::underlying_type_t<gpu_special_station>>(x)};
}

constexpr std::string_view get_gpu_special_station_name(gpu_special_station const x) {
  return gpu_special_stations_names
      [static_cast<std::underlying_type_t<gpu_special_station>>(x)];
}
struct gpu_stop {
  using value_type = gpu_location_idx_t::value_t;

  __host__ __device__ gpu_stop(gpu_location_idx_t::value_t const val) {
    *reinterpret_cast<value_type*>(this) = val;
  }

  __host__ __device__ gpu_stop(gpu_location_idx_t const location,
       bool const in_allowed,
       bool const out_allowed)
      : location_{location},
        in_allowed_{in_allowed ? 1U : 0U},
        out_allowed_{out_allowed ? 1U : 0U} {}

  __host__ __device__ gpu_location_idx_t gpu_location_idx() const { return gpu_location_idx_t{location_}; }
  __host__ __device__ bool in_allowed() const { return in_allowed_ != 0U; }
  __host__ __device__ bool out_allowed() const { return out_allowed_ != 0U; }
  __host__ __device__ bool is_cancelled() const { return !in_allowed() && !out_allowed(); }

  __host__ __device__ gpu_location_idx_t::value_t value() const {
    return *reinterpret_cast<gpu_location_idx_t::value_t const*>(this);
  }
  __host__ __device__ friend auto operator<=>(gpu_stop const&, gpu_stop const&) = default;

  gpu_stop() = default;
  gpu_location_idx_t::value_t location_ : 30;
  gpu_location_idx_t::value_t in_allowed_ : 1;
  gpu_location_idx_t::value_t out_allowed_ : 1;
};

template <typename BeginIt, typename EndIt = BeginIt>
struct gpu_it_range {
  using value_type =
      std::remove_reference_t<decltype(*std::declval<BeginIt>())>;
  using reference_type = std::add_lvalue_reference_t<value_type>;
  using const_iterator = BeginIt;
  using iterator = BeginIt;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;

  template <typename Collection>
  __host__ __device__ explicit gpu_it_range(Collection const& c)
      : begin_{std::cbegin(c)}, end_{std::cend(c)} {}
  __host__ __device__ explicit gpu_it_range(BeginIt begin, EndIt end)
      : begin_{std::move(begin)}, end_{std::move(end)} {}
  __host__ __device__ BeginIt begin() const { return begin_; }
  __host__ __device__ EndIt end() const { return end_; }
  __host__ __device__ reference_type operator[](std::size_t const i) const {
    return *std::next(begin_, static_cast<difference_type>(i));
  }
  __host__ __device__ value_type const* data() const { return begin_; }
  __host__ __device__ friend BeginIt begin(gpu_it_range const& r) { return r.begin(); }
  __host__ __device__ friend EndIt end(gpu_it_range const& r) { return r.end(); }
  __host__ __device__ reference_type front() const { return *begin_; }
  __host__ __device__ reference_type back() const {
    return *std::next(begin_, static_cast<difference_type>(size() - 1U));
  }
  __host__ __device__ std::size_t size() const {
    return static_cast<std::size_t>(std::distance(begin_, end_));
  }
  __host__ __device__ bool empty() const { return begin_ == end_; }

  BeginIt begin_;
  EndIt end_;
};

__host__ __device__  inline gpu_delta event_mam(gpu_route_idx_t const r,
                                        gpu_transport_idx_t t,
                                        gpu_stop_idx_t const stop_idx,
                                        gpu_event_type const ev_type,
                                        gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const route_transport_ranges,
                                        gpu_delta* route_stop_times,
                                        gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> route_stop_time_ranges){
  auto const range = route_transport_ranges[r];
  auto const n_transports = static_cast<unsigned>(range.size());
  auto const route_stop_begin = static_cast<unsigned>(
      route_stop_time_ranges[r].from_ +
      n_transports * (stop_idx * 2 - (ev_type == gpu_event_type::kArr ? 1 : 0)));
  auto const t_idx_in_route = gpu_to_idx(t) - gpu_to_idx(range.from_);
  return route_stop_times[route_stop_begin + t_idx_in_route];
}
__device__ inline cuda::std::span<gpu_delta const> gpu_event_times_at_stop(gpu_route_idx_t const r,
                                                                    gpu_stop_idx_t const stop_idx,
                                                                    gpu_event_type const ev_type,
                                                                    gpu_vector_map<gpu_route_idx_t,gpu_interval<uint32_t >> const* route_stop_time_ranges,
                                                                           gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t>> const* route_transport_ranges,
                                                                    gpu_delta const* route_stop_times){
  auto const n_transports = static_cast<unsigned>((*route_transport_ranges)[r].size());
  auto const idx = static_cast<unsigned>(
      (*route_stop_time_ranges)[r].from_ + n_transports * (stop_idx * 2 - (ev_type == gpu_event_type::kArr ? 1 : 0)));
  return  cuda::std::span<gpu_delta const>{&route_stop_times[idx], n_transports};
}
__device__ inline gpu_interval<gpu_sys_days> gpu_internal_interval_days(gpu_interval<gpu_sys_days> const* date_range_ptr){
  auto date_range = *date_range_ptr;
  return {date_range.from_ - (gpu_days{1} + gpu_days{4}),
          date_range.to_ + gpu_days{1}};
}