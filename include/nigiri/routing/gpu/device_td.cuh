#pragma once

#include "nigiri/constants.h"
#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

inline constexpr auto const kMaxFpMinutes = static_cast<duration_t::rep>(
    (std::int32_t{1} << footpath::kDurationBits) - 1);
static_assert(duration_t{kMaxFpMinutes} == footpath::kMaxDuration);

struct d_td_result {
  duration_t duration_{kMaxFpMinutes};
  std::uint32_t idx_{0U};  // winning entry index (for transport_mode_id_)
  bool valid_{false};
};

template <direction SearchDir, typename Collection>
__device__ d_td_result d_get_td_duration(Collection const& c,
                                         std::uint32_t const from,
                                         std::uint32_t const to,
                                         unixtime_t const t) {
  constexpr auto const kMaxWait = i32_minutes{kMaxTransferTime};
  if constexpr (SearchDir == direction::kForward) {
    for (auto i = from; i != to; ++i) {
      auto const& e = c[i];
      if (e.duration_.count() == kMaxFpMinutes ||
          (e.valid_from_ < t && (i + 1U) != to && c[i + 1U].valid_from_ <= t)) {
        continue;
      }
      if (e.valid_from_ - t > kMaxWait) {
        break;
      }
      auto const start = e.valid_from_ < t ? t : e.valid_from_;
      auto const d = (start + e.duration_ - t).count();
      return {duration_t{static_cast<duration_t::rep>(d)}, i, true};
    }
  } else {
    for (auto i = to; i != from; --i) {
      auto const& e = c[i - 1U];
      if (e.duration_.count() == kMaxFpMinutes ||
          e.valid_from_ + e.duration_ > t) {
        continue;
      }
      auto const next_end =
          i == to ? t : c[i].valid_from_ - i32_minutes{1} + e.duration_;
      auto const latest_arr = next_end < t ? next_end : t;
      if (t - latest_arr > kMaxWait) {
        break;
      }
      auto const d = (t - (latest_arr - e.duration_)).count();
      return {duration_t{static_cast<duration_t::rep>(d)}, i - 1U, true};
    }
  }
  return {};
}

template <direction SearchDir, typename Collection, typename Fn>
__device__ void d_for_each_td_footpath(Collection const& c,
                                       unixtime_t const t,
                                       Fn&& f) {
  auto const n = static_cast<std::uint32_t>(c.size());
  auto i = std::uint32_t{0U};
  while (i < n) {
    auto j = i + 1U;
    while (j < n && c[j].target_ == c[i].target_) {
      ++j;
    }
    auto const r = d_get_td_duration<SearchDir>(c, i, j, t);
    if (r.valid_) {
      auto const d = r.duration_.count() > kMaxFpMinutes
                         ? duration_t{kMaxFpMinutes}
                         : r.duration_;
      f(c[i].target_, d);
    }
    i = j;
  }
}

}  // namespace nigiri::routing::gpu
