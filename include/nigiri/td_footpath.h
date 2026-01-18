#pragma once

#include <span>

#include "cista/reflection/comparable.h"

#include "utl/cflow.h"
#include "utl/equal_ranges_linear.h"
#include "utl/pairwise.h"

#include "nigiri/constants.h"
#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {

constexpr auto const kNull = unixtime_t{0_minutes};

struct td_footpath {
  CISTA_FRIEND_COMPARABLE(td_footpath)
  location_idx_t target_;
  unixtime_t valid_from_;
  duration_t duration_;
};

template <direction SearchDir, typename Collection>
std::optional<std::pair<duration_t, typename Collection::value_type>>
get_td_duration(Collection const& c, unixtime_t const t) {
  using namespace std::chrono_literals;

  if constexpr (SearchDir == direction::kForward) {
    for (auto i = cbegin(c); i != cend(c); ++i) {
      if (i->duration_ == footpath::kMaxDuration ||
          (i->valid_from_ < t && (i + 1) != cend(c) &&
           (i + 1)->valid_from_ <= t)) {
        continue;
      }

      if (i->valid_from_ - t > std::chrono::minutes{kMaxTransferTime}) {
        break;
      }

      return std::pair{std::max(i->valid_from_, t) + i->duration_ - t, *i};
    }

  } else /* (SearchDir == direction::kBackward) */ {
    for (auto i = crbegin(c); i != crend(c); ++i) {
      if (i->duration_ == footpath::kMaxDuration ||
          i->valid_from_ + i->duration_ > t) {
        continue;
      }

      auto const latest_arr =
          i == crbegin(c)
              ? t
              : std::min(
                    t, unixtime_t{(i - 1)->valid_from_ - 1min + i->duration_});

      if (t - latest_arr > std::chrono::minutes{kMaxTransferTime}) {
        break;
      }

      return std::pair{t - (latest_arr - i->duration_), *i};
    }
  }

  return std::nullopt;
}

template <typename Collection>
std::optional<std::pair<duration_t, typename Collection::value_type>>
get_td_duration(direction const search_dir,
                Collection const& c,
                unixtime_t const t) {
  return search_dir == direction::kForward
             ? get_td_duration<direction::kForward>(c, t)
             : get_td_duration<direction::kBackward>(c, t);
}

template <direction SearchDir, typename Collection, typename Fn>
void for_each_footpath(Collection const& c, unixtime_t const t, Fn&& f) {
  utl::equal_ranges_linear(
      begin(c), end(c),
      [](td_footpath const& a, td_footpath const& b) {
        return a.target_ == b.target_;
      },
      [&](auto&& from, auto&& to) {
        auto const fp = get_td_duration<SearchDir>(std::span{from, to}, t);
        if (fp.has_value()) {
          f(footpath{from->target_, fp->first});
        }
      });
}

template <typename Collection, typename Fn>
void for_each_footpath(direction const search_dir,
                       Collection const& c,
                       unixtime_t const t,
                       Fn&& f) {
  search_dir == direction::kForward
      ? for_each_footpath<direction::kForward>(c, t, std::forward<Fn>(f))
      : for_each_footpath<direction::kBackward>(c, t, std::forward<Fn>(f));
}

}  // namespace nigiri