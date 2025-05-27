#pragma once

#include <span>

#include "cista/reflection/comparable.h"

#include "utl/cflow.h"
#include "utl/equal_ranges_linear.h"
#include "utl/pairwise.h"

#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {

constexpr auto const kNull = unixtime_t{0_minutes};
constexpr auto const kInfeasible =
    duration_t{std::numeric_limits<duration_t::rep>::max()};

struct td_footpath {
  CISTA_FRIEND_COMPARABLE(td_footpath)
  location_idx_t target_;
  unixtime_t valid_from_;
  duration_t duration_;
};

template <direction SearchDir, typename Collection>
std::optional<std::pair<duration_t, typename Collection::value_type>>
get_td_duration(Collection const& c, unixtime_t const t) {
  auto const r = to_range<SearchDir>(c);
  auto const from = r.begin();
  auto const to = r.end();

  using Type = std::decay_t<decltype(*from)>;

  if constexpr (SearchDir == direction::kForward) {
    Type const* pred = nullptr;

    auto const get = [&]() -> std::optional<std::pair<duration_t, Type>> {
      auto const start = std::max(pred->valid_from_, t);
      auto const target_time = start + pred->duration_;
      auto const duration_with_waiting = target_time - t;
      if (duration_with_waiting < footpath::kMaxDuration) {
        return std::pair{static_cast<duration_t>(duration_with_waiting), *pred};
      } else {
        return std::nullopt;
      }
    };

    for (auto it = from; it != to; ++it) {
      if (pred == nullptr || pred->duration_ == footpath::kMaxDuration ||
          it->valid_from_ < t + pred->duration_) {
        pred = &*it;
      } else {
        return get();
      }
    }

    if (pred != nullptr && pred->duration_ != footpath::kMaxDuration) {
      return get();
    }

    return std::nullopt;
  } else /* (SearchDir == direction::kBackward) */ {
    auto const get_valid_from = [](auto const& x) {
      // Example interval [10:00, 11:00], duration 10 min
      // -> backward at 10:00 => result 10:00 - 10 min = 9:50 !! outside [10-11]
      // -> fixed interval: [10:10, 11:10], duration 10 min **
      //    so 10:00 is not possible
      if (x.duration_ == footpath::kMaxDuration) {
        return x.valid_from_;
      } else {
        return x.valid_from_ + x.duration_;
      }
    };

    auto const get =
        [&](unixtime_t const valid_from,
            Type const& fp) -> std::optional<std::pair<duration_t, Type>> {
      auto const start = std::min(valid_from, t);
      auto const target_time = start - fp.duration_;
      auto const duration_with_waiting = t - target_time;
      if (duration_with_waiting < footpath::kMaxDuration) {
        return std::pair{duration_with_waiting, fp};
      } else {
        return std::nullopt;
      }
    };

    if (from != to && get_valid_from(*from) <= t) {
      if (from->duration_ != footpath::kMaxDuration) {
        return get(t, *from);
      } else if (auto const next = std::next(from); next != to) {
        return get(get_valid_from(*from), *next);
      }
    }

    using namespace std::chrono_literals;
    for (auto const [a, b] : utl::pairwise(it_range{from, to})) {
      if (b.duration_ != footpath::kMaxDuration &&
          (t >= a.valid_from_ ||
           interval{get_valid_from(b), get_valid_from(a) + 1min}.contains(t))) {
        return get(get_valid_from(a), b);
      }
    }

    return std::nullopt;
  }
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