#pragma once

#include <span>

#include "cista/reflection/comparable.h"

#include "utl/cflow.h"
#include "utl/equal_ranges_linear.h"
#include "utl/pairwise.h"

#include "nigiri/footpath.h"
#include "nigiri/types.h"
#include "routing/query.h"

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
std::optional<duration_t> get_td_duration(Collection const& c,
                                          unixtime_t const t) {
  auto const r = to_range<SearchDir>(c);
  auto const from = r.begin();
  auto const to = r.end();

  using Type = std::decay_t<decltype(*from)>;

  if constexpr (SearchDir == direction::kForward) {
    Type const* pred = nullptr;
    Type const* curr = nullptr;

    auto const get = [&]() -> std::optional<duration_t> {
      auto const start = std::max(pred->valid_from_, t);
      auto const target_time = start + pred->duration_;
      auto const duration_with_waiting = target_time - t;
      if (duration_with_waiting < footpath::kMaxDuration) {
        return duration_with_waiting;
      } else {
        return std::nullopt;
      }
    };

    for (auto it = from; it != to; ++it) {
      if (curr == nullptr || curr->duration_ == footpath::kMaxDuration ||
          it->valid_from_ < t + curr->duration_) {
        curr = &*it;
      } else if (pred == nullptr || curr->duration_ < pred->duration_) {
        pred = &*curr;
      }
    }

    if (pred == nullptr || curr->duration_ < pred->duration_) {
      pred = &*curr;
    }

    if (pred != nullptr && pred->duration_ != footpath::kMaxDuration) {
      return get();
    }

    return std::nullopt;
  } else /* (SearchDir == direction::kBackward) */ {
    Type const* pred = nullptr;
    auto dep = unixtime_t{};

    if (from->duration_ != footpath::kMaxDuration &&
        from->valid_from_ <= t - from->duration_) {
      pred = &*from;
      dep = t - from->duration_;
    }

    using namespace std::chrono_literals;
    for (auto const [a, b] : utl::pairwise(it_range{from, to})) {
      if (b.duration_ != footpath::kMaxDuration &&
          std::max(b.valid_from_, dep) + b.duration_ <= t &&
          interval{b.valid_from_, a.valid_from_ + 1min}.overlaps(
              interval{dep + 1min, t + 1min})) {
        const auto new_dep = std::min(a.valid_from_, t) - b.duration_;
        if (dep < new_dep) {
          dep = new_dep;
          pred = &b;
        }
      }
    }

    if (pred != nullptr && pred->duration_ != footpath::kMaxDuration) {
      return t - dep;
    }

    return std::nullopt;
  }
}

template <typename Collection>
std::optional<duration_t> get_td_duration(direction const search_dir,
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
        auto const duration =
            get_td_duration<SearchDir>(std::span{from, to}, t);
        if (duration.has_value()) {
          f(footpath{from->target_, *duration});
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