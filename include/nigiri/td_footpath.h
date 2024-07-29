#pragma once

#include "cista/reflection/comparable.h"

#include "utl/cflow.h"

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

template <direction SearchDir, typename Collection, typename Fn>
void for_each_footpath(Collection const& c, unixtime_t const t, Fn&& f) {
  static constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const r = to_range<SearchDir>(c);

  auto to = location_idx_t::invalid();
  auto pred = static_cast<td_footpath const*>(nullptr);
  auto const call = [&](td_footpath const& curr) -> std::pair<bool, bool> {
    std::cout << "CALL: " << (pred != nullptr) << ", duration="
              << (pred == nullptr ? kInfeasible : pred->duration_) << "\n";
    auto x = kFwd ? pred : &curr;
    if (x != nullptr && x->duration_ != footpath::kMaxDuration) {
      auto const start =
          kFwd ? std::max(x->valid_from_, t)
               : std::min(/* TODO why pred */ pred->valid_from_, t);
      std::cout << "  start=" << start << "=min(" << x->valid_from_ << ", " << t
                << ")\n";
      auto const target_time = start + (kFwd ? 1 : -1) * x->duration_;
      std::cout << "  x_duration=" << x->duration_
                << ", target_time=" << target_time << "\n";
      auto const duration = kFwd ? (target_time - t) : (t - target_time);
      std::cout << "duration=" << duration << "\n";
      auto const fp = footpath{x->target_, duration};
      auto const stop = f(fp) == utl::cflow::kBreak;
      return {true, stop};
    }
    return {false, false};
  };

  auto called = false;
  auto stop = false;
  for (auto const& fp : r) {
    std::cout << "fp: valid_from=" << fp.valid_from_
              << ", duration=" << fp.duration_ << ", called=" << called
              << ", to=" << to << ", fp_target=" << fp.target_ << ", reached="
              << (kFwd ? fp.valid_from_ > t : fp.valid_from_ < t) << "\n";

    if (!called && (fp.target_ != to ||
                    (kFwd ? fp.valid_from_ > t : fp.valid_from_ < t))) {
      std::tie(called, stop) = call(fp);
      if (stop) {
        return;
      }
    }

    if (fp.target_ != to) {
      called = false;
    }
    to = fp.target_;
    pred = &fp;
  }

  if constexpr (kFwd) {
    if (!called) {
      std::cout << "-> last call\n";
      call(td_footpath{});
    }
  }
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