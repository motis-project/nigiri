#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

template <direction SearchDir>
struct interval_estimator {
  explicit interval_estimator(timetable const& tt, query const& q)
      : tt_{tt}, q_{q} {}

  interval<unixtime_t> initial(interval<unixtime_t> const& itv) const {
    if (q_.min_connection_count_ == 0 ||
        (!q_.extend_interval_earlier_ && !q_.extend_interval_later_)) {
      return itv;
    }

    auto new_itv = itv;
    auto const ext = 1_hours * q_.min_connection_count_;

    if (can_extend_bad_dir(itv)) {
      if constexpr (SearchDir == direction::kForward) {
        if (can_extend_earlier(itv)) {
          new_itv.from_ -= 1_hours;
        } else {
          new_itv.to_ += 1_hours;
        }
        new_itv.to_ += ext;
      } else {
        if (can_extend_later(itv)) {
          new_itv.to_ += 1_hours;
        } else {
          new_itv.from_ -= 1_hours;
        }
        new_itv.from_ -= ext;
      }
    } else {
      if constexpr (SearchDir == direction::kForward) {
        if (q_.extend_interval_earlier_) {
          new_itv.from_ -= 1_hours + ext;
        }
      } else {
        if (q_.extend_interval_later_) {
          new_itv.to_ += 1_hours + ext;
        }
      }
    }

    new_itv.from_ = tt_.external_interval().clamp(new_itv.from_);
    new_itv.to_ = tt_.external_interval().clamp(new_itv.to_);

    return new_itv;
  }

  interval<unixtime_t> extension(interval<unixtime_t> const& itv,
                                 std::uint32_t const num_con_req) const {
    if (num_con_req == 0 ||
        (!q_.extend_interval_earlier_ && !q_.extend_interval_later_)) {
      return itv;
    }

    auto new_itv = itv;
    auto const ext = itv.size() * num_con_req;

    if (can_extend_both_dir(itv)) {
      new_itv.from_ -= ext / 2;
      new_itv.to_ += ext / 2;
    } else {
      if (q_.extend_interval_earlier_) {
        new_itv.from_ -= ext;
      }
      if (q_.extend_interval_later_) {
        new_itv.to_ += ext;
      }
    }

    new_itv.from_ = tt_.external_interval().clamp(new_itv.from_);
    new_itv.to_ = tt_.external_interval().clamp(new_itv.to_);

    return new_itv;
  }

private:
  bool can_extend_earlier(interval<unixtime_t> const& itv) const {
    return q_.extend_interval_earlier_ &&
           itv.from_ != tt_.external_interval().from_;
  }

  bool can_extend_later(interval<unixtime_t> const& itv) const {
    return q_.extend_interval_later_ && itv.to_ != tt_.external_interval().to_;
  }

  bool can_extend_bad_dir(interval<unixtime_t> const& itv) const {
    if constexpr (SearchDir == direction::kForward) {
      return can_extend_later(itv);
    } else {
      return can_extend_earlier(itv);
    }
  }

  bool can_extend_both_dir(interval<unixtime_t> const& itv) const {
    return can_extend_earlier(itv) && can_extend_later(itv);
  }

  timetable const& tt_;
  query const& q_;
};

}  // namespace nigiri::routing