#pragma once

#include <variant>

#include "nigiri/types.h"

namespace nigiri::rt {

// A run represents a single trip on a specific day. It abstracts over
// transport=(transport_idx_t, day_idx_t) and rt_transport_idx_t and provides a
// uniform interface to get information such as scheduled and real-time
// timestamps for stop times, the location sequence, etc. about this specific
// instance.
//
// Variations:
//
//  rt_transport_idx_t  |  transport  |  case
// =====================+=============+=====================================
//  invalid             |  valid      |  SCHEDULED transport, not changed
// ---------------------+-------------+-------------------------------------
//  valid               |  invalid    |  additional real-time transport
//                      |             |  NOT known from static timetable
// ---------------------+-------------+-------------------------------------
//  valid               |  valid      |  SCHEDULED transport with real-time
//                      |             |  update (delays, rerouting, etc.)
// ---------------------+-------------+-------------------------------------
//  invalid             | invalid     |  invalid / not found
struct run {
  bool is_rt() const noexcept { return rt_ != rt_transport_idx_t::invalid(); }
  bool valid() const noexcept {
    return t_.is_valid() || rt_ != rt_transport_idx_t::invalid();
  }

  // from static timetable, not set for additional services
  transport t_{transport::invalid()};

  // real-time instance, not set if no real-time info available
  rt_transport_idx_t rt_{rt_transport_idx_t::invalid()};
};

// Same as `run` data structure, extended with timetable and rt_timetable
// to be able to look up additional info.
struct frun : public run {
  struct run_stop {
    location get_location() const noexcept {
      return location{
          fr_->tt_,
          fr_->is_rt()
              ? stop{fr_->rtt_->rt_transport_location_seq_[fr_->rt_][stop_idx_]}
                    .location_idx()
              : stop{fr_->tt_.route_location_seq_
                         [fr_->tt_.transport_route_[fr_->t_.t_idx_]][stop_idx_]}
                    .location_idx()};
    }

    unixtime_t get_scheduled_arr_time() const noexcept {
      return fr_->is_rt()
          ? fr_->tt_.event_time(fr_->t_, stop_idx_, event_type::kArr);
    }
    unixtime_t get_scheduled_dep_time() const noexcept {}

    unixtime_t get_real_arr_time() const noexcept {}
    unixtime_t get_real_dep_time() const noexcept {}

    stop_idx_t stop_idx_{0U};
    frun const* fr_{nullptr};
  };

  frun(timetable const& tt, rt_timetable const* rtt, run r)
      : run{r}, tt_{tt}, rtt_{rtt} {}

  frun(timetable const& tt,
       rt_timetable const& rtt,
       rt_transport_idx_t const rt_t)
      : run{.t_ = rtt.resolve_static(rt_t), .rt_ = rt_t}, tt_{tt}, rtt_{&rtt} {}

  frun(timetable const& tt, rt_timetable const* rtt, transport const t)
      : run{.t_ = t,
            .rt_ = rtt == nullptr ? rt_transport_idx_t::invalid()
                                  : rtt->resolve_rt(t)},
        tt_{tt},
        rtt_{rtt} {}

  struct stop_iterator {};

  timetable const& tt_;
  rt_timetable const* rtt_;
};

}  // namespace nigiri::rt