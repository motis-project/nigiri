#pragma once

#include "date/tz.h"

#include "utl/enumerate.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/pipes/accumulate.h"

#include "nigiri/loader/assistance.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/common/day_list.h"
#include "nigiri/common/split_duration.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

struct frequency_expanded_trip {
  gtfs_trip_idx_t trip_;
  duration_t offset_;
  bitfield const* traffic_days_;
};

struct utc_trip {
  duration_t first_dep_offset_;
  gtfs_trip_idx_t trip_;
  basic_string<duration_t> utc_times_;
  bitfield utc_traffic_days_;
  stop_seq_t stop_seq_;
};

template <typename Consumer>
void expand_frequencies(trip_data const& trip_data,
                        gtfs_trip_idx_t const trip_idx,
                        bitfield const* traffic_days,
                        Consumer&& consumer) {
  auto const& t = trip_data.get(trip_idx);
  if (!t.event_times_.empty() && t.frequency_.has_value()) {
    for (auto const [i, freq] : utl::enumerate(*t.frequency_)) {
      auto const index = i;
      for (auto it = 0U; it < freq.number_of_iterations(); ++it) {
        consumer(frequency_expanded_trip{
            .trip_ = trip_idx,
            .offset_ = t.event_times_.front().dep_ -
                       (*t.frequency_)[index].get_iteration_start_time(it),
            .traffic_days_ = traffic_days});
      }
    }
  } else {
    consumer(frequency_expanded_trip{.trip_ = trip_idx,
                                     .offset_ = 0_minutes,
                                     .traffic_days_ = traffic_days});
  }
}

struct conversion_key {
  CISTA_FRIEND_COMPARABLE(conversion_key)
  date::days first_dep_day_offset_;
  duration_t tz_offset_;
};

template <typename Consumer>
void expand_local_to_utc(trip_data const& trip_data,
                         noon_offset_hours_t const& noon_offsets,
                         timetable const& tt,
                         frequency_expanded_trip&& fet,
                         interval<date::sys_days> const& selection,
                         Consumer&& consumer) {
  auto const& t = trip_data.get(fet.trip_);

  if (t.event_times_.size() <= 1U || t.requires_interpolation_) {
    log(log_lvl::error, "loader.gtfs.trip",
        R"(trip "{}": invalid event times, skipping)", t.id_);
    return;
  }

  auto const n_stops = t.stop_seq_.size();

  auto const first_dep_time = t.event_times_.front().dep_ - fet.offset_;
  auto const last_arr_time = t.event_times_.back().arr_ - fet.offset_;
  auto const first_day_offset = (first_dep_time / 1_days) * date::days{1};
  auto const last_day_offset = (last_arr_time / 1_days) * date::days{1};

  auto utc_time_traffic_days = hash_map<conversion_key, bitfield>{};
  auto prev_key = conversion_key{date::days{2}, duration_t{-1}};
  auto prev_it = utc_time_traffic_days.end();
  auto const tt_interval = tt.internal_interval_days();
  for (auto day = tt_interval.from_; day != tt_interval.to_;
       day += date::days{1}) {
    auto const service_days =
        interval{day + first_day_offset, day + last_day_offset + date::days{1}};
    if (!selection.overlaps(service_days)) {
      continue;
    }

    auto const gtfs_local_day_idx =
        static_cast<std::size_t>((day - tt_interval.from_).count());
    if (!fet.traffic_days_->test(gtfs_local_day_idx)) {
      continue;
    }

    auto const tz_offset = noon_offsets.at(tt.providers_[t.route_->agency_].tz_)
                               .value()
                               .at(gtfs_local_day_idx);
    auto const first_dep_utc = first_dep_time - tz_offset;
    auto const first_dep_day_offset = date::days{static_cast<date::days::rep>(
        std::floor(static_cast<double>(first_dep_utc.count()) / 1440))};
    auto const utc_traffic_day =
        (day - tt_interval.from_ + first_dep_day_offset).count();

    if (utc_traffic_day < 0 || utc_traffic_day >= kMaxDays) {
      continue;
    }

    auto const key = conversion_key{first_dep_day_offset, tz_offset};
    if (key == prev_key) {
      prev_it->second.set(static_cast<std::size_t>(utc_traffic_day));
    } else {
      (prev_it = utc_time_traffic_days.emplace(key, bitfield{}).first)
          ->second.set(static_cast<std::size_t>(utc_traffic_day));
      prev_key = key;
    }
  }

  auto const build_time_string = [&](conversion_key const key) {
    auto utc_time_mem = basic_string<minutes_after_midnight_t>{};
    utc_time_mem.resize(n_stops * 2U - 2U);
    auto const [first_dep_day_offset, tz_offset] = key;
    auto i = 0U;
    for (auto const [from, to] : utl::pairwise(t.event_times_)) {
      utc_time_mem[i++] =
          from.dep_ - fet.offset_ - tz_offset - first_dep_day_offset;
      utc_time_mem[i++] =
          to.arr_ - fet.offset_ - tz_offset - first_dep_day_offset;
    }

    auto pred = minutes_after_midnight_t{0U};
    for (auto& x : utc_time_mem) {
      x = std::max(pred, x);
      pred = x;
    }

    return utc_time_mem;
  };

  for (auto& [key, traffic_days] : utc_time_traffic_days) {
    consumer(utc_trip{
        .first_dep_offset_ =
            std::chrono::duration_cast<duration_t>(key.first_dep_day_offset_) +
            key.tz_offset_,
        .trip_ = fet.trip_,
        .utc_times_ = build_time_string(key),
        .utc_traffic_days_ = traffic_days,
        .stop_seq_ = {}});
  }
}

inline stop_seq_t const* get_stop_seq(trip_data const& trip_data,
                                      utc_trip const& t) {
  if (!t.stop_seq_.empty()) {
    return &t.stop_seq_;
  } else {
    return &trip_data.get(t.trip_).stop_seq_;
  }
}

template <typename Consumer>
void expand_assistance(timetable const& tt,
                       trip_data const& trip_data,
                       assistance_times& assist,
                       utc_trip&& ut,
                       Consumer&& consumer) {
  auto assistance_traffic_days = hash_map<stop_seq_t, bitfield>{};
  auto prev_key = stop_seq_t{};
  auto prev_it = assistance_traffic_days.end();
  ut.utc_traffic_days_.for_each_set_bit([&](std::size_t const day_idx) {
    auto const day = date::local_days{
        (tt.internal_interval_days().from_ + date::days{day_idx})
            .time_since_epoch()};

    auto stop_seq = *get_stop_seq(trip_data, ut);
    auto stop_times_it = begin(ut.utc_times_);
    for (auto [a, b] : utl::pairwise(stop_seq)) {
      auto const [dep_day_offset, dep] =
          split_time_mod(*stop_times_it++ + ut.first_dep_offset_);
      auto const [arr_day_offset, arr] =
          split_time_mod(*stop_times_it++ + ut.first_dep_offset_);

      auto from = stop{a};
      auto to = stop{b};
      from.in_allowed_wheelchair_ =
          (from.in_allowed_ &&
           assist.is_available(tt, from.location_idx(),
                               oh::local_minutes{day + dep_day_offset + dep}))
              ? 1U
              : 0U;
      to.out_allowed_wheelchair_ =
          (to.out_allowed_ &&
           assist.is_available(tt, to.location_idx(),
                               oh::local_minutes{day + arr_day_offset + arr}))
              ? 1U
              : 0U;

      a = from.value();
      b = to.value();
    }

    if (stop_seq == prev_key) {
      prev_it->second.set(day_idx);
    } else {
      (prev_it = assistance_traffic_days.emplace(stop_seq, bitfield{}).first)
          ->second.set(day_idx);
      prev_key = stop_seq;
    }
  });
  for (auto const& [stop_seq, traffic_days] : assistance_traffic_days) {
    consumer(utc_trip{.first_dep_offset_ = ut.first_dep_offset_,
                      .trip_ = ut.trip_,
                      .utc_times_ = ut.utc_times_,
                      .utc_traffic_days_ = traffic_days,
                      .stop_seq_ = stop_seq});
  }
}

template <typename Consumer>
void expand_trip(trip_data& trip_data,
                 noon_offset_hours_t const& noon_offsets,
                 timetable const& tt,
                 basic_string<gtfs_trip_idx_t> const& trips,
                 bitfield const* traffic_days,
                 interval<date::sys_days> const& selection,
                 assistance_times* assist,
                 Consumer&& consumer) {
  expand_frequencies(
      trip_data, trips, traffic_days, [&](frequency_expanded_trip&& fet) {
        expand_local_to_utc(
            trip_data, noon_offsets, tt, std::move(fet), selection,
            [&](utc_trip&& ut) {
              auto const c = trip_data.get(ut.trip_).route_->clasz_;
              if (assist != nullptr &&
                  (c == clasz::kHighSpeed || c == clasz::kLongDistance ||
                   c == clasz::kNight)) {
                expand_assistance(tt, trip_data, *assist, std::move(ut),
                                  consumer);
              } else {
                consumer(std::move(ut));
              }
            });
      });
}

}  // namespace nigiri::loader::gtfs
