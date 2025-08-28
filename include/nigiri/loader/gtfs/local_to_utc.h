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
  basic_string<gtfs_trip_idx_t> trips_;
  basic_string<duration_t> offsets_;
  bitfield const* traffic_days_;
};

struct utc_trip {
  date::days first_dep_offset_;
  duration_t tz_offset_;
  basic_string<gtfs_trip_idx_t> trips_;
  basic_string<duration_t> utc_times_;
  bitfield utc_traffic_days_;
  stop_seq_t stop_seq_;
};

inline bool headways_match(trip_data const& trip_data,
                           basic_string<gtfs_trip_idx_t> const& trips) {
  if (trips.size() <= 1) {
    return true;
  }

  assert(!trips.empty());
  auto const& first = trip_data.data_[trips.front()];
  return utl::all_of(trips, [&](gtfs_trip_idx_t const idx) {
    auto const& t = trip_data.get(idx);
    if (!t.frequency_.has_value() ||
        t.frequency_->size() != first.frequency_->size()) {
      return false;
    }
    for (auto const [a, b] : utl::zip(*first.frequency_, *t.frequency_)) {
      return a.headway_ != b.headway_;
    }
    return true;
  });
};

inline bool stays_sorted(trip_data const& trip_data,
                         basic_string<gtfs_trip_idx_t> const& trips) {
  if (trips.size() <= 1) {
    return true;
  }

  assert(!trips.empty());
  auto const& first = trip_data.data_[trips.front()];
  for (auto i = 0U; i != first.frequency_->size(); ++i) {
    auto const sorted = utl::is_sorted(
        trips, [&](gtfs_trip_idx_t const a_idx, gtfs_trip_idx_t const b_idx) {
          auto const& a_freq = *trip_data.get(a_idx).frequency_;
          auto const& b_freq = *trip_data.get(b_idx).frequency_;
          return a_freq[i].start_time_ < b_freq[i].start_time_;
        });
    if (!sorted) {
      return false;
    }
  }
  return true;
};

template <typename Consumer>
void expand_frequencies(trip_data const& trip_data,
                        basic_string<gtfs_trip_idx_t> const& trips,
                        bitfield const* traffic_days,
                        Consumer&& consumer) {
  auto const has_frequency = [&](gtfs_trip_idx_t const i) {
    auto const& t = trip_data.get(i);
    return !t.event_times_.empty() && t.frequency_.has_value();
  };

  if (utl::any_of(trips, has_frequency)) {
    if (trips.size() == 1U /* shortcut, no checks needed */ ||
        (utl::all_of(trips, has_frequency) &&
         headways_match(trip_data, trips) && stays_sorted(trip_data, trips))) {
      auto const& first_trp = trip_data.get(trips.front());
      for (auto const [i, freq] : utl::enumerate(*first_trp.frequency_)) {
        auto const index = i;
        for (auto it = 0U; it < freq.number_of_iterations(); ++it) {
          consumer(frequency_expanded_trip{
              .trips_ = trips,
              .offsets_ = utl::transform_to<basic_string<duration_t>>(
                  trips,
                  [&](gtfs_trip_idx_t const t_idx) {
                    auto const& t = trip_data.get(t_idx);
                    auto const first_dep =
                        (*t.frequency_)[index].get_iteration_start_time(it);
                    return t.event_times_.front().dep_ - first_dep;
                  }),
              .traffic_days_ = traffic_days});
        }
      }
    } else {
      for (auto const& t : trips) {
        expand_frequencies(trip_data, {t}, traffic_days, consumer);
      }
    }
  } else {
    consumer(frequency_expanded_trip{
        .trips_ = trips,
        .offsets_ = basic_string<duration_t>{trips.size(), 0_minutes},
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
  auto const tt_interval = tt.internal_interval_days();
  auto trip_it = begin(fet.trips_);
  auto offsets_it = begin(fet.offsets_);
  while (trip_it != end(fet.trips_)) {
    auto const& t = trip_data.get(*trip_it);
    if (t.event_times_.size() <= 1U || t.requires_interpolation_) {
      log(log_lvl::error, "loader.gtfs.trip",
          R"(trip "{}": invalid event times, skipping)", t.id_);
      trip_it = fet.trips_.erase(trip_it);
      offsets_it = fet.offsets_.erase(offsets_it);
    } else {
      ++trip_it;
      ++offsets_it;
    }
  }
  if (fet.trips_.empty()) {
    return;
  }

  auto const n_stops = std::accumulate(
      begin(fet.trips_), end(fet.trips_), 0U,
      [&](unsigned const acc, gtfs_trip_idx_t const t_idx) {
        auto const n_trip_stops =
            static_cast<unsigned>(trip_data.get(t_idx).stop_seq_.size());
        return acc + n_trip_stops;
      });

  auto const first_dep_time =
      trip_data.get(fet.trips_.front()).event_times_.front().dep_ -
      fet.offsets_.back();
  auto const last_arr_time =
      trip_data.get(fet.trips_.back()).event_times_.back().arr_ -
      fet.offsets_.back();
  auto const first_day_offset = (first_dep_time / 1_days) * date::days{1};
  auto const last_day_offset = (last_arr_time / 1_days) * date::days{1};

  auto utc_time_traffic_days = hash_map<conversion_key, bitfield>{};
  auto prev_key = conversion_key{date::days{2}, duration_t{-1}};
  auto prev_it = utc_time_traffic_days.end();
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

    auto const& first_trp = trip_data.get(fet.trips_.front());
    auto const tz_offset =
        noon_offsets.at(tt.providers_[first_trp.route_->agency_].tz_)
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
    basic_string<minutes_after_midnight_t> utc_time_mem;
    utc_time_mem.resize(n_stops * 2U - fet.trips_.size() * 2U);
    auto const [first_dep_day_offset, tz_offset] = key;
    auto i = 0U;
    for (auto const [t, freq_offset] : utl::zip(fet.trips_, fet.offsets_)) {
      auto const& trp = trip_data.get(t);
      for (auto const [from, to] : utl::pairwise(trp.event_times_)) {
        utc_time_mem[i++] =
            from.dep_ - freq_offset - tz_offset - first_dep_day_offset;
        utc_time_mem[i++] =
            to.arr_ - freq_offset - tz_offset - first_dep_day_offset;
      }
    }

    auto pred = minutes_after_midnight_t{0U};
    for (auto& x : utc_time_mem) {
      x = std::max(pred, x);
      pred = x;
    }

    return utc_time_mem;
  };

  for (auto& [key, traffic_days] : utc_time_traffic_days) {
    consumer(utc_trip{.first_dep_offset_ = key.first_dep_day_offset_,
                      .tz_offset_ = key.tz_offset_,
                      .trips_ = fet.trips_,
                      .utc_times_ = build_time_string(key),
                      .utc_traffic_days_ = traffic_days,
                      .stop_seq_ = {}});
  }
}

inline stop_seq_t const* get_stop_seq(trip_data const& trip_data,
                                      utc_trip const& t,
                                      stop_seq_t& stop_seq_cache) {
  if (!t.stop_seq_.empty()) {
    return &t.stop_seq_;
  } else if (t.trips_.size() == 1U) {
    utl::verify(trip_data.get(t.trips_.front()).stop_seq_.size() > 1,
                "trip must have at least two stops");
    return &trip_data.get(t.trips_.front()).stop_seq_;
  } else {
    stop_seq_cache.clear();
    for (auto const [i, t_idx] : utl::enumerate(t.trips_)) {
      auto const& trp = trip_data.get(t_idx);
      if (i != 0) {
        auto const prev_last = stop{stop_seq_cache.back()};
        auto const curr_first = stop{trp.stop_seq_.front()};
        stop_seq_cache.back() =
            stop{prev_last.location_idx(), curr_first.in_allowed(),
                 prev_last.out_allowed(), curr_first.in_allowed_wheelchair(),
                 prev_last.out_allowed_wheelchair()}
                .value();
      }
      stop_seq_cache.insert(
          end(stop_seq_cache),
          i == 0 ? begin(trp.stop_seq_) : std::next(begin(trp.stop_seq_)),
          end(trp.stop_seq_));
    }
    return &stop_seq_cache;
  }
}

template <typename Consumer>
void expand_assistance(timetable const& tt,
                       trip_data const& trip_data,
                       assistance_times& assist,
                       utc_trip&& ut,
                       Consumer&& consumer) {
  auto stop_seq_cache = stop_seq_t{};
  auto assistance_traffic_days = hash_map<stop_seq_t, bitfield>{};
  auto prev_key = stop_seq_t{};
  auto prev_it = assistance_traffic_days.end();
  ut.utc_traffic_days_.for_each_set_bit([&](std::size_t const day_idx) {
    auto const day = date::local_days{
        (tt.internal_interval_days().from_ + date::days{day_idx})
            .time_since_epoch()};

    auto stop_seq = *get_stop_seq(trip_data, ut, stop_seq_cache);
    auto stop_times_it = begin(ut.utc_times_);
    for (auto [a, b] : utl::pairwise(stop_seq)) {
      auto const offset =
          std::chrono::duration_cast<duration_t>(ut.first_dep_offset_) +
          ut.tz_offset_;
      auto const [dep_day_offset, dep] =
          split_time_mod(*stop_times_it++ + offset);
      auto const [arr_day_offset, arr] =
          split_time_mod(*stop_times_it++ + offset);

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
                      .tz_offset_ = ut.tz_offset_,
                      .trips_ = ut.trips_,
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
              auto const c = trip_data.get(ut.trips_.front()).route_->clasz_;
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
