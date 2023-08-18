#pragma once

#include "date/tz.h"

#include "utl/enumerate.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"

#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/common/day_list.h"
#include "nigiri/timetable.h"
#include "utl/pipes/accumulate.h"

namespace nigiri::loader::gtfs {

struct frequency_expanded_trip {
  std::basic_string<gtfs_trip_idx_t> trips_;
  std::basic_string<duration_t> offsets_;
  bitfield const* traffic_days_;
};

struct utc_trip {
  duration_t first_dep_offset_;
  std::basic_string<gtfs_trip_idx_t> trips_;
  std::basic_string<duration_t> utc_times_;
  bitfield utc_traffic_days_;
};

inline bool headways_match(trip_data const& trip_data,
                           std::basic_string<gtfs_trip_idx_t> const& trips) {
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
                         std::basic_string<gtfs_trip_idx_t> const& trips) {
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
                        std::basic_string<gtfs_trip_idx_t> const& trips,
                        bitfield const* traffic_days,
                        Consumer&& consumer) {
  auto const has_frequency = [&](gtfs_trip_idx_t const i) {
    return trip_data.get(i).frequency_.has_value();
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
              .offsets_ = utl::transform_to<std::basic_string<duration_t>>(
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
        .offsets_ = std::basic_string<duration_t>{trips.size(), 0_minutes},
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
                         interval<date::sys_days> const& gtfs_interval,
                         interval<date::sys_days> const& selection,
                         Consumer&& consumer) {
  auto trip_it = begin(fet.trips_);
  auto offsets_it = begin(fet.offsets_);
  while (trip_it != end(fet.trips_)) {
    if (trip_data.get(*trip_it).event_times_.size() <= 1U) {
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
  for (auto day = gtfs_interval.from_; day != gtfs_interval.to_;
       day += date::days{1}) {
    auto const service_days =
        interval{day + first_day_offset, day + last_day_offset + date::days{1}};
    if (!selection.overlaps(service_days)) {
      continue;
    }

    auto const gtfs_local_day_idx =
        static_cast<std::size_t>((day - gtfs_interval.from_).count());
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
        (day - tt.internal_interval_days().from_ + first_dep_day_offset)
            .count();

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
    std::basic_string<minutes_after_midnight_t> utc_time_mem;
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
    return utc_time_mem;
  };

  for (auto& [key, traffic_days] : utc_time_traffic_days) {
    consumer(utc_trip{
        .first_dep_offset_ =
            std::chrono::duration_cast<duration_t>(key.first_dep_day_offset_) +
            key.tz_offset_,
        .trips_ = fet.trips_,
        .utc_times_ = build_time_string(key),
        .utc_traffic_days_ = traffic_days});
  }
}

template <typename Consumer>
void expand_trip(trip_data& trip_data,
                 noon_offset_hours_t const& noon_offsets,
                 timetable const& tt,
                 std::basic_string<gtfs_trip_idx_t> const& trips,
                 bitfield const* traffic_days,
                 interval<date::sys_days> const& gtfs_interval,
                 interval<date::sys_days> const& selection,
                 Consumer&& consumer) {
  expand_frequencies(
      trip_data, trips, traffic_days, [&](frequency_expanded_trip&& fet) {
        expand_local_to_utc(trip_data, noon_offsets, tt, std::move(fet),
                            gtfs_interval, selection,
                            [&](utc_trip&& ut) { consumer(std::move(ut)); });
      });
}

}  // namespace nigiri::loader::gtfs
