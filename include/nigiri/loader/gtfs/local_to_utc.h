#pragma once

#include "date/tz.h"

#include "utl/get_or_create.h"

#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"
#include "noon_offsets.h"
#include "utl/pairwise.h"

namespace nigiri::loader::gtfs {

struct frequency_expanded_trip {
  trip const* orig_{nullptr};
  duration_t offset_{0U};
};

struct utc_trip {
  trip const* orig_{nullptr};
  bitfield utc_traffic_days_;
  std::basic_string<duration_t> utc_times_;
};

template <typename Consumer>
void expand_frequencies(trip const* t, Consumer&& consumer) {
  if (t->frequency_.has_value()) {
    for (auto const& f : *t->frequency_) {
      for (auto i = f.start_time_; i < f.end_time_; i += f.headway_) {
        consumer(frequency_expanded_trip{
            .orig_ = t, .offset_ = t->event_times_.front().dep_ - i});
      }
    }
  } else {
    consumer(frequency_expanded_trip{.orig_ = t, .offset_ = 0_minutes});
  }
}

template <typename Consumer>
void expand_local_to_utc(noon_offset_hours_t const& noon_offsets,
                         timetable const& tt,
                         frequency_expanded_trip const& fet,
                         interval<date::sys_days> const& gtfs_interval,
                         interval<date::sys_days> const& selection,
                         Consumer&& consumer) {
  using utc_time_sequence = std::basic_string<minutes_after_midnight_t>;
  struct hash_stop_times {
    std::size_t operator()(utc_time_sequence const& x) const {
      return cista::hashing<utc_time_sequence>{}(x);
    }
  };

  auto const* t = fet.orig_;
  if (t->event_times_.size() <= 1) {
    return;
  }

  auto utc_time_traffic_days =
      tsl::hopscotch_map<utc_time_sequence, bitfield, hash_stop_times>{};

  auto utc_times = std::basic_string<minutes_after_midnight_t>{};
  utc_times.resize(t->event_times_.size() * 2U - 2U);

  auto const last_day_offset =
      (1 + (t->event_times_.back().arr_ - fet.offset_) / 1_days) *
      date::days{1};

  auto prev_conversion_parameters =
      std::tuple<duration_t, date::days>{duration_t{-1}, date::days{2}};
  auto prev_it = utc_time_traffic_days.end();
  for (auto day = gtfs_interval.from_; day != gtfs_interval.to_;
       day += std::chrono::days{1}) {
    auto const service_days = interval{day, day + last_day_offset};

    if (!selection.overlaps(service_days)) {
      continue;
    }

    auto const gtfs_local_day_idx =
        static_cast<std::size_t>((day - gtfs_interval.from_).count());
    if (!t->service_->test(gtfs_local_day_idx)) {
      continue;
    }

    utl::verify(
        t->route_->agency_ != provider_idx_t::invalid() &&
            tt.providers_[t->route_->agency_].tz_ != timezone_idx_t::invalid(),
        "could not find timezone");
    auto const tz_offset =
        noon_offsets.at(tt.providers_[t->route_->agency_].tz_)
            .value()
            .at(gtfs_local_day_idx);

    auto const first_dep_utc =
        t->event_times_.front().dep_ - fet.offset_ - tz_offset;
    auto const first_day_offset = date::days{static_cast<date::days::rep>(
        std::floor(static_cast<double>(first_dep_utc.count()) / 1440))};

    auto const utc_traffic_day =
        (day - tt.internal_interval_days().from_ + first_day_offset).count();
    if (utc_traffic_day < 0 || utc_traffic_day > kMaxDays) {
      continue;
    }

    if (std::tuple{tz_offset, first_day_offset} != prev_conversion_parameters) {
      auto i = 0U;
      for (auto const [from, to] : utl::pairwise(t->event_times_)) {
        utc_times[i++] = from.dep_ - fet.offset_ - tz_offset - first_day_offset;
        utc_times[i++] = to.arr_ - fet.offset_ - tz_offset - first_day_offset;
      }

      auto it = utc_time_traffic_days.find(utc_times);
      if (it == end(utc_time_traffic_days)) {
        (it = utc_time_traffic_days.emplace(utc_times, bitfield{}).first)
            .value()
            .set(static_cast<std::size_t>(utc_traffic_day));
      } else {
        it.value().set(static_cast<std::size_t>(utc_traffic_day));
      }

      prev_conversion_parameters = {tz_offset, first_day_offset};
      prev_it = it;
    } else {
      prev_it.value().set(static_cast<std::size_t>(utc_traffic_day));
    }
  }

  for (auto& [times, traffic_days] : utc_time_traffic_days) {
    consumer(utc_trip{.orig_ = t,
                      .utc_traffic_days_ = traffic_days,
                      .utc_times_ = std::move(times)});
  }
}

template <typename Consumer>
void expand_trip(noon_offset_hours_t const& noon_offsets,
                 timetable const& tt,
                 trip const* t,
                 interval<date::sys_days> const& gtfs_interval,
                 interval<date::sys_days> const& selection,
                 Consumer&& consumer) {
  expand_frequencies(t, [&](frequency_expanded_trip const& fet) {
    expand_local_to_utc(noon_offsets, tt, fet, gtfs_interval, selection,
                        [&](utc_trip&& ut) { consumer(std::move(ut)); });
  });
}

}  // namespace nigiri::loader::gtfs
