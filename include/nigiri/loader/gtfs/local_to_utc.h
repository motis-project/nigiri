#pragma once

#include "date/tz.h"

#include "utl/get_or_create.h"

#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"
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

inline duration_t get_noon_offset(date::local_days const days,
                                  date::time_zone const* tz) {
  auto const abs_zoned_time = date::zoned_time<std::chrono::minutes>(
      tz, days + std::chrono::minutes{12 * 60});
  return duration_t{abs_zoned_time.get_info().offset.count() / 60};
}

template <typename Consumer>
void expand_frequencies(trip const* t, Consumer&& consumer) {
  if (t->frequency_.has_value()) {
    for (auto const& f : *t->frequency_) {
      for (auto i = f.start_time_; i < f.end_time_; i += f.headway_) {
        consumer(frequency_expanded_trip{
            .orig_ = t, .offset_ = t->stop_times_.front().dep_.time_ - i});
      }
    }
  } else {
    consumer(frequency_expanded_trip{.orig_ = t, .offset_ = 0_minutes});
  }
}

template <typename Consumer>
void expand_local_to_utc(timetable const& tt,
                         frequency_expanded_trip const& feq,
                         interval<date::sys_days> const& gtfs_interval,
                         interval<date::sys_days> const& selection,
                         Consumer&& consumer) {
  auto const* t = feq.orig_;

  auto utc_time_traffic_days =
      hash_map<std::basic_string<minutes_after_midnight_t>, bitfield>{};

  auto utc_times = std::basic_string<minutes_after_midnight_t>{};
  utc_times.resize(t->stop_times_.size() * 2U - 2U);

  auto const last_day_offset =
      (1 + (t->stop_times_.back().arr_.time_ - feq.offset_) / 1_days) *
      date::days{1};
  for (auto day = gtfs_interval.from_; day != gtfs_interval.to_;
       day += std::chrono::days{1}) {
    auto const service_days = interval{day, day + last_day_offset};

    if (!selection.overlaps(service_days)) {
      continue;
    }

    auto const gtfs_local_day_idx = (day - gtfs_interval.from_).count();
    if (!t->service_->test(static_cast<std::size_t>(gtfs_local_day_idx))) {
      continue;
    }

    auto const stop_tz_idx =
        tt.locations_.location_timezones_[t->stop_times_.front().stop_];
    auto const agency_tz_idx = t->route_->agency_ == provider_idx_t::invalid()
                                   ? timezone_idx_t::invalid()
                                   : tt.providers_[t->route_->agency_].tz_;
    utl::verify(
        stop_tz_idx != timezone_idx_t::invalid() ||
            agency_tz_idx != timezone_idx_t::invalid(),
        R"(no timezone given for trip "{}" (first_stop="{}", agency="{}"))",
        t->id_, location{tt, t->stop_times_.front().stop_},
        t->route_->agency_ == provider_idx_t::invalid()
            ? "NOT_FOUND"
            : tt.providers_[t->route_->agency_].short_name_);
    auto const tz_idx =
        stop_tz_idx == timezone_idx_t::invalid() ? agency_tz_idx : stop_tz_idx;
    auto const tz = reinterpret_cast<date::time_zone const*>(
        tt.locations_.timezones_[tz_idx].as<void const*>());
    auto const tz_offset =
        get_noon_offset(date::local_days{date::year_month_day{day}}, tz);

    auto const first_dep_utc =
        t->stop_times_.front().dep_.time_ - feq.offset_ - tz_offset;
    auto const first_day_offset = date::days{static_cast<date::days::rep>(
        std::floor(static_cast<double>(first_dep_utc.count()) / 1440))};

    auto i = 0U;
    for (auto const [from, to] : utl::pairwise(t->stop_times_)) {
      auto const& [from_seq, from_stop] = from;
      auto const& [to_seq, to_stop] = to;
      utc_times[i++] =
          from_stop.dep_.time_ - feq.offset_ - tz_offset - first_day_offset;
      utc_times[i++] =
          to_stop.arr_.time_ - feq.offset_ - tz_offset - first_day_offset;
    }

    auto const utc_traffic_day =
        (day - tt.internal_interval_days().from_ + first_day_offset).count();
    if (utc_traffic_day < 0 || utc_traffic_day > kMaxDays) {
      continue;
    }

    auto const it = utc_time_traffic_days.find(utc_times);
    if (it == end(utc_time_traffic_days)) {
      utc_time_traffic_days.emplace(utc_times, bitfield{})
          .first->second.set(static_cast<std::size_t>(utc_traffic_day));
    } else {
      it->second.set(static_cast<std::size_t>(utc_traffic_day));
    }
  }

  for (auto& [times, traffic_days] : utc_time_traffic_days) {
    consumer(utc_trip{
        .orig_ = t, .utc_traffic_days_ = traffic_days, .utc_times_ = times});
  }
}

template <typename Consumer>
void expand_trip(timetable const& tt,
                 trip const* t,
                 interval<date::sys_days> const& gtfs_interval,
                 interval<date::sys_days> const& selection,
                 Consumer&& consumer) {
  expand_frequencies(t, [&](frequency_expanded_trip const& feq) {
    expand_local_to_utc(tt, feq, gtfs_interval, selection,
                        [&](utc_trip&& ut) { consumer(std::move(ut)); });
  });
}

}  // namespace nigiri::loader::gtfs
