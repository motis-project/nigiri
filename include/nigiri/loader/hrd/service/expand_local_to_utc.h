#pragma once

#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/service/service.h"
#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

void build_stop_seq(ref_service const& s,
                    service_store const& store,
                    stamm const& st,
                    std::size_t const hrd_local_day_idx,
                    std::vector<duration_t> const& local_times,
                    std::basic_string<stop::value_type>& stop_seq) {
  auto const& ref = store.get(s.ref_);
  auto const n_stops = s.stops(store).size();
  for (auto i = 0U; i != n_stops; ++i) {
    auto const is_last = (i == n_stops - 1);
    auto const section_idx =
        s.split_info_.sections_.from_ + i - (is_last ? 1U : 0U);
    auto const time_idx = i * 2 - (is_last ? 1U : 0U);
    auto const stop_idx = s.split_info_.stop_range().from_ + i;

    auto const train_nr =
        ref.begin_to_end_info_.train_num_.has_value()
            ? ref.begin_to_end_info_.train_num_.value()
            : s.sections(store)[section_idx].train_num_.value();
    auto const admin = ref.begin_to_end_info_.admin_.has_value()
                           ? ref.begin_to_end_info_.admin_.value()
                           : s.sections(store)[section_idx].admin_.value();
    auto const day_offset = static_cast<cista::base_t<day_idx_t>>(
        local_times[time_idx].count() / 1440);
    auto const mam = duration_t{local_times[time_idx].count() % 1440};
    auto const l_idx = st.resolve_track(
        track_rule_key{st.resolve_location(ref.stops_[stop_idx].eva_num_),
                       train_nr, admin},
        mam,
        day_idx_t{
            static_cast<day_idx_t::value_t>(hrd_local_day_idx + day_offset)});

    stop_seq[i] = stop{l_idx, ref.stops_[stop_idx].dep_.in_out_allowed_,
                       ref.stops_[stop_idx].arr_.in_out_allowed_}
                      .value();
  }
}

std::optional<duration_t> build_utc_time_seq(
    parser_info const& origin,
    std::vector<duration_t> const& local_times,
    std::vector<tz_offsets> const& stop_timezones,
    std::chrono::sys_days const day,
    std::basic_string<duration_t>& utc_times) {
  auto const [_, first_day_offset, first_valid] = local_mam_to_utc_mam(
      stop_timezones.front(), day, local_times.front(), true);

  if (!first_valid) {
    log(log_lvl::error, "loader.hrd.service.utc",
        "first departure local to utc failed for {}: local_time={}, day={}",
        origin, local_times.front(), day);
    return std::nullopt;
  }

  auto i = 0U;
  auto pred = duration_t{0};
  for (auto const [local_time, tz] : utl::zip(local_times, stop_timezones)) {
    auto const [utc_mam, day_offset, valid] = local_mam_to_utc_mam(
        tz, day + first_day_offset, local_time - first_day_offset);
    if (day_offset != 0_days || pred > utc_mam || !valid) {
      log(log_lvl::error, "loader.hrd.service.utc",
          "local to utc failed, ignoring: {}, day={}, time={}, offset={}, "
          "pred={}, utc_mam={}, valid={}",
          origin, day, local_time, day_offset, pred, utc_mam, valid);
      return std::nullopt;
    }
    utc_times[i++] = utc_mam;
    pred = utc_mam;
  }

  return first_day_offset;
}

template <typename Fn>
void to_utc(service_store const& store,
            stamm const& st,
            interval<std::chrono::sys_days> const& hrd_interval,
            interval<std::chrono::sys_days> const& selection,
            ref_service const& s,
            Fn&& consumer) {
  using key_t = std::pair<std::basic_string<minutes_after_midnight_t>,
                          std::basic_string<stop::value_type>>;
  auto utc_time_traffic_days = hash_map<key_t, bitfield>{};

  auto const local_times = s.local_times(store);
  auto const last_day_offset =
      (1 + local_times.back() / 1_days) * std::chrono::days{1};
  auto const stop_timezones = s.get_stop_timezones(store, st);

  auto key = key_t{};

  auto& utc_service_times = key.first;
  utc_service_times.resize(s.split_info_.stop_range().size() * 2U - 2U);

  auto& stop_seq = key.second;
  stop_seq.resize(s.split_info_.stop_range().size());

  auto const offset =
      (selection.from_ - hrd_interval.from_ - kTimetableOffset).count();
  for (auto day = hrd_interval.from_; day != hrd_interval.to_;
       day += std::chrono::days{1}) {
    auto const service_days = interval{day, day + last_day_offset};

    if (!selection.overlaps(service_days)) {
      continue;
    }

    auto const hrd_local_day_idx = (day - hrd_interval.from_).count();
    if (!s.local_traffic_days().test(
            static_cast<std::size_t>(hrd_local_day_idx))) {
      continue;
    }

    auto const first_day_offset = build_utc_time_seq(
        s.origin(store), local_times, stop_timezones, day, utc_service_times);
    if (!first_day_offset.has_value()) {
      continue;
    }
    build_stop_seq(s, store, st, static_cast<std::size_t>(hrd_local_day_idx),
                   local_times, stop_seq);

    auto const utc_traffic_day =
        hrd_local_day_idx - offset + first_day_offset.value() / 1_days;
    if (utc_traffic_day < 0) {
      continue;
    }

    auto const it = utc_time_traffic_days.find(key);
    if (it == end(utc_time_traffic_days)) {
      utc_time_traffic_days.emplace(key, bitfield{})
          .first->second.set(static_cast<std::size_t>(utc_traffic_day));
    } else {
      it->second.set(static_cast<std::size_t>(utc_traffic_day));
    }
  }

  for (auto& [times, traffic_days] : utc_time_traffic_days) {
    consumer(ref_service{s, std::move(times.first), std::move(times.second),
                         traffic_days});
  }
}

}  // namespace nigiri::loader::hrd
