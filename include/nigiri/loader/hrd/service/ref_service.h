#pragma once

#include "utl/pairwise.h"

#include "nigiri/loader/hrd/service/service.h"
#include "nigiri/loader/hrd/service/service_store.h"
#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct split_info {
  interval<unsigned> stop_range() const {
    // Example:
    // Stops:    0 --- 1 --- 2 --- 3
    // Sections:    0     1     2
    // --> section i connects stop (i) to (i+1)
    // --> section range [0, 2[ = {0, 1}
    //     is stop range [0, 3[ = {0, 1, 2}
    return {sections_.from_, sections_.to_ + 1};
  }

  bitfield traffic_days_;
  interval<unsigned> sections_;
};

struct ref_service {
  ref_service(service_idx_t ref, split_info split)
      : ref_{ref}, split_info_{split} {}

  ref_service(ref_service const& s, unsigned const r) : ref_service{s} {
    repetition_ = r;
  }

  ref_service(ref_service const& s,
              vector<duration_t> utc_times,
              bitfield utc_traffic_days)
      : ref_service{s} {
    utc_traffic_days_ = std::move(utc_traffic_days);
    utc_times_ = std::move(utc_times);
  }

  vector<duration_t> local_times(service_store const& store) const {
    auto const& ref = store.get(ref_);

    auto i = 0U;
    vector<duration_t> stop_times{split_info_.stop_range().size() * 2U - 2U};
    for (auto const [from, to] : utl::pairwise(split_info_.stop_range())) {
      stop_times[i++] = duration_t{ref.stops_.at(from).dep_.time_ +
                                   repetition_ * ref.interval_};
      stop_times[i++] = duration_t{ref.stops_.at(to).arr_.time_ +
                                   repetition_ * ref.interval_};
    }
    return stop_times;
  }

  vector<tz_offsets> get_stop_timezones(service_store const& store,
                                        timezone_map_t const& tz) const {
    auto const& ref = store.get(ref_);

    auto i = 0U;
    vector<tz_offsets> stop_tzs{split_info_.stop_range().size() * 2U - 2U};
    for (auto const [from, to] : utl::pairwise(split_info_.stop_range())) {
      stop_tzs[i++] = get_tz(tz, ref.stops_.at(from).eva_num_).second;
      stop_tzs[i++] = get_tz(tz, ref.stops_.at(to).eva_num_).second;
    }
    return stop_tzs;
  }

  it_range<std::vector<service::section>::const_iterator> sections(
      service_store const& store) const {
    return make_it_range(store.get(ref_).sections_, split_info_.sections_);
  }

  it_range<std::vector<service::stop>::const_iterator> stops(
      service_store const& store) const {
    return make_it_range(store.get(ref_).stops_, split_info_.stop_range());
  }

  parser_info origin(service_store const& store) const {
    return store.get(ref_).origin_;
  }

  bitfield const& local_traffic_days() const {
    return split_info_.traffic_days_;
  }

  service_idx_t ref_;
  split_info split_info_;
  unsigned repetition_;
  bitfield utc_traffic_days_;
  vector<duration_t> utc_times_;
};

}  // namespace nigiri::loader::hrd
