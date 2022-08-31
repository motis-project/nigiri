#pragma once

#include "nigiri/loader/hrd/service.h"
#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct split_info {
  bitfield traffic_days_;
  interval<unsigned> sections_;
};

struct ref_service {
  ref_service(service const& ref, split_info split)
      : ref_{ref}, split_info_{split} {}

  ref_service(ref_service const& s, unsigned const r) : ref_service{s} {
    repetition_ = r;
  }

  ref_service(ref_service const& s,
              bitfield utc_traffic_days,
              vector<duration_t> utc_times)
      : utc_traffic_days_{utc_traffic_days}, utc_times_{std::move(utc_times)} {}

  service const& ref_;
  split_info split_info_;
  unsigned repetition_;
  bitfield utc_traffic_days_;
  vector<duration_t> utc_times_;
};

}  // namespace nigiri::loader::hrd
