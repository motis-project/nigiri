#pragma once

#include <cassert>

#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

struct expanded_seated {
  using seated_idx_t = cista::strong<std::uint32_t, struct _rule_service_idx>;
  vector_map<seated_idx_t, gtfs_trip_idx_t> seated_;
  hash_map<gtfs_trip_idx_t, seated_idx_t> ref_;
  vecvec<seated_idx_t, utc_trip> expanded_;
};

std::vector<utc_trip> build_seated_trips(timetable const&,
                                         trip_data const&,
                                         expanded_seated&);

template <typename Fn>
expanded_seated expand_seated_trips(trip_data const& trip_data, Fn&& expand) {
  auto ret = expanded_seated{};

  for (auto const [i, trp] : utl::enumerate(trip_data.data_)) {
    if (trp.has_seated_transfers()) {
      auto const seated_idx = expanded_seated::seated_idx_t{ret.seated_.size()};
      ret.seated_.emplace_back(gtfs_trip_idx_t{i});
      ret.ref_.emplace(gtfs_trip_idx_t{i}, seated_idx);
    }
  }

  for (auto const& t : ret.seated_) {
    auto bucket = ret.expanded_.add_back_sized(0U);
    expand(t, [&](utc_trip&& s) {
      assert(s.trips_.size() == 1 && s.trips_.front() == t);
      bucket.push_back(std::move(s));
    });
  }

  return ret;
}
}  // namespace nigiri::loader::gtfs