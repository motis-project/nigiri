#pragma once

#include <cassert>
#include <ranges>

#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

using rule_trip_idx_t = cista::strong<std::uint32_t, struct _rule_trip_idx>;
using remaining_idx_t = unsigned;  // i where expanded_.data_[i]

template <typename UtcTrip, typename OriginIdx>
struct expanded_seated {
  vecvec<rule_trip_idx_t, rule_trip_idx_t> seated_in_;
  vecvec<rule_trip_idx_t, rule_trip_idx_t> seated_out_;
  vecvec<rule_trip_idx_t, UtcTrip> expanded_;
  vector_map<remaining_idx_t, rule_trip_idx_t> remaining_rule_trip_;
};

template <typename UtcTrip, typename OriginIdx>
void build_seated_trips(timetable&,
                        expanded_seated<UtcTrip, OriginIdx>&,
                        std::function<void(UtcTrip&&)> const& consumer);

template <typename Fn>
expanded_seated<gtfs::utc_trip, gtfs_trip_idx_t> expand_seated_trips(
    trip_data const& trip_data, Fn&& expand) {
  auto ret = expanded_seated<utc_trip, gtfs_trip_idx_t>{};

  auto gtfs_trip_rule_trip = vector_map<gtfs_trip_idx_t, rule_trip_idx_t>{};
  auto rule_trip_gtfs_trip = vector_map<rule_trip_idx_t, gtfs_trip_idx_t>{};
  gtfs_trip_rule_trip.resize(trip_data.data_.size(),
                             rule_trip_idx_t::invalid());
  for (auto const [i, trp] : utl::enumerate(trip_data.data_)) {
    if (trp.has_seated_transfers()) {
      auto const rule_trip = rule_trip_idx_t{rule_trip_gtfs_trip.size()};
      auto const gtfs_trip = gtfs_trip_idx_t{i};
      rule_trip_gtfs_trip.emplace_back(gtfs_trip);
      gtfs_trip_rule_trip[gtfs_trip] = rule_trip;
    }
  }

  for (auto const [i, gtfs_trip] : utl::enumerate(rule_trip_gtfs_trip)) {
    using std::views::transform;
    auto const to_rule_trip_idx = [&](gtfs_trip_idx_t const x) {
      return gtfs_trip_rule_trip.at(x);
    };
    ret.seated_out_.emplace_back(trip_data.get(gtfs_trip).seated_out_ |
                                 transform(to_rule_trip_idx));
    ret.seated_in_.emplace_back(trip_data.get(gtfs_trip).seated_in_ |
                                transform(to_rule_trip_idx));
  }

  for (auto const [rule_trip, t] : utl::enumerate(rule_trip_gtfs_trip)) {
    auto bucket = ret.expanded_.add_back_sized(0U);
    expand(t, [&](utc_trip&& s) {
      assert(s.trips_.size() == 1 && s.trips_.front() == t);
      assert(s.utc_times_.size() >= 2U);
      s.rule_trip_idx_ = rule_trip;
      bucket.push_back(std::move(s));
      ret.remaining_rule_trip_.push_back(rule_trip_idx_t{rule_trip});
    });
  }

  return ret;
}
}  // namespace nigiri::loader::gtfs