#include "nigiri/loader/merge_duplicates.h"

#include "utl/zip.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

unsigned get_delta(timetable const& tt,
                   route_idx_t const a_route,
                   route_idx_t const b_route,
                   transport_idx_t const a,
                   transport_idx_t const b) {
  auto const size = tt.route_location_seq_[a_route].size();

  auto delta = 0U;
  for (auto i = stop_idx_t{0U}; i != size; ++i) {
    if (i != 0U) {
      delta += static_cast<unsigned>(
          std::abs(tt.event_mam(a_route, a, i, event_type::kArr).count() -
                   tt.event_mam(b_route, b, i, event_type::kArr).count()));
    }
    if (i != size - 1U) {
      delta += static_cast<unsigned>(
          std::abs(tt.event_mam(a_route, a, i, event_type::kDep).count() -
                   tt.event_mam(b_route, b, i, event_type::kDep).count()));
    }
  }

  return delta;
}

bool merge(timetable& tt,
           stop_idx_t const size,
           transport_idx_t const a,
           transport_idx_t const b) {
  assert(a != b);

  auto const bf_a = tt.bitfields_[tt.transport_traffic_days_[a]];
  auto const bf_b = tt.bitfields_[tt.transport_traffic_days_[b]];
  if ((bf_a & bf_b).none()) {
    return false;
  }

  auto const merge_and_nullify = [&tt](transport_idx_t const x,
                                       transport_idx_t const y) {
    tt.transport_traffic_days_[x] = bitfield_idx_t{0U};  // disable transport x

    for (auto const merged_trips_idx_x : tt.transport_to_trip_section_[x]) {
      for (auto const x_trp : tt.merged_trips_[merged_trips_idx_x]) {
        for (auto& [t, range] : tt.trip_transport_ranges_[x_trp]) {
          if (t == x) {
            t = y;  // replace x with y in x's trip transport ranges
          }
        }
      }
    }
  };

  auto const is_superset = [](bitfield const& x, bitfield const& y) {
    return (x & y) == y;
  };

  if (is_superset(bf_b, bf_a)) {
    merge_and_nullify(a, b);
  } else if (is_superset(bf_a, bf_b)) {
    merge_and_nullify(b, a);
  } else {
    tt.transport_traffic_days_[a] = tt.register_bitfield(bf_a & ~(bf_a & bf_b));

    hash_set<trip_idx_t> b_trips;
    for (auto const merged_trips_idx_b : tt.transport_to_trip_section_[b]) {
      for (auto const b_trp : tt.merged_trips_[merged_trips_idx_b]) {
        for (auto& [t, range] : tt.trip_transport_ranges_[b_trp]) {
          if (t == b) {
            b_trips.emplace(b_trp);
          }
        }
      }
    }

    for (auto const b_trp : b_trips) {
      tt.trip_transport_ranges_[b_trp].push_back(
          transport_range_t{a, {0U, size}});
    }
  }

  return true;
}

unsigned find_duplicates(timetable& tt,
                         location_idx_t const a,
                         location_idx_t const b) {
  auto merged = 0U;
  for (auto const a_route : tt.location_routes_[a]) {
    auto const first_stop_a_route =
        stop{tt.route_location_seq_[a_route].front()}.location_idx();
    if (first_stop_a_route != a) {
      continue;
    }

    auto const a_loc_seq = tt.route_location_seq_[a_route];
    for (auto const& b_route : tt.location_routes_[b]) {
      auto const first_stop_b_route =
          stop{tt.route_location_seq_[b_route].front()}.location_idx();
      if (first_stop_b_route != b) {
        continue;
      }

      auto const b_loc_seq = tt.route_location_seq_[b_route];
      if (a_loc_seq.size() != b_loc_seq.size()) {
        continue;
      }

      auto const station_sequence_matches = [&]() {
        return utl::all_of(utl::zip(a_loc_seq, b_loc_seq), [&](auto&& pair) {
          auto const [x, y] = pair;
          return matches(tt, routing::location_match_mode::kEquivalent,
                         stop{x}.location_idx(), stop{y}.location_idx());
        });
      };

      if (!station_sequence_matches()) {
        continue;
      }

      auto const a_transport_range = tt.route_transport_ranges_[a_route];
      auto const b_transport_range = tt.route_transport_ranges_[b_route];
      auto a_t = begin(a_transport_range), b_t = begin(b_transport_range);

      while (a_t != end(a_transport_range) && b_t != end(b_transport_range)) {
        if (*a_t == *b_t) {
          ++a_t;
          ++b_t;
          continue;
        }

        auto const time_a = tt.event_mam(a_route, *a_t, 0U, event_type::kDep);
        auto const time_b = tt.event_mam(b_route, *b_t, 0U, event_type::kDep);

        if (time_a == time_b) {
          if (get_delta(tt, a_route, b_route, *a_t, *b_t) < a_loc_seq.size()) {
            if (merge(tt, static_cast<stop_idx_t>(a_loc_seq.size()), *a_t,
                      *b_t)) {
              ++merged;
            }
          }
          ++a_t;
          ++b_t;
        } else if (time_a < time_b) {
          ++a_t;
        } else /* time_a > time_b */ {
          ++b_t;
        }
      }
    }
  }
  return merged;
}

}  // namespace nigiri::loader
