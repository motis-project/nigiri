#include "nigiri/loader/merge_duplicates.h"

#include <ranges>

#include "utl/pairwise.h"
#include "utl/zip.h"

#include "nigiri/common/day_list.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/resolve.h"
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

  return true;
}

void find_intra_route_duplicates(timetable& tt,
                                 merge_threshold_t const& clasz_threshold) {
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const threshold =
        clasz_threshold[static_cast<unsigned>(tt.route_clasz_[r])];
    auto const transports = tt.route_transport_ranges_[r];
    for (auto a = begin(transports); a != end(transports); ++a) {
      auto const time_a = tt.event_mam(r, *a, 0U, event_type::kDep);

      auto init_b = a;
      ++init_b;
      for (auto b = init_b; b != end(transports); ++b) {
        auto const time_b = tt.event_mam(r, *b, 0U, event_type::kDep);
        assert(time_b > time_a);

        auto const& bf_a = tt.bitfields_[tt.transport_traffic_days_[*a]];
        auto const& bf_b = tt.bitfields_[tt.transport_traffic_days_[*b]];
        auto const intersection = bf_a & bf_b;
        if (intersection.none()) {
          continue;
        }

        if ((time_b.as_duration() - time_a.as_duration()) >= threshold) {
          break;
        }

        auto const delta = get_delta(tt, r, r, *a, *b);
        auto const loc_seq = tt.route_location_seq_[r];

        if (delta < loc_seq.size() * threshold) {
          std::clog << "  " << tt.trip_id(*a)
                    << " [name=" << tt.transport_name(*b) << "] vs "
                    << tt.trip_id(*b) << " [nam e=" << tt.transport_name(*b)
                    << "], time_a=" << time_a << ", time_b=" << time_b
                    << ", MIN_DIST=" << threshold << ", FIRST_DIFF="
                    << (time_b.as_duration() - time_a.as_duration())
                    << ", DELTA=" << delta << ", days=" << tt.days(intersection)
                    << "\n";
        }
      }
    }
  }
}

unsigned merge_duplicates(timetable& tt,
                          location_idx_t const a,
                          location_idx_t const b) {
  // http://localhost:5173/?tripId=20260115_13%3A07_de-DELFI_3064419662&motis=localhost%3A8080
  auto const needle1 =
      tt.find(location_id{"de:12060:900350694::1", source_idx_t{0U}});
  utl::verify(needle1.has_value(), "location de-DELFI_3064419536 not found");

  // http://localhost:5173/?tripId=20260115_13%3A07_de-VBB_283184597&motis=localhost%3A8080
  auto const needle2 =
      tt.find(location_id{"de:12060:900350354::1", source_idx_t{1U}});
  utl::verify(needle2.has_value(), "location de-VBB_283183671 not found");

  auto const fr1 =
      resolve(tt, nullptr, source_idx_t{0}, "3064419662", "20260114", "13:07");
  utl::verify(fr1.valid(), "3064419662 not found");

  auto const fr2 =
      resolve(tt, nullptr, source_idx_t{1}, "283184597", "20260114", "13:07");
  utl::verify(fr2.valid(), "283184597 not found");

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

      auto const debug = (a_transport_range.contains(fr1.t_.t_idx_) &&
                          b_transport_range.contains(fr2.t_.t_idx_)) ||
                         (a_transport_range.contains(fr2.t_.t_idx_) &&
                          b_transport_range.contains(fr1.t_.t_idx_));

      auto const print_route = [&](route_idx_t const r,
                                   interval<transport_idx_t> const& x) {
        for (auto const t : x) {
          std::clog << "  trip_id=" << tt.trip_id(t)
                    << ", time=" << tt.event_mam(r, t, 0U, event_type::kDep)
                    << ", name=" << tt.transport_name(t) << "\n";
        }
      };
      if (debug) {
        std::clog << "ROUTE " << a << " VS " << b << "\n";
        std::clog << "ROUTE_A\n";
        print_route(a_route, a_transport_range);
        std::clog << "ROUTE_B\n";
        print_route(b_route, b_transport_range);
      }

      while (a_t != end(a_transport_range) && b_t != end(b_transport_range)) {
        if (*a_t == *b_t) {
          ++a_t;
          ++b_t;
          continue;
        }

        auto const time_a = tt.event_mam(a_route, *a_t, 0U, event_type::kDep);
        auto const time_b = tt.event_mam(b_route, *b_t, 0U, event_type::kDep);

        if (time_a == time_b) {
          auto const delta = get_delta(tt, a_route, b_route, *a_t, *b_t);
          if (delta < a_loc_seq.size()) {
            if (merge(tt, static_cast<stop_idx_t>(a_loc_seq.size()), *a_t,
                      *b_t)) {
              if (debug) {
                std::clog << "MERGED " << tt.trip_id(*a_t)
                          << " [name=" << tt.transport_name(*a_t) << "] vs "
                          << tt.trip_id(*b_t)
                          << " [name=" << tt.transport_name(*b_t)
                          << "], delta=" << delta << ", time_a=" << time_a
                          << ", time_b=" << time_b << "\n";
              }
              ++merged;
            }
          }
          ++a_t;
          ++b_t;
        } else if (time_a < time_b) {
          if (debug) {
            std::clog << "TIME MISMATCH time_a < time_b " << tt.trip_id(*a_t)
                      << " [name=" << tt.transport_name(*a_t) << "] vs "
                      << tt.trip_id(*b_t)
                      << " [name=" << tt.transport_name(*b_t)
                      << "], time_a=" << time_a << ", time_b=" << time_b
                      << "\n";
          }
          ++a_t;
        } else /* time_a > time_b */ {
          if (debug) {
            std::clog << "TIME MISMATCH time_a > time_b " << tt.trip_id(*a_t)
                      << " [name=" << tt.transport_name(*a_t) << "] vs "
                      << tt.trip_id(*b_t)
                      << " [name=" << tt.transport_name(*b_t)
                      << "], time_a=" << time_a << ", time_b=" << time_b
                      << "\n";
          }
          ++b_t;
        }
      }

      for (; a_t != end(a_transport_range); ++a_t) {
        auto const time = tt.event_mam(a_route, *a_t, 0U, event_type::kDep);
        if (debug) {
          std::clog << "TIME MISMATCH OVERFLOW A: " << tt.trip_id(*a_t)
                    << " [name=" << tt.transport_name(*a_t)
                    << "], time=" << time << "\n";
        }
      }

      for (; b_t != end(b_transport_range); ++b_t) {
        auto const time = tt.event_mam(a_route, *a_t, 0U, event_type::kDep);
        if (debug) {
          std::clog << "TIME MISMATCH OVERFLOW B: " << tt.trip_id(*b_t)
                    << " [name=" << tt.transport_name(*b_t)
                    << "], time=" << time << "\n";
        }
      }
    }
  }
  return merged;
}

}  // namespace nigiri::loader
