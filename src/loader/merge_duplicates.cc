#include "nigiri/loader/merge_duplicates.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "utl/enumerate.h"
#include "utl/zip.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include <algorithm>
#include <iostream>
#include <iterator>

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

inline uint16_t parse_train_number(std::string_view const s) {
  auto const first = std::find_if(begin(s), end(s),
                                  [](char const c) { return std::isdigit(c); });

  if (first == end(s)) {
    return 0;
  }

  int v = 0;
  std::from_chars(first, end(s), v);

  return v;
}

uint16_t trip_train_nr(timetable const& tt,
                       route_idx_t route,
                       transport_idx_t transport,
                       event_type ev_type) {
  auto const& trip_sections = tt.transport_to_trip_section_[transport];

  auto const i = ev_type == event_type::kArr ? 0 : trip_sections.size() - 1;

  auto const clasz = tt.route_section_clasz_[route][i];
  // switch (clasz) {
  //   case clasz::kHighSpeed:
  //   case clasz::kLongDistance:
  //   case clasz::kNight:
  //   case clasz::kRegional:
  //   case clasz::kSuburban: break;
  //   default: return 0;
  // }

  auto const trip = tt.merged_trips_[trip_sections[i]].front();
  auto const number = tt.get_default_translation(tt.trip_short_names_[trip]);

  return parse_train_number(number);
}

uint16_t route_train_nr(timetable const& tt,
                        route_idx_t route,
                        event_type ev_type) {
  auto const first_transport = tt.route_transport_ranges_[route].from_;
  return trip_train_nr(tt, route, first_transport, ev_type);
}

delta route_departure(timetable const& tt,
                      route_idx_t a_route,
                      transport_idx_t transport) {
  auto const first_stop = tt.route_location_seq_[a_route].front();
  return tt.event_mam(a_route, transport, first_stop, event_type::kDep);
}

void concat_trains(timetable& tt,
                   location_idx_t a_loc,
                   location_idx_t b_loc,
                   route_idx_t a_route,
                   route_idx_t b_route,
                   transport_idx_t a_transport,
                   transport_idx_t b_transport) {
  auto const a_start_time = route_departure(tt, a_route, a_transport);
  auto const b_start_time = route_departure(tt, b_route, b_transport);

  // whether a or b is first
  auto const a_first = a_start_time < b_start_time;

  // ordered routes and transports by time. Earlier first.
  auto const [first_route, second_route] =
      a_first ? std::pair{a_route, b_route} : std::pair{b_route, a_route};
  auto const [first_transport, second_transport] =
      a_first ? std::pair{a_transport, b_transport}
              : std::pair{b_transport, a_transport};

  // find connecting stop
  auto const first_seq = tt.route_location_seq_[first_route];
  auto const second_seq = tt.route_location_seq_[second_route];

  auto first_end_i = -1;
  auto second_start_i = -1;

  for (auto const [i, s] : utl::enumerate(first_seq)) {
    stop st{s};
    if (st.location_idx() == a_loc || st.location_idx() == b_loc) {
      first_end_i = i;
    }
  }

  // second trip is contained in first already
  if (first_end_i == 0) {
    std::cerr << "Trip already complete" << std::endl;
    return;
  }

  for (auto const [i, s] : utl::enumerate(second_seq)) {
    stop st{s};
    if (st.location_idx() == a_loc || st.location_idx() == b_loc) {
      second_start_i = i;
    }
  }

  // Otherwise they wouldn't really pass our location we searched from
  assert(first_end_i != -1 && second_start_i != -1);

  // stop sequence
  basic_string<stop::value_type> new_stop_seq;
  new_stop_seq.reserve(first_seq.size() + second_seq.size());

  std::for_each(std::begin(first_seq), std::begin(first_seq) + first_end_i,
                [&](auto const s) { new_stop_seq.push_back(s); });

  std::for_each(std::begin(second_seq) + second_start_i, std::end(second_seq),
                [&](auto const s) { new_stop_seq.push_back(s); });

  // section claszes
  auto const first_claszs = tt.route_section_clasz_[first_route];
  auto const second_claszs = tt.route_section_clasz_[second_route];

  basic_string<clasz> new_section_clasz;
  new_section_clasz.reserve(first_claszs.size() + second_claszs.size());

  for (auto const c : first_claszs) {
    new_section_clasz.push_back(c);
  }

  for (auto const c : second_claszs) {
    new_section_clasz.push_back(c);
  }

  // flags
  std::array<bitvec, route_flag::kNumRouteFlags> flags_per_section;

  for (size_t f = 0; f < route_flag::kNumRouteFlags; f++) {
    auto const first_sections = tt.route_flags_per_section_[f][first_route];
    auto const second_sections = tt.route_flags_per_section_[f][second_route];

    flags_per_section[f].resize(first_sections.size() + second_sections.size());

    auto const second_section_begin = first_sections.size();
    for (auto const [section_i, flag] : utl::enumerate(first_sections)) {
      if (flag) {
        flags_per_section[f].set(section_i);
      }
    }

    for (auto const [section_i, flag] : utl::enumerate(second_sections)) {
      if (flag) {
        flags_per_section[f].set(second_section_begin + section_i);
      }
    }
  }

  // traffic days
  auto const first_bitfield =
      tt.bitfields_[tt.transport_traffic_days_[first_transport]];
  auto const second_bitfield =
      tt.bitfields_[tt.transport_traffic_days_[second_transport]];

  // TODO stop if intersection empty
  auto const common_bitfield = first_bitfield & second_bitfield;
  if (common_bitfield.none()) {
    std::cerr << "No common traffic days :(" << std::endl;
    return;
  }

  auto const common_bitfield_idx = tt.register_bitfield(common_bitfield);

  // departure time
  auto const first_dep_offset =
      tt.transport_first_dep_offset_.at(first_transport);

  // external trip ids
  basic_string<merged_trips_idx_t> external_trip_ids;

  auto const first_ids = tt.transport_to_trip_section_[first_transport];
  auto const second_ids = tt.transport_to_trip_section_[second_transport];

  external_trip_ids.reserve(first_ids.size() + second_ids.size());

  for (auto const trip_id : first_ids) {
    external_trip_ids.push_back(trip_id);
  }
  for (auto const trip_id : second_ids) {
    external_trip_ids.push_back(trip_id);
  }

  // section attributes
  basic_string<attribute_combination_idx_t> section_attributes;

  auto const first_attributes =
      tt.transport_section_attributes_[first_transport];
  auto const second_attributes =
      tt.transport_section_attributes_[second_transport];

  section_attributes.reserve(first_attributes.size() +
                             second_attributes.size());

  for (auto const ac : first_attributes) {
    section_attributes.push_back(ac);
  }
  for (auto const ac : second_attributes) {
    section_attributes.push_back(ac);
  }

  // providers
  basic_string<provider_idx_t> section_providers;

  auto const first_providers = tt.transport_section_providers_[first_transport];
  auto const second_providers =
      tt.transport_section_providers_[second_transport];
  section_providers.reserve(first_providers.size() + second_providers.size());

  for (auto const p : first_providers) {
    section_providers.push_back(p);
  }
  for (auto const p : second_providers) {
    section_providers.push_back(p);
  }

  // directions
  basic_string<translation_idx_t> section_directions;

  auto const first_directions =
      tt.transport_section_directions_[first_transport];
  auto const second_directions =
      tt.transport_section_directions_[second_transport];

  section_directions.reserve(first_directions.size() +
                             second_directions.size());

  for (auto const d : first_directions) {
    section_directions.push_back(d);
  }
  for (auto const d : second_directions) {
    section_directions.push_back(d);
  }

  auto const new_route_idx =
      tt.register_route(new_stop_seq, new_section_clasz, flags_per_section);

  // TODO shape bboxes

  tt.add_transport(
      timetable::transport{.bitfield_idx_ = common_bitfield_idx,
                           .route_idx_ = new_route_idx,
                           .first_dep_offset_ = first_dep_offset,
                           .external_trip_ids_ = external_trip_ids,
                           .section_attributes_ = section_attributes,
                           .section_providers_ = section_providers,
                           .section_directions_ = section_directions});

  tt.finish_route();

  auto const stop_times_begin = tt.route_stop_times_.size();

  for (auto const [from, to] : utl::pairwise(
           interval{std::size_t{0U}, static_cast<size_t>(first_end_i - 1)})) {
    tt.route_stop_times_.emplace_back(
        tt.event_mam(first_transport, first_seq[from], event_type::kDep));
    tt.route_stop_times_.emplace_back(
        tt.event_mam(first_transport, first_seq[to], event_type::kArr));
  }
  for (auto const [from, to] :
       utl::pairwise(interval{static_cast<size_t>(second_start_i),
                              static_cast<size_t>(second_seq.size())})) {
    tt.route_stop_times_.emplace_back(
        tt.event_mam(first_transport, first_seq[from], event_type::kDep));
    tt.route_stop_times_.emplace_back(
        tt.event_mam(first_transport, first_seq[to], event_type::kArr));
  }

  auto const stop_times_end = tt.route_stop_times_.size();
  tt.route_stop_time_ranges_.emplace_back(
      interval{stop_times_begin, stop_times_end});

  // HACK add to location routes. This is probably horribly inefficient right
  // now as we are moving everything afterwards.
  std::for_each(
      std::begin(first_seq), std::begin(first_seq) + first_end_i,
      [&](auto const s) {
        tt.location_routes_[stop{s}.location_idx()].push_back(new_route_idx);
      });

  std::for_each(
      std::begin(second_seq) + second_start_i, std::end(second_seq),
      [&](auto const s) {
        tt.location_routes_[stop{s}.location_idx()].push_back(new_route_idx);
      });

  std::cerr << "merged trains"
            << trip_train_nr(tt, new_route_idx,
                             transport_idx_t{tt.transport_route_.size() - 1},
                             event_type::kDep)
            << std::endl;
}

void connect_by_train_nr(timetable& tt,
                         location_idx_t const a,
                         location_idx_t const b) {
  if (a == b) {
    return;
  }

  // a and b are suspected equivalent locations
  for (auto const a_route : tt.location_routes_[a]) {
    // only evaluate for each route once at the station where it starts.
    auto const first_stop_a_route =
        stop{tt.route_location_seq_[a_route].front()}.location_idx();
    if (first_stop_a_route != a) {
      continue;
    }

    auto a_transports = tt.route_transport_ranges_[a_route];

    for (auto const a_transport : a_transports) {
      // TODO consider trying departure and arrival
      auto const train_nr_a =
          trip_train_nr(tt, a_route, a_transport, event_type::kArr);
      if (train_nr_a == 0) {
        std::cerr << "1" << std::endl;
        continue;
      }

      for (auto const& b_route : tt.location_routes_[b]) {
        auto const first_stop_b_route =
            stop{tt.route_location_seq_[b_route].front()}.location_idx();
        // if (first_stop_b_route != b) {
        //   std::cerr << "4" << std::endl;
        //
        //   continue;
        // }

        auto b_transports = tt.route_transport_ranges_[b_route];

        for (auto const& b_transport : b_transports) {
          if (a_transport == b_transport) {
            std::cerr << "3" << std::endl;
            continue;
          }

          // TODO consider trying departure and arrival
          auto const train_nr_b =
              trip_train_nr(tt, b_route, b_transport, event_type::kArr);
          std::cerr << "Saw " << train_nr_a << " " << train_nr_b << std::endl;

          if (train_nr_a == train_nr_b) {
            std::cerr << "Trying to connect trains with number " << train_nr_a
                      << std::endl;
            concat_trains(tt, a, b, a_route, b_route, a_transport, b_transport);
          }
        }
      }
    }
  }
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
