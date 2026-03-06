#include "nigiri/export_gtfs.h"

#include <filesystem>
#include <fstream>

#include "nigiri/common/day_list.h"
#include "nigiri/timetable.h"

namespace nigiri {

static std::string format_time(delta d) {
  auto minutes = static_cast<int>(d.count());
  auto h = minutes / 60;
  auto m = minutes % 60;

  char buf[16];
  std::snprintf(buf, sizeof(buf), "%02d:%02d:00", h, m);
  return buf;
}

void export_gtfs(timetable const& tt, std::filesystem::path const& dir) {
  std::filesystem::create_directories(dir);

  write_agencies(tt, dir);
  write_stops(tt, dir);
  write_routes(tt, dir);
  write_trips(tt, dir);
  write_stop_times(tt, dir);
  write_calendar(tt, dir);
  write_calendar_dates(tt, dir);
  write_transfers(tt, dir);
}

void write_agencies(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "agency.txt");

  out << "agency_id,agency_name,agency_url,agency_timezone\n";

  for (provider_idx_t p{0}; p < tt.providers_.size(); ++p) {
    auto const& provider = tt.providers_[p];

    out << to_idx(p) << ",\"" << tt.get_default_translation(provider.name_)
        << "\","
        << "http://example.com,Europe/Berlin\n";
  }
}

void write_stops([[maybe_unused]] timetable const& tt,
                 std::filesystem::path const& output_dir) {
  std::ofstream out(output_dir / "stops.txt");
  out << "stop_id,stop_name,stop_lat,stop_lon,parent_station\n";

  for (location_idx_t l{9}; l < tt.n_locations(); ++l) {
    auto const id = tt.locations_.ids_[l].view();
    auto const name = tt.get_default_name(l);
    auto const coord = tt.locations_.coordinates_[l];

    auto parent = tt.locations_.parents_[l];
    auto parentValue = (parent != location_idx_t::invalid() &&
                        tt.locations_.ids_[parent].view() != id)
                           ? tt.locations_.ids_[parent].view()
                           : "";

    out << id << ",\"" << name << "\"," << coord.lat_ << "," << coord.lng_
        << "," << parentValue << "\n";
  }
}

void write_stop_times(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "stop_times.txt");

  out << "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n";

  for (route_idx_t r{0}; r < tt.n_routes(); ++r) {
    auto const stops = tt.route_location_seq_.at(r);
    auto const transports = tt.route_transport_ranges_.at(r);

    for (transport_idx_t t = transports.from_; t != transports.to_; ++t) {
      for (auto s = stop_idx_t(0); s < stops.size(); ++s) {
        auto loc = stop{stops[s]}.location_idx();

        auto arr = (s == stop_idx_t(0) ? tt.event_mam(t, s, event_type::kDep)
                                       : tt.event_mam(t, s, event_type::kArr));
        auto dep = (s == stop_idx_t(stops.size() - 1)
                        ? tt.event_mam(t, s, event_type::kArr)
                        : tt.event_mam(t, s, event_type::kDep));
        auto trip =
            tt.merged_trips_[tt.transport_to_trip_section_[t].front()].front();

        out << "trip_" << to_idx(trip) << "," << format_time(arr) << ","
            << format_time(dep) << "," << tt.locations_.ids_[loc].view() << ","
            << s << "\n";
      }
    }
  }
}

void write_trips(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "trips.txt");

  out << "route_id,trip_id,service_id,trip_headsign\n";

  for (auto const [id, idx] : tt.trip_id_to_idx_) {
    /* auto const str = tt.trip_id_strings_[id].view(); */
    /* out << str << ": "; */
    for (auto const& t : tt.trip_transport_ranges_.at(idx)) {
      out << tt.transport_route_[t.first] << "," << t.first << "," << t.second
          << "\n";
      /* out << tt.transport_section_attributes_[t.first] << "\n"; */
    }
  }
}
/*
void write_trips(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "trips.txt");

  // TODO check for empty day bitsets trips
  out << "route_id,service_id,trip_id,trip_headsign,direction_id\n";

  // TODO transport idx vs trip idx
  // -> external_trip_ids_[0]
  // check merged_trips_
  // check transport_to_trip_section_
  // TODO change auto t = trip_idx_t(0);
  for (trip_idx_t t{0}; t < tt.n_trips(); ++t) {
    auto route_id = tt.trip_route_id_[t];
    // TODO get source_idx_ from trip
    // -> trip_id_src_[trip_id_idx_t]
    auto route = tt.route_ids_[source_idx_t{0}].ids_.get(route_id);
    auto headsign = tt.get_default_translation(tt.trip_display_names_[t]);
    auto direction = tt.trip_direction_id_[t] ? 1 : 0;

    out << route << ","
        << "service_" << t << "," << t << ","
        << "\"" << headsign << "\"," << direction << "\n";
  }
}
*/

void write_routes(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "routes.txt");

  out << "route_id,agency_id,route_short_name,route_long_name,route_type\n";

  for (source_idx_t s{0}; s < tt.route_ids_.size(); ++s) {
    auto const& routes = tt.route_ids_[s];

    for (route_id_idx_t r{0}; r < routes.ids_.size(); ++r) {
      auto short_name =
          tt.get_default_translation(routes.route_id_short_names_[r]);

      auto long_name =
          tt.get_default_translation(routes.route_id_long_names_[r]);

      auto agency = routes.route_id_provider_[r];
      auto type = to_idx(routes.route_id_type_[r]);

      out << routes.ids_.get(r) << "," << agency << ","
          << "\"" << short_name << "\","
          << "\"" << long_name << "\"," << type << "\n";
    }
  }
}

void write_transfers(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "transfers.txt");

  out << "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n";

  for (location_idx_t l{0}; l < tt.n_locations(); ++l) {
    for (auto const& fp : tt.locations_.footpaths_out_[0][l]) {
      auto to = fp.target();

      // TODO check if transfers contain wheelchair or some
      out << tt.locations_.ids_[l].view() << ","
          << tt.locations_.ids_[to].view() << ","
          << "2," << (fp.duration_ * 60) << "\n";
    }
  }
}

void write_calendar([[maybe_unused]] timetable const& tt,
                    std::filesystem::path const& dir) {
  std::ofstream out(dir / "calendar.txt");
}

void write_calendar_dates([[maybe_unused]] timetable const& tt,
                          std::filesystem::path const& dir) {
  std::ofstream out(dir / "calendar_dates.txt");
}

}  // namespace nigiri
