#include "nigiri/loader/gtfs/ticketing.h"
#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"

#include "utl/parser/csv_range.h"
#include "utl/progress_tracker.h"

constexpr auto kTicketingDeeplinks = "ticketing_deep_links.txt";
constexpr auto kTicketingIdentifiers = "ticketing_identifiers.txt";

namespace nigiri::loader::gtfs {
void read_ticketing_identifiers(timetable& tt,
                                std::string_view file_content,
                                stops_map_t const& stops) {
  struct ticketing_identifiers_row {
    utl::csv_col<utl::cstr, UTL_NAME("ticketing_stop_id")> ticketing_stop_id;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id;
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> agency_id;
  };

  utl::for_each_row<ticketing_identifiers_row>(
      file_content, [&](ticketing_identifiers_row const& t) {
        auto location_it = stops.find(std::string{t.stop_id->view()});
        if (location_it != stops.end()) {
          tt.location_ticketing_identifier_.emplace_back(
              string{t.ticketing_stop_id->view()});
        }
      });
}

hash_map<std::string_view, ticketing_link_idx_t> read_ticketing_deep_links(
    timetable& tt, std::string_view file_content) {
  struct ticketing_deep_links_row {
    utl::csv_col<utl::cstr, UTL_NAME("ticketing_deep_link_id")>
        ticketing_deep_link_id;
    utl::csv_col<utl::cstr, UTL_NAME("web_url")> web_url;
    utl::csv_col<utl::cstr, UTL_NAME("android_intent_uri")> android_intent_uri;
    utl::csv_col<utl::cstr, UTL_NAME("ios_universal_link_url")> ios_url;
  };

  hash_map<std::string_view, ticketing_link_idx_t> map;

  utl::for_each_row<ticketing_deep_links_row>(
      file_content, [&](ticketing_deep_links_row const& t) {
        ticketing_link_idx_t const idx =
            ticketing_link_idx_t{tt.ticketing_links_.size()};
        timetable::ticketing_link links{string{t.web_url->view()},
                                        string{t.android_intent_uri->view()},
                                        string{t.ios_url->view()}};
        tt.ticketing_links_.emplace_back({std::move(links)});
        map.emplace(t.ticketing_deep_link_id->view(), idx);
      });

  return map;
}

void load_ticketing(timetable& tt,
                    dir const& d,
                    agency_ticketing_map_t const& agency_ticketing,
                    stops_map_t const& stops,
                    route_map_t const& routes,
                    trip_data const& trips) {
  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  utl::get_active_progress_tracker()->status("Load Ticketing");

  read_ticketing_identifiers(tt, load(kTicketingIdentifiers).data(), stops);
  auto const deep_links =
      read_ticketing_deep_links(tt, load(kTicketingDeeplinks).data());

  for (auto const& [provider_idx, deep_link_id] : agency_ticketing) {
    tt.ticketing_agencies.emplace(provider_idx, deep_links.at(deep_link_id));
  }

  for (auto const& [route_id, route] : routes) {
    if (!route->ticketing_deep_link_id_.empty()) {
      ticketing_link_idx_t idx = deep_links.at(route->ticketing_deep_link_id_);
      tt.ticketing_routes_.emplace(route->route_id_idx_, idx);
    }
  }

  for (auto const& trip : trips.data_) {
    if (trip.ticketing_unavailable_) {
      tt.trip_ticketing_unavailable_.emplace(trip.trip_idx_, std::monostate{});
    }

    for (auto [idx, unavailable] :
         utl::enumerate(trip.stop_ticketing_unavailable_)) {
      if (unavailable) {
        // Google says:
        // Important: For a particular stop_id, if any of its stop_times has
        // ticketing_type=1, then all of its other stop_times must also have
        // ticketing_type=1.
        //
        // conclusion -> we can store it per location instead of per stop time.
        auto const stop = nigiri::stop{trip.stop_seq_.at(idx)};
        tt.location_ticketing_unavailable_.emplace(stop.location_,
                                                   std::monostate{});
      }
    }
  }
}

}  // namespace nigiri::loader::gtfs