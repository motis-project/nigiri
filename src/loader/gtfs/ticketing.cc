#include "nigiri/loader/gtfs/ticketing.h"

#include <algorithm>

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "utl/parser/csv_range.h"
#include "utl/progress_tracker.h"

constexpr auto kTicketingDeeplinks = "ticketing_deep_links.txt";
constexpr auto kTicketingIdentifiers = "ticketing_identifiers.txt";

namespace nigiri::loader::gtfs {

void read_ticketing_identifiers(timetable& tt,
                                std::string_view file_content,
                                stops_map_t const& stops,
                                source_idx_t const src) {
  struct ticketing_identifiers_row {
    utl::csv_col<utl::cstr, UTL_NAME("ticketing_stop_id")> ticketing_stop_id;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id;
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> agency_id;
  };

  tt.location_ticketing_identifier_.resize(tt.n_locations());

  utl::for_each_row<ticketing_identifiers_row>(
      file_content, [&](ticketing_identifiers_row const& t) {
        auto location_it = stops.find(std::string{t.stop_id->view()});
        if (location_it != stops.end()) {
          auto provider = tt.get_provider_idx(t.agency_id->view(), src);
          string_idx_t const str_idx =
              tt.strings_.store(t.ticketing_stop_id->view());
          tt.location_ticketing_identifier_.at(location_it->second)
              .push_back(pair<provider_idx_t, string_idx_t>{provider, str_idx});
        }
      });

  for (auto loc = location_idx_t{0U};
       loc != location_idx_t{tt.location_ticketing_identifier_.size()}; ++loc) {
    auto bucket = tt.location_ticketing_identifier_[loc];
    utl::sort(bucket, [](auto&& a, auto&& b) { return a.first < b.first; });
  }
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

  auto map = hash_map<std::string_view, ticketing_link_idx_t>{};

  utl::for_each_row<ticketing_deep_links_row>(
      file_content, [&](ticketing_deep_links_row const& t) {
        ticketing_link_idx_t const idx =
            ticketing_link_idx_t{tt.ticketing_links_.web_.size()};

        tt.ticketing_links_.web_.emplace_back(string{t.web_url->view()});
        tt.ticketing_links_.andoid_.emplace_back(
            string{t.android_intent_uri->view()});
        tt.ticketing_links_.ios_.emplace_back(string{t.ios_url->view()});

        map.emplace(t.ticketing_deep_link_id->view(), idx);
      });

  assert(tt.ticketing_links_.web_.size() == tt.ticketing_links_.andoid_.size());
  assert(tt.ticketing_links_.andoid_.size() == tt.ticketing_links_.ios_.size());

  return map;
}

void load_ticketing(timetable& tt,
                    dir const& d,
                    agency_ticketing_map_t const& agency_ticketing,
                    stops_map_t const& stops,
                    route_map_t const& routes,
                    trip_data const& trips,
                    source_idx_t const src) {
  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  utl::get_active_progress_tracker()->status("Load Ticketing");

  read_ticketing_identifiers(tt, load(kTicketingIdentifiers).data(), stops,
                             src);
  auto const deep_links =
      read_ticketing_deep_links(tt, load(kTicketingDeeplinks).data());

  for (auto const& [provider_idx, deep_link_id] : agency_ticketing) {
    tt.providers_[provider_idx].ticketing_link_ = deep_links.at(deep_link_id);
  }

  for (auto const& [route_id, route] : routes) {
    if (!route->ticketing_deep_link_id_.empty()) {
      ticketing_link_idx_t idx = deep_links.at(route->ticketing_deep_link_id_);
      tt.route_ids_[src].route_id_ticketing_link_[route->route_id_idx_] = idx;
    }
  }

  tt.trip_ticketing_identifier_.resize(tt.trip_ticketing_identifier_.size() +
                                       trips.data_.size());

  for (auto const& trip : trips.data_) {
    if (trip.ticketing_unavailable_) {
      tt.trip_ticketing_unavailable_.set(trip.trip_idx_);
    }

    for (auto [idx, unavailable] :
         utl::enumerate(trip.stop_ticketing_unavailable_)) {
      if (unavailable) {
        // Google says:
        // Important: For a particular stop_id, if any of its stop_times has
        // ticketing_type=1, then all of its other stop_times must also have
        // ticketing_type=1.
        //
        // conclusion -> we can store it per location instead of per stop
        // time.
        auto const stop = nigiri::stop{trip.stop_seq_.at(idx)};
        tt.locations_.ticketing_unavailable_.set(stop.location_idx());
      }
    }

    if (!trip.ticketing_trip_id_.empty()) {
      auto const str_idx = tt.strings_.store(trip.ticketing_trip_id_);
      tt.trip_ticketing_identifier_[trip.trip_idx_].push_back(str_idx);
    }
  }
}

}  // namespace nigiri::loader::gtfs
