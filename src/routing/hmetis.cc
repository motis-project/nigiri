#include "nigiri/routing/hmetis.h"

#include "geo/box.h"

#include "utl/enumerate.h"
#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

#include "nigiri/timetable.h"

using namespace std::string_view_literals;

namespace nigiri::routing {

void write_hmetis_file(std::ostream& out, timetable const& tt) {
  auto const location_has_routes = [&](auto const& locations) {
    return std::any_of(begin(locations), end(locations),
                       [&](location_idx_t const l) {
                         return !tt.location_routes_[l].empty();
                       });
  };

  auto const n_non_empty_locations = std::count_if(
      begin(tt.locations_.component_locations_),
      end(tt.locations_.component_locations_),
      [&](auto const& locations) { return location_has_routes(locations); });

  out << n_non_empty_locations << " " << tt.n_routes() << " 11\n";

  for (auto const& locations : tt.locations_.component_locations_) {
    if (!location_has_routes(locations)) {
      continue;
    }

    auto n_stop_events = 0U;
    for (auto const l : locations) {
      for (auto const r : tt.location_routes_[l]) {
        for (auto const t : tt.route_transport_ranges_[r]) {
          n_stop_events += tt.bitfields_[tt.transport_traffic_days_[t]].count();
        }
      }
    }
    out << static_cast<unsigned>(std::round(std::log2(n_stop_events))) << " ";

    for (auto const l : locations) {
      for (auto const r : tt.location_routes_[l]) {
        out << (r + 1) << " ";
      }
    }
    out << "\n";
  }

  for (auto const route_transports : tt.route_transport_ranges_) {
    auto runs = 0U;
    for (auto const t : route_transports) {
      runs += tt.bitfields_[tt.transport_traffic_days_[t]].count();
    }
    out << runs << "\n";
  }
}

constexpr auto const marker_fmt_str_start = R"({{
"type": "Feature",
"properties": {{
  "stroke": "{}"
}},
"geometry": {{
  "coordinates": [)";
constexpr auto const marker_fmt_str_end = R"(],
  "type": "LineString"
}},
"id": {}
}},)";

constexpr std::array<std::string_view, 8> kColors = {
    "red"sv,  "blue"sv, "green"sv,  "yellow"sv,
    "cyan"sv, "pink"sv, "orange"sv, "black"sv};

void hmetis_out_to_geojson(std::string_view in,
                           std::ostream& out,
                           timetable const& tt) {

  auto const print_pos = [&](geo::latlng const p) {
    out << "[" << p.lng_ << ", " << p.lat_ << "]";
  };

  auto const print_marker = [&](geo::latlng const& pos) {
    fmt::print(out, R"({{
  "type": "Feature",
  "properties": {{}},
  "geometry": {{
    "coordinates": [ {}, {} ],
    "type": "Point"
  }}
}},)",
               pos.lng_, pos.lat_);
  };

  auto route_partitions = vector_map<route_idx_t, partition_idx_t>{};
  utl::for_each_line(in, [&](utl::cstr line) {
    route_partitions.emplace_back(utl::parse<unsigned>(line));
  });
  utl::verify(route_partitions.size() == tt.n_routes(),
              "invalid partitions size n_route_partitions={} vs n_routes={}",
              route_partitions.size(), tt.n_routes());

  out << R"({
  "type": "FeatureCollection",
  "features": [)";

  // Print line strings for route sequences.
  for (auto const [r, partition] : utl::enumerate(route_partitions)) {
    out << fmt::format(marker_fmt_str_start, kColors[partition]);
    auto first = true;
    for (auto const& stp : tt.route_location_seq_[route_idx_t{r}]) {
      if (!first) {
        out << ", ";
      }
      first = false;
      print_pos(tt.locations_.coordinates_[stop{stp}.location_idx()]);
    }
    out << fmt::format(marker_fmt_str_end, r);
  }

  // Print markers for cut locations.
  auto n_cut_components = 0U;
  for (auto const& locations : tt.locations_.component_locations_) {
    std::optional<partition_idx_t> partition = std::nullopt;
    for (auto const l : locations) {
      for (auto const r : tt.location_routes_[l]) {
        if (partition.has_value() && *partition != route_partitions[r]) {
          print_marker(tt.locations_.coordinates_[l]);
          ++n_cut_components;
          goto next;
        } else if (!partition.has_value()) {
          partition = route_partitions[r];
        }
      }
    }
  next:;
  }
  std::cout << "n_cut_components: " << n_cut_components << "\n";

  // Print bounding boxes of location components.
  for (auto const& locations : tt.locations_.component_locations_) {
    geo::box bbox;
    for (auto const l : locations) {
      bbox.extend(tt.locations_.coordinates_[l]);
    }

    out << R"({
"type": "Feature",
"properties": {
  "fill": "#000000",
  "fill-opacity": 0.7
},
"geometry": {
  "coordinates": [
    [)";
    print_pos(bbox.min_);
    out << ", ";
    print_pos({bbox.min_.lat_, bbox.max_.lng_});
    out << ", ";
    print_pos(bbox.max_);
    out << ", ";
    print_pos({bbox.max_.lat_, bbox.min_.lng_});
    out << ", ";
    print_pos(bbox.min_);

    out << R"(]
],
"type": "Polygon"
},
"id": 0
},)";
  }

  out << "  ]\n"
         "}";
}

}  // namespace nigiri::routing