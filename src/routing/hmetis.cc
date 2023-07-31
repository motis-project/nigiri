#include "nigiri/routing/hmetis.h"

#include "utl/enumerate.h"

#include "nigiri/timetable.h"
#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

using namespace std::string_view_literals;

namespace nigiri::routing {

void write_hmetis_file(std::ostream& out, timetable const& tt) {
  auto const n_non_empty_locations =
      std::count_if(begin(tt.location_routes_), end(tt.location_routes_),
                    [](auto const& routes) { return !routes.empty(); });

  out << n_non_empty_locations << " " << tt.n_routes() << "\n";
  for (auto const& [l, routes] : utl::enumerate(tt.location_routes_)) {
    if (routes.empty()) {
      continue;
    }
    for (auto const r : routes) {
      out << (r + 1) << " ";
    }
    out << " % " << tt.locations_.names_[location_idx_t{l}].view();
    out << "\n";
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
  auto r = route_idx_t{1U};

  out << R"({
"type": "FeatureCollection",
"features": [)";
  utl::for_each_line(in, [&](utl::cstr const line) {
    auto const partition = utl::parse<unsigned>(line);

    out << fmt::format(marker_fmt_str_start, kColors[partition]);
    auto first = true;
    for (auto const& stp : tt.route_location_seq_[r - 1]) {
      if (!first) {
        out << ", ";
      }
      first = false;
      auto const pos = tt.locations_.coordinates_[stop{stp}.location_idx()];
      out << "[" << pos.lng_ << ", " << pos.lat_ << "]";
    }
    out << fmt::format(marker_fmt_str_end, r);
    
    ++r;
  });
  out << "  ]\n"
         "}";
}

}  // namespace nigiri::routing