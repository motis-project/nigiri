#include "boost/program_options.hpp"

#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include "nigiri/common/parse_time.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/routing/tb/tb_search.h"
#include "nigiri/routing/astar/astar_engine.h"
#include "nigiri/routing/astar/astar_search.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include "date/date.h"
#include "geo/latlng.h"
#include "utl/progress_tracker.h"

namespace bpo = boost::program_options;

std::vector<std::string> tokenize(std::string_view const& str,
                                  char const delim,
                                  std::size_t const n_tokens) {
  auto tokens = std::vector<std::string>{};
  tokens.reserve(n_tokens);
  auto start = str.data();
  auto const end = start + str.size();
  auto pos = std::find(start, end, delim);
  while (pos != end) {
    tokens.emplace_back(start, pos);
    start = pos + 1;
    pos = std::find(start, end, delim);
  }
  tokens.emplace_back(start, end);
  return tokens;
}

std::optional<geo::latlng> parse_coord(std::string const& str) {
  static auto const coord_regex =
      std::regex{R"(^\([-+]?[0-9]*\.?[0-9]+,[ \t]*[-+]?[0-9]*\.?[0-9]+\))"};
  if (!std::regex_match(begin(str), end(str), coord_regex)) {
    return std::nullopt;
  }
  auto const str_trimmed = std::string_view{begin(str) + 1, end(str) - 2};
  auto const tokens = tokenize(str_trimmed, ',', 2U);
  auto const parse_token = [](std::string const& token) {
    auto const start = token.find_first_not_of(" \t");
    auto const end = token.find_last_not_of(" \t");
    return std::stod(token.substr(start, end - start + 1));
  };
  return geo::latlng{parse_token(tokens[0]), parse_token(tokens[1])};
}

nigiri::location_idx_t find_nearest_location(nigiri::timetable const& tt,
                                             geo::latlng const& coord) {
  auto best = nigiri::location_idx_t::invalid();
  auto best_dist = std::numeric_limits<double>::max();
  for (auto i = nigiri::location_idx_t{0U}; i != tt.n_locations(); ++i) {
    auto const& pos = tt.locations_.coordinates_[i];
    auto const d_lat = coord.lat_ - pos.lat_;
    auto const d_lng = coord.lng_ - pos.lng_;
    auto const dist = d_lat * d_lat + d_lng * d_lng;
    if (dist < best_dist) {
      best_dist = dist;
      best = i;
    }
  }
  return best;
}

std::optional<nigiri::location_idx_t> resolve_location(
    nigiri::timetable const& tt,
    std::string const& coord_str,
    std::optional<nigiri::location_idx_t> const loc,
    std::string_view const label) {
  auto const has_coord = !coord_str.empty();
  auto const has_loc = loc.has_value();
  auto const provided_count =
      static_cast<int>(has_coord) + static_cast<int>(has_loc);
  if (provided_count != 1) {
    std::cout << "Error: specify exactly one of " << label << "_coord or "
              << label << "_loc\n";
    return std::nullopt;
  }

  if (has_coord) {
    auto const coord = parse_coord(coord_str);
    if (!coord.has_value()) {
      std::cout << "Error: invalid " << label << "_coord format\n";
      return std::nullopt;
    }
    auto const loc = find_nearest_location(tt, coord.value());
    if (loc == nigiri::location_idx_t::invalid()) {
      std::cout << "Error: no " << label << " location found\n";
      return std::nullopt;
    }
    return loc;
  }

  if (has_loc) {
    if (loc.value() == nigiri::location_idx_t::invalid() ||
        loc.value() >= tt.n_locations()) {
      std::cout << "Error: invalid " << label << "_loc index\n";
      return std::nullopt;
    }
    return loc;
  }

  return std::nullopt;
}

int main(int argc, char** argv) {
  auto tt_path = std::filesystem::path{};
  auto start_coord_str = std::string{};
  auto dest_coord_str = std::string{};
  auto start_loc_val = nigiri::location_idx_t::value_t{0U};
  auto dest_loc_val = nigiri::location_idx_t::value_t{0U};
  auto start_time_str = std::string{};
  auto max_transfers = std::uint32_t{nigiri::routing::kMaxTransfers};
  auto prf_idx = std::uint32_t{0U};
  auto use_start_footpaths = true;

  auto desc = bpo::options_description{"Options"};
  desc.add_options()("help,h", "produce this help message")  //
      ("tt_path,p", bpo::value(&tt_path)->required(),
       "path to a binary file containing a serialized nigiri timetable")  //
      ("start_coord", bpo::value(&start_coord_str),
       "start coordinate (e.g. \"(50.767, 6.091)\")")  //
      ("start_loc", bpo::value(&start_loc_val),
       "start location index (internal timetable index)")  //
      ("dest_coord", bpo::value(&dest_coord_str),
       "destination coordinate (e.g. \"(50.777, 6.084)\")")  //
      ("dest_loc", bpo::value(&dest_loc_val),
       "destination location index (internal timetable index)")  //
      ("start_time", bpo::value(&start_time_str)->required(),
       "start time in UTC (e.g. 2026-01-26T11:00 or 2026-01-26T11:00:00Z)")  //
      ("max_transfers",
       bpo::value<std::uint32_t>(&max_transfers)->default_value(max_transfers),
       "maximum number of transfers")  //
      ("profile_idx", bpo::value<std::uint32_t>(&prf_idx)->default_value(0U),
       "footpath profile index")  //
      ("use_start_footpaths",
       bpo::value<bool>(&use_start_footpaths)->default_value(true),
       "use start footpaths (true/false)");

  auto vm = bpo::variables_map{};
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  bpo::notify(vm);

  auto const start_time =
      nigiri::parse_time(start_time_str, "%FT%R", "%FT%T", "%FT%RZ", "%FT%TZ");
  auto const start_window = nigiri::interval<nigiri::unixtime_t>{
      start_time - std::chrono::minutes{5},
      start_time + std::chrono::minutes{25}};

  auto tt = *nigiri::timetable::read(tt_path);
  tt.resolve();
  auto const tt_window = tt.external_interval();
  std::cout << "timetable window: " << date::format("%FT%RZ", tt_window.from_)
            << " -> " << date::format("%FT%RZ", tt_window.to_) << "\n";

  auto const progress_tracker = utl::activate_progress_tracker("tb-single");
  utl::get_global_progress_trackers().silent_ = false;

  auto const start_loc_input =
      vm.count("start_loc") != 0U
          ? std::optional<nigiri::location_idx_t>{nigiri::location_idx_t{
                start_loc_val}}
          : std::nullopt;
  auto const dest_loc_input =
      vm.count("dest_loc") != 0U
          ? std::optional<nigiri::location_idx_t>{nigiri::location_idx_t{
                dest_loc_val}}
          : std::nullopt;
  auto const start_loc =
      resolve_location(tt, start_coord_str, start_loc_input, "start");
  if (!start_loc.has_value()) {
    return 1;
  }
  auto const dest_loc =
      resolve_location(tt, dest_coord_str, dest_loc_input, "dest");
  if (!dest_loc.has_value()) {
    return 1;
  }

  auto q = nigiri::routing::query{};

  q.start_match_mode_ = nigiri::routing::location_match_mode::kEquivalent;
  q.dest_match_mode_ = nigiri::routing::location_match_mode::kEquivalent;

  //q.start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal;
  //q.dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal;

  q.start_time_ = start_window;
  q.extend_interval_earlier_ = true;
  q.extend_interval_later_ = true;
  q.use_start_footpaths_ = use_start_footpaths;
  q.max_transfers_ = static_cast<std::uint8_t>(std::min<std::uint32_t>(
      max_transfers, std::numeric_limits<std::uint8_t>::max()));
  q.prf_idx_ = static_cast<nigiri::profile_idx_t>(prf_idx);
  q.start_.emplace_back(start_loc.value(), std::chrono::minutes{0}, 0U);
  q.destination_.emplace_back(dest_loc.value(), std::chrono::minutes{0}, 0U);

  auto search_state = nigiri::routing::search_state{};
  auto const tbd = nigiri::routing::tb::preprocess(tt, q.prf_idx_);

  auto algo_state = nigiri::routing::astar::astar_state{tt, tbd};
  auto const result = nigiri::routing::astar::astar_search(tt, search_state, algo_state, std::move(q));

  //auto algo_state = nigiri::routing::tb::query_state{tt, tbd};
  //auto const result = nigiri::routing::tb::tb_search(tt, search_state, algo_state, std::move(q));

  if (result.journeys_ == nullptr || result.journeys_->empty()) {
    std::cout << "no journeys found\n";
    return 0;
  }

  std::cout << "journeys: " << result.journeys_->size() << "\n";
  for (auto const& j : *result.journeys_) {
    j.print(std::cout, tt, nullptr, true);
    std::cout << "\n";
  }

  return 0;
}