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
#include "nigiri/query_generator/transport_mode.h"
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
#include "geo/point_rtree.h"
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

bool validate_location_input(std::string const& coord_str, 
                             std::optional<nigiri::location_idx_t> const loc, 
                             std::string_view const label) {
  auto const has_coord = !coord_str.empty();
  auto const has_loc = loc.has_value();

  if (static_cast<int>(has_coord) + static_cast<int>(has_loc) != 1) {
    std::cout << "Error: specify exactly one of " << label << "_coord or "
              << label << "_loc\n";
    return false;
  }
  return true;
}

void add_offsets_for_pos(std::vector<nigiri::routing::offset>& offsets,
                         nigiri::timetable const& tt,
                         geo::point_rtree const& rtree,
                         geo::latlng const& pos,
                         nigiri::query_generation::transport_mode const& mode) {
  for (auto const loc : rtree.in_radius(pos, mode.range())) {
    auto const duration = nigiri::duration_t{
        static_cast<std::int16_t>(
            geo::distance(
                pos, tt.locations_.coordinates_[nigiri::location_idx_t{loc}]) /
            mode.speed_) +
        1};
    offsets.emplace_back(nigiri::location_idx_t{loc}, duration, mode.mode_id_);
  }
}

int main(int argc, char** argv) {
  auto tt_path = std::filesystem::path{};
  auto start_coord_str = std::string{};
  auto dest_coord_str = std::string{};
  auto start_loc_val = nigiri::location_idx_t::value_t{0U};
  auto dest_loc_val = nigiri::location_idx_t::value_t{0U};
  auto start_time_str = std::string{};
  auto start_mode_str = std::string{};
  auto dest_mode_str = std::string{};
  auto intermodal_start_str = std::string{};
  auto intermodal_dest_str = std::string{};
  auto max_transfers = std::uint32_t{nigiri::routing::kMaxTransfers};
  auto prf_idx = std::uint32_t{0U};
  auto use_start_footpaths = true;
  auto astar_transfer_penalty = std::uint32_t{2U};
  auto extend_interval_earlier = true;
  auto extend_interval_later = true;
  auto allowed_claszes = nigiri::routing::all_clasz_allowed();
  auto min_transfer_time = nigiri::duration_t::rep{0U};
  auto transfer_time_factor = 1.0F;

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
      ("start_mode", bpo::value(&start_mode_str)->default_value("intermodal"),
       "intermodal | station")  //
      ("dest_mode", bpo::value(&dest_mode_str)->default_value("intermodal"),
       "intermodal | station")  //
      ("intermodal_start",
       bpo::value(&intermodal_start_str)->default_value("walk"),
       "walk | bicycle | car")  //
      ("intermodal_dest",
       bpo::value(&intermodal_dest_str)->default_value("walk"),
       "walk | bicycle | car")  //
      ("max_transfers",
       bpo::value<std::uint32_t>(&max_transfers)->default_value(max_transfers),
       "maximum number of transfers")  //
      ("profile_idx", bpo::value<std::uint32_t>(&prf_idx)->default_value(0U),
       "footpath profile index")  //
      ("use_start_footpaths",
       bpo::value<bool>(&use_start_footpaths)->default_value(true),
       "use start footpaths (true/false)") //
      ("extend_interval_earlier",
       bpo::value<bool>(&extend_interval_earlier)->default_value(true, "true"),
       "allows extension of the search interval into the past")  //
      ("extend_interval_later",
       bpo::value<bool>(&extend_interval_later)->default_value(true, "true"),
       "allows extension of the search interval into the future")  //
      ("allowed_claszes",
       bpo::value<nigiri::routing::clasz_mask_t>(&allowed_claszes)
           ->default_value(nigiri::routing::all_clasz_allowed()),
       "allowed transport classes bitmask")  //
      ("min_transfer_time",
       bpo::value<nigiri::duration_t::rep>(&min_transfer_time)
           ->default_value(0U),
       "minimum transfer time in minutes")  //
      ("transfer_time_factor",
       bpo::value<float>(&transfer_time_factor)->default_value(1.0F),
       "multiply all transfer times by this factor") //
      ("astar_transfer_penalty",
       bpo::value<std::uint32_t>(&astar_transfer_penalty)
           ->default_value(astar_transfer_penalty),
       "penalty per transfer for A* cost function in minutes");

  auto vm = bpo::variables_map{};
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  bpo::notify(vm);

  auto const start_time =
      nigiri::parse_time(start_time_str, "%FT%R", "%FT%T", "%FT%RZ", "%FT%TZ");

  auto tt = *nigiri::timetable::read(tt_path);
  tt.resolve();
  auto const tt_window = tt.external_interval();
  std::cout << "timetable window: " << date::format("%FT%RZ", tt_window.from_)
            << " -> " << date::format("%FT%RZ", tt_window.to_) << "\n";

  if (start_mode_str != "intermodal" && start_mode_str != "station") {
    std::cout << "Error: Invalid start mode\n";
    return 1;
  }
  if (dest_mode_str != "intermodal" && dest_mode_str != "station") {
    std::cout << "Error: Invalid destination mode\n";
    return 1;
  }

  auto const intermodal_start_mode =
      nigiri::query_generation::to_transport_mode(intermodal_start_str);
  if (!intermodal_start_mode.has_value()) {
    std::cout << "Error: Unknown intermodal start mode\n";
    return 1;
  }

  auto const intermodal_dest_mode =
      nigiri::query_generation::to_transport_mode(intermodal_dest_str);
  if (!intermodal_dest_mode.has_value()) {
    std::cout << "Error: Unknown intermodal destination mode\n";
    return 1;
  }

  if (prf_idx >= nigiri::kNProfiles) {
    std::cout << "Error: profile idx exceeds numeric limits\n";
    return 1;
  }

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

  if (!validate_location_input(start_coord_str, start_loc_input, "start")) {
    return 1;
  }
  if (!validate_location_input(dest_coord_str, dest_loc_input, "dest")) {
    return 1;
  }

  auto q = nigiri::routing::query{};

  q.start_match_mode_ = start_mode_str == "intermodal"
                            ? nigiri::routing::location_match_mode::kIntermodal
                            : nigiri::routing::location_match_mode::kEquivalent;
  q.dest_match_mode_ = dest_mode_str == "intermodal"
                           ? nigiri::routing::location_match_mode::kIntermodal
                           : nigiri::routing::location_match_mode::kEquivalent;

  q.start_time_ = start_time;
  q.extend_interval_earlier_ = extend_interval_earlier;
  q.extend_interval_later_ = extend_interval_later;
  q.use_start_footpaths_ = use_start_footpaths;
  q.max_transfers_ = static_cast<std::uint8_t>(std::min<std::uint32_t>(
      max_transfers, std::numeric_limits<std::uint8_t>::max()));
  q.prf_idx_ = static_cast<nigiri::profile_idx_t>(prf_idx);
  q.allowed_claszes_ = allowed_claszes;
  q.transfer_time_settings_.min_transfer_time_ =
      nigiri::duration_t{min_transfer_time};
  q.transfer_time_settings_.factor_ = transfer_time_factor;
  q.transfer_time_settings_.default_ =
      min_transfer_time == 0U && transfer_time_factor == 1.0F;

  auto const rtree = geo::make_point_rtree(tt.locations_.coordinates_);
  if (!start_coord_str.empty()) {
    auto const start_coord = parse_coord(start_coord_str);
    if (!start_coord.has_value()) {
      std::cout << "Error: invalid start_coord format\n";
      return 1;
    }
    if (q.start_match_mode_ ==
        nigiri::routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.start_, tt, rtree, start_coord.value(),
                          intermodal_start_mode.value());
      if (q.start_.empty()) {
        std::cout << "Error: no start locations in intermodal range\n";
        return 1;
      }
    } else {
      auto const start_loc = find_nearest_location(tt, start_coord.value());
      q.start_.emplace_back(start_loc, std::chrono::minutes{0}, 0U);
    }
  } else {
    if (start_loc_input.value() == nigiri::location_idx_t::invalid() ||
        start_loc_input.value() >= tt.n_locations()) {
      std::cout << "Error: invalid start_loc index\n";
      return 1;
    }
    q.start_.emplace_back(start_loc_input.value(), std::chrono::minutes{0},
                          intermodal_start_mode.value().mode_id_);
  }

  if (!dest_coord_str.empty()) {
    auto const dest_coord = parse_coord(dest_coord_str);
    if (!dest_coord.has_value()) {
      std::cout << "Error: invalid dest_coord format\n";
      return 1;
    }
    if (q.dest_match_mode_ ==
        nigiri::routing::location_match_mode::kIntermodal) {
      add_offsets_for_pos(q.destination_, tt, rtree,
                          dest_coord.value(), intermodal_dest_mode.value());
      if (q.destination_.empty()) {
        std::cout << "Error: no destination locations in intermodal range\n";
        return 1;
      }
    } else {
      auto const dest_loc = find_nearest_location(tt, dest_coord.value());
      q.destination_.emplace_back(dest_loc, std::chrono::minutes{0}, 0U);
    }
  } else {
    if (dest_loc_input.value() == nigiri::location_idx_t::invalid() ||
        dest_loc_input.value() >= tt.n_locations()) {
      std::cout << "Error: invalid dest_loc index\n";
      return 1;
    }
    q.destination_.emplace_back(dest_loc_input.value(), std::chrono::minutes{0},
                                intermodal_dest_mode.value().mode_id_);
  }

  auto search_state = nigiri::routing::search_state{};
  auto const tbd = nigiri::routing::tb::preprocess(tt, q.prf_idx_);

  auto algo_state = nigiri::routing::astar::astar_state{tt, tbd};
  auto const result = nigiri::routing::astar::astar_search(tt, search_state, algo_state, std::move(q), astar_transfer_penalty);

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