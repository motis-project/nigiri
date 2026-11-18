#include "nigiri/loader/gtfs/pathways.h"

#include <cmath>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "geo/latlng.h"

#include "utl/get_or_create.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"

#include "nigiri/loader/station_graph.h"
#include "nigiri/logging.h"

namespace nigiri::loader::gtfs {

// Durations if pathways.txt gives no traversal_time.
constexpr auto const kPathwayWalkSpeed = 1.2;  // m/s, as osr foot
constexpr auto const kSecondsPerStair = 0.6;
constexpr auto const kStairsSlowdown = 2.0;  // on length, without stair_count
constexpr auto const kEscalatorSpeed = 0.5;  // m/s
constexpr auto const kDefaultEscalatorSeconds = 30U;
constexpr auto const kDefaultElevatorSeconds = 90U;  // incl. waiting, as osr

enum class pathway_mode : std::uint8_t {
  kWalkway = 1U,
  kStairs = 2U,
  kTravelator = 3U,
  kEscalator = 4U,
  kElevator = 5U,
  kFareGate = 6U,
  kExitGate = 7U
};

enum class gtfs_location_type : std::uint8_t {
  kStop = 0U,
  kStation = 1U,
  kEntrance = 2U,
  kGeneric = 3U,
  kBoardingArea = 4U
};

std::optional<double> parse_optional(utl::cstr const s) {
  auto const trimmed = s.trim();
  return trimmed.empty() ? std::nullopt
                         : std::optional{utl::parse<double>(trimmed)};
}

struct node_row {
  std::string_view parent_;
  std::string_view level_;
  std::string_view name_;
  std::optional<geo::latlng> pos_;
  gtfs_location_type type_;
  bool is_street_accessible_;
};

struct pathway_row {
  std::string_view from_;
  std::string_view to_;
  pathway_mode mode_;
  bool is_bidirectional_;
  std::optional<double> length_;
  std::optional<double> traversal_time_;
  std::optional<double> stair_count_;
  std::optional<double> max_slope_;
  std::optional<double> min_width_;
  std::string_view signposted_as_;
  std::string_view reversed_signposted_as_;
};

hash_map<std::string_view, float> read_levels(std::string_view file_content) {
  struct csv_level {
    utl::csv_col<utl::cstr, UTL_NAME("level_id")> id_;
    utl::csv_col<utl::cstr, UTL_NAME("level_index")> index_;
  };
  auto levels = hash_map<std::string_view, float>{};
  utl::line_range{utl::make_buf_reader(file_content)}  //
      | utl::csv<csv_level>()  //
      | utl::for_each([&](csv_level const& l) {
          if (auto const index = parse_optional(*l.index_); index.has_value()) {
            levels.emplace(l.id_->trim().view(), static_cast<float>(*index));
          }
        });
  return levels;
}

std::vector<pathway_row> read_pathway_rows(std::string_view file_content) {
  struct csv_pathway {
    utl::csv_col<utl::cstr, UTL_NAME("from_stop_id")> from_;
    utl::csv_col<utl::cstr, UTL_NAME("to_stop_id")> to_;
    utl::csv_col<unsigned, UTL_NAME("pathway_mode")> mode_;
    utl::csv_col<unsigned, UTL_NAME("is_bidirectional")> is_bidirectional_;
    utl::csv_col<utl::cstr, UTL_NAME("length")> length_;
    utl::csv_col<utl::cstr, UTL_NAME("traversal_time")> traversal_time_;
    utl::csv_col<utl::cstr, UTL_NAME("stair_count")> stair_count_;
    utl::csv_col<utl::cstr, UTL_NAME("max_slope")> max_slope_;
    utl::csv_col<utl::cstr, UTL_NAME("min_width")> min_width_;
    utl::csv_col<utl::cstr, UTL_NAME("signposted_as")> signposted_as_;
    utl::csv_col<utl::cstr, UTL_NAME("reversed_signposted_as")>
        reversed_signposted_as_;
  };
  auto rows = std::vector<pathway_row>{};
  utl::line_range{utl::make_buf_reader(file_content)}  //
      | utl::csv<csv_pathway>()  //
      | utl::for_each([&](csv_pathway const& p) {
          if (*p.mode_ < 1U || *p.mode_ > 7U) {
            log(log_lvl::error, "loader.gtfs.pathways",
                "invalid pathway_mode {} ({} -> {})", *p.mode_, p.from_->view(),
                p.to_->view());
            return;
          }
          rows.push_back(
              pathway_row{.from_ = p.from_->trim().view(),
                          .to_ = p.to_->trim().view(),
                          .mode_ = static_cast<pathway_mode>(*p.mode_),
                          .is_bidirectional_ = *p.is_bidirectional_ == 1U,
                          .length_ = parse_optional(*p.length_),
                          .traversal_time_ = parse_optional(*p.traversal_time_),
                          .stair_count_ = parse_optional(*p.stair_count_),
                          .max_slope_ = parse_optional(*p.max_slope_),
                          .min_width_ = parse_optional(*p.min_width_),
                          .signposted_as_ = p.signposted_as_->trim().view(),
                          .reversed_signposted_as_ =
                              p.reversed_signposted_as_->trim().view()});
        });
  return rows;
}

void read_node_rows(std::string_view file_content,
                    hash_map<std::string_view, std::optional<node_row>>& rows) {
  struct csv_node {
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_name")> name_;
    utl::csv_col<unsigned, UTL_NAME("location_type")> location_type_;
    utl::csv_col<utl::cstr, UTL_NAME("parent_station")> parent_station_;
    utl::csv_col<utl::cstr, UTL_NAME("level_id")> level_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_lat")> lat_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_lon")> lon_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_access")> stop_access_;
  };
  utl::line_range{utl::make_buf_reader(file_content)}  //
      | utl::csv<csv_node>()  //
      | utl::for_each([&](csv_node const& s) {
          auto const it = rows.find(s.id_->trim().view());
          if (it == end(rows) || it->second.has_value()) {
            return;
          }
          auto const lat = parse_optional(*s.lat_);
          auto const lng = parse_optional(*s.lon_);
          it->second = node_row{
              .parent_ = s.parent_station_->trim().view(),
              .level_ = s.level_id_->trim().view(),
              .name_ = s.name_->trim().view(),
              .pos_ = lat.has_value() && lng.has_value()
                          ? std::optional{geo::latlng{*lat, *lng}}
                          : std::nullopt,
              .type_ = static_cast<gtfs_location_type>(
                  std::min(*s.location_type_, 4U)),
              .is_street_accessible_ = s.stop_access_->trim().view() == "1"};
        });
}

station_graph::node_type to_node_type(gtfs_location_type const t) {
  switch (t) {
    case gtfs_location_type::kEntrance:
      return station_graph::node_type::kEntrance;
    case gtfs_location_type::kBoardingArea:
      return station_graph::node_type::kBoardingArea;
    case gtfs_location_type::kStop: return station_graph::node_type::kStop;
    case gtfs_location_type::kStation: [[fallthrough]];
    case gtfs_location_type::kGeneric: break;
  }
  return station_graph::node_type::kGeneric;
}

station_graph::edge_mode to_edge_mode(pathway_mode const m) {
  using em = station_graph::edge_mode;
  switch (m) {
    case pathway_mode::kWalkway: return em::kWalkway;
    case pathway_mode::kStairs: return em::kStairs;
    case pathway_mode::kTravelator: return em::kTravelator;
    case pathway_mode::kEscalator: return em::kEscalator;
    case pathway_mode::kElevator: return em::kElevator;
    case pathway_mode::kFareGate: return em::kFareGate;
    case pathway_mode::kExitGate: return em::kExitGate;
  }
  return em::kWalkway;
}

std::uint32_t to_seconds(double const s) {
  return static_cast<std::uint32_t>(std::max(0.0, std::ceil(s)));
}

// Walking or fixed time and restriction of one pathway direction.
station_graph::edge to_edge(pathway_row const& p,
                            std::optional<double> const distance) {
  auto e = station_graph::edge{};
  e.mode_ = to_edge_mode(p.mode_);

  auto const walk_seconds = [&](double const meters) {
    return to_seconds(meters / kPathwayWalkSpeed);
  };
  auto const length = p.length_.has_value() ? p.length_ : distance;

  switch (p.mode_) {
    case pathway_mode::kWalkway: [[fallthrough]];
    case pathway_mode::kTravelator: [[fallthrough]];
    case pathway_mode::kFareGate: [[fallthrough]];
    case pathway_mode::kExitGate:
      e.walk_seconds_ = p.traversal_time_.has_value()
                            ? to_seconds(*p.traversal_time_)
                        : length.has_value() ? walk_seconds(*length)
                                             : 0U;
      break;

    case pathway_mode::kStairs:
      e.walk_seconds_ =
          p.traversal_time_.has_value() ? to_seconds(*p.traversal_time_)
          : p.stair_count_.has_value()
              ? to_seconds(std::abs(*p.stair_count_) * kSecondsPerStair)
          : length.has_value() ? walk_seconds(*length * kStairsSlowdown)
                               : 0U;
      break;

    case pathway_mode::kEscalator:
      e.fixed_seconds_ =
          p.traversal_time_.has_value() ? to_seconds(*p.traversal_time_)
          : length.has_value()          ? to_seconds(*length / kEscalatorSpeed)
                                        : kDefaultEscalatorSeconds;
      break;

    case pathway_mode::kElevator:
      e.fixed_seconds_ = p.traversal_time_.has_value()
                             ? to_seconds(*p.traversal_time_)
                             : kDefaultElevatorSeconds;
      break;
  }

  auto& r = e.restriction_;
  switch (p.mode_) {
    case pathway_mode::kStairs:
      r.flags_ = p.stair_count_.has_value() && std::abs(*p.stair_count_) == 1.0
                     ? footpath_restriction::kStep
                     : footpath_restriction::kStairs;
      break;
    case pathway_mode::kEscalator:
      r.flags_ = footpath_restriction::kEscalator;
      break;
    case pathway_mode::kElevator:
      r.flags_ = footpath_restriction::kElevator;
      break;
    case pathway_mode::kWalkway:
      // GTFS: empty max_slope = no slope.
      if (p.max_slope_.has_value() && *p.max_slope_ != 0.0) {
        r.flags_ = footpath_restriction::kRamp;
        r.slope_ =
            footpath_restriction::to_slope(std::abs(*p.max_slope_) * 100.0);
      }
      break;
    default: break;
  }
  r.width_ = p.min_width_.has_value()
                 ? footpath_restriction::to_width(*p.min_width_)
                 : static_cast<std::uint8_t>(footpath_restriction::kNoWidth |
                                             footpath_restriction::kUnknown);
  return e;
}

void read_pathways(source_idx_t const src,
                   station_graph& g,
                   stops_map_t const& stops,
                   std::string_view const stops_file_content,
                   std::string_view const levels_file_content,
                   std::string_view const pathways_file_content,
                   script_runner const& script) {
  if (pathways_file_content.empty()) {
    return;
  }

  auto const timer = scoped_timer{"loader.gtfs.pathways"};

  auto const pathways = read_pathway_rows(pathways_file_content);
  if (pathways.empty()) {
    return;
  }

  // stops.txt rows of all pathway endpoints and of the platforms of
  // boarding areas
  auto rows = hash_map<std::string_view, std::optional<node_row>>{};
  for (auto const& p : pathways) {
    rows.emplace(p.from_, std::nullopt);
    rows.emplace(p.to_, std::nullopt);
  }
  read_node_rows(stops_file_content, rows);
  auto n_platforms = 0U;
  for (auto const& [id, row] : std::vector(begin(rows), end(rows))) {
    if (row.has_value() && row->type_ == gtfs_location_type::kBoardingArea &&
        !row->parent_.empty() &&
        rows.emplace(row->parent_, std::nullopt).second) {
      ++n_platforms;
    }
  }
  if (n_platforms != 0U) {
    read_node_rows(stops_file_content, rows);
  }

  auto const levels = read_levels(levels_file_content);
  auto nodes = hash_map<std::string_view, station_graph::node_idx_t>{};
  for (auto const& [id, row] : rows) {
    if (!row.has_value()) {
      log(log_lvl::error, "loader.gtfs.pathways", "stop {:?} not in stops.txt",
          id);
      continue;
    }
    auto n = station_graph::node{};
    if (row->type_ == gtfs_location_type::kStop) {
      if (auto const it = stops.find(std::string{id}); it != end(stops)) {
        n.l_ = it->second;
      }
    }
    n.pos_ = row->pos_;
    if (auto const it = levels.find(row->level_); it != end(levels)) {
      n.level_ = it->second;
    }
    n.type_ = to_node_type(row->type_);
    n.is_street_accessible_ = row->is_street_accessible_;
    if (row->type_ == gtfs_location_type::kEntrance) {
      n.name_ = row->name_;
    }
    nodes.emplace(id, g.add_node(std::move(n)));
  }

  // Boarding areas are part of their platform.
  for (auto const& [id, row] : rows) {
    if (!row.has_value() || row->type_ != gtfs_location_type::kBoardingArea) {
      continue;
    }
    auto const area = nodes.find(id);
    auto const platform = nodes.find(row->parent_);
    if (area == end(nodes) || platform == end(nodes)) {
      continue;
    }
    g.add_edge(
        station_graph::edge{.from_ = platform->second, .to_ = area->second});
    g.add_edge(
        station_graph::edge{.from_ = area->second, .to_ = platform->second});
  }

  // Station of a node: parent station, boarding areas via their platform.
  auto const get_station = [&](std::string_view const id) {
    auto const& row = rows.at(id);
    if (row->type_ == gtfs_location_type::kBoardingArea) {
      auto const platform = rows.find(row->parent_);
      if (platform != end(rows) && platform->second.has_value()) {
        return platform->second->parent_;
      }
    }
    return row->parent_.empty() ? id : row->parent_;
  };
  auto is_station_used = hash_map<std::string_view, bool>{};
  auto const is_used = [&](std::string_view const id) {
    auto const station = get_station(id);
    return utl::get_or_create(is_station_used, station, [&]() {
      auto x = station_pathways{.src_ = src, .id_ = station};
      return process_pathways(script, x);
    });
  };

  auto n_edges = 0U;
  for (auto const& p : pathways) {
    auto const from = nodes.find(p.from_);
    auto const to = nodes.find(p.to_);
    if (from == end(nodes) || to == end(nodes) || !is_used(p.from_) ||
        !is_used(p.to_)) {
      continue;
    }
    auto const& a = g.nodes_[from->second];
    auto const& b = g.nodes_[to->second];
    auto const distance = a.pos_.has_value() && b.pos_.has_value()
                              ? std::optional{geo::distance(*a.pos_, *b.pos_)}
                              : std::nullopt;

    auto e = to_edge(p, distance);
    e.from_ = from->second;
    e.to_ = to->second;
    if (p.is_bidirectional_) {
      auto reverse = e;
      std::swap(reverse.from_, reverse.to_);
      reverse.signpost_ = p.reversed_signposted_as_;
      g.add_edge(std::move(reverse));
      ++n_edges;
    }
    e.signpost_ = p.signposted_as_;
    g.add_edge(std::move(e));
    ++n_edges;
  }

  log(log_lvl::info, "loader.gtfs.pathways",
      "{} pathways -> {} nodes, {} edges", pathways.size(), nodes.size(),
      n_edges);
}

}  // namespace nigiri::loader::gtfs
