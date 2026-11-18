#include "nigiri/loader/isa/loader.h"

#include <cmath>
#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "nigiri/loader/import_context.h"
#include "nigiri/loader/station_graph.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::isa {

constexpr auto const kStopsFile = std::string_view{"halteste.asc"};
constexpr auto const kAreasFile = std::string_view{"umbwbereich.asc"};
constexpr auto const kAreaTypesFile = std::string_view{"umbwbereichtyp.asc"};
constexpr auto const kPathsFile = std::string_view{"umbwweg.asc"};
constexpr auto const kPathTypesFile = std::string_view{"umbwwegtyp.asc"};
constexpr auto const kPathAttributesFile = std::string_view{"umbwwegattr.asc"};
constexpr auto const kAttributesFile = std::string_view{"attribut.asc"};

// Speed profile header of umbwweg.asc: speed * 100 in km/h.
constexpr auto const kPreferredSpeed = 500U;  // "mittel"
constexpr auto const kDefaultWalkSpeed = 1.2;  // m/s, as osr foot

using row_t = std::vector<std::string_view>;

std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\r')) {
    s.remove_prefix(1U);
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\r')) {
    s.remove_suffix(1U);
  }
  return s;
}

// '#'-separated rows, '%' = comment.
template <typename Fn>
void for_each_row(dir const& d, std::string_view const name, Fn&& fn) {
  if (!d.exists(name)) {
    return;
  }
  auto const f = d.get_file(name);
  auto content = f.data();
  auto row = row_t{};
  while (!content.empty()) {
    auto const eol = content.find('\n');
    auto line = content.substr(0U, eol);
    content = eol == std::string_view::npos ? std::string_view{}
                                            : content.substr(eol + 1U);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1U);
    }
    if (line.empty() || line.front() == '%') {
      continue;
    }
    row.clear();
    for (auto pos = std::size_t{0U};;) {
      auto const sep = line.find('#', pos);
      row.push_back(trim(line.substr(pos, sep - pos)));
      if (sep == std::string_view::npos) {
        break;
      }
      pos = sep + 1U;
    }
    fn(row);
  }
}

std::string_view col(row_t const& r, std::size_t const i) {
  return i < r.size() ? r[i] : std::string_view{};
}

std::optional<double> to_number(std::string_view const s) {
  if (s.empty()) {
    return std::nullopt;
  }
  auto const str = std::string{s};
  char* end = nullptr;
  auto const v = std::strtod(str.c_str(), &end);
  return end == str.c_str() || !std::isfinite(v) ? std::nullopt
                                                 : std::optional{v};
}

std::string key(row_t const& r, std::size_t const from, std::size_t const n) {
  auto k = std::string{};
  for (auto i = from; i != from + n; ++i) {
    k.append(col(r, i));
    k.push_back('#');
  }
  return k;
}

// Measured values of a path (DELFI attribute codes with a value).
struct path_values {
  std::optional<double> width_cm_;  // 2021 path, 2034 door, 2091 elevator door
  std::optional<double> slope_percent_;  // 2124
  std::optional<double> step_cm_;  // 2101
  std::optional<double> cabin_length_cm_;  // 2093
  std::optional<double> cabin_width_cm_;  // 2094
};

// Highest step a walker / an assisted wheelchair passes (DELFI handbook,
// plan decision); higher steps count as stairs.
constexpr auto const kMaxStepCm = 15.0;

// Smallest elevator cabin for wheelchairs (DIN 18040-1: 1.10 m x 1.40 m).
constexpr auto const kMinCabinWidthCm = 110.0;
constexpr auto const kMinCabinLengthCm = 140.0;

std::optional<double> to_value(std::string_view const s) {
  auto x = std::string{trim(s)};
  if (x.empty()) {
    return std::nullopt;
  }
  std::replace(begin(x), end(x), ',', '.');
  try {
    return std::stod(x);
  } catch (...) {
    return std::nullopt;
  }
}

void add_value(path_values& v, std::string_view const code, double const x) {
  auto const set_min = [](std::optional<double>& o, double const y) {
    o = o.has_value() ? std::min(*o, y) : y;
  };
  if (code == "2021" || code == "2034" || code == "2091") {
    set_min(v.width_cm_, x);
  } else if (code == "2124") {
    v.slope_percent_ = std::max(v.slope_percent_.value_or(0.0), x);
  } else if (code == "2101") {
    v.step_cm_ = std::max(v.step_cm_.value_or(0.0), x);
  } else if (code == "2093") {
    v.cabin_length_cm_ = x;
  } else if (code == "2094") {
    v.cabin_width_cm_ = x;
  }
}

footpath_restriction to_restriction(std::uint8_t flags, path_values const* v) {
  using fr = footpath_restriction;
  if (v != nullptr) {
    if ((flags & fr::kStep) != 0U && v->step_cm_.has_value() &&
        *v->step_cm_ > kMaxStepCm) {
      flags |= fr::kStairs;
    }
    if ((flags & fr::kElevator) != 0U && v->cabin_length_cm_.has_value() &&
        v->cabin_width_cm_.has_value() &&
        (std::min(*v->cabin_length_cm_, *v->cabin_width_cm_) <
             kMinCabinWidthCm ||
         std::max(*v->cabin_length_cm_, *v->cabin_width_cm_) <
             kMinCabinLengthCm)) {
      flags |= fr::kBanWheelchair | fr::kBanWheelchairAssisted;
    }
  }
  auto const has_slope = v != nullptr && v->slope_percent_.has_value();
  auto const has_width = v != nullptr && v->width_cm_.has_value();
  // Paths other than ramps are level (as GTFS without max_slope).
  return fr{.flags_ = flags,
            .slope_ = has_slope ? fr::to_slope(*v->slope_percent_)
                      : (flags & fr::kRamp) != 0U ? fr::kUnknown
                                                  : std::uint8_t{0U},
            .width_ = has_width ? fr::to_width(*v->width_cm_ / 100.0)
                                : static_cast<std::uint8_t>(fr::kNoWidth |
                                                            fr::kUnknown),
            .fixed_ = 0U};
}

std::uint8_t to_flags(std::string_view const code,
                      std::string_view const text) {
  using fr = footpath_restriction;
  // DELFI attribute codes (Attributkuerzel).
  if (code == "2090") {
    return fr::kElevator;
  } else if (code == "2100") {
    return fr::kStep;
  } else if (code == "2110") {
    return fr::kStairs;
  } else if (code == "2120") {
    return fr::kRamp;
  } else if (code == "2130") {
    return fr::kEscalator;
  }
  // User group bans: matched by text (codes differ between deliveries).
  if (text.starts_with("Gesperrt f") &&
      text.find("Rollstuhl") != std::string_view::npos) {
    return text.find("ohne Begleit") != std::string_view::npos
               ? fr::kBanWheelchair
               : fr::kBanWheelchairAssisted;
  } else if (text.starts_with("Gesperrt f") &&
             text.find("Rollator") != std::string_view::npos) {
    return fr::kBanWalker;
  }
  return 0U;
}

void load_station_graph(dir const& d, station_graph& g) {
  auto const timer = scoped_timer{"loader.isa.station_graph"};

  // ISA stop (number, supplier) -> global id (DHID)
  auto global_ids = hash_map<std::string, std::string>{};
  for_each_row(d, kStopsFile, [&](row_t const& r) {
    if (!col(r, 19).empty()) {
      global_ids.emplace(key(r, 0U, 2U), col(r, 19));
    }
  });

  // area type -> (for public transport, name)
  auto area_types = hash_map<std::string, std::pair<bool, std::string>>{};
  for_each_row(d, kAreaTypesFile, [&](row_t const& r) {
    area_types.emplace(std::string{col(r, 0)},
                       std::pair{col(r, 1) == "1", std::string{col(r, 3)}});
  });

  // areas -> nodes; every further linked stop -> extra node, no cost
  auto nodes = hash_map<std::string, station_graph::node_idx_t>{};
  auto extra_refs =
      std::vector<std::pair<station_graph::node_idx_t, std::string>>{};
  auto current = std::optional<station_graph::node_idx_t>{};
  auto is_current_pt = false;
  for_each_row(d, kAreasFile, [&](row_t const& r) {
    if (!col(r, 0).empty()) {
      auto const type = area_types.find(std::string{col(r, 6)});
      is_current_pt = type != end(area_types) && type->second.first;
      auto n = station_graph::node{};
      auto const x = to_number(col(r, 12));
      auto const y = to_number(col(r, 13));
      if (x.has_value() && y.has_value()) {
        n.pos_ = geo::latlng{*y, *x};
      }
      if (auto const level = to_number(col(r, 9)); level.has_value()) {
        n.level_ = static_cast<float>(*level);
      }
      n.type_ = is_current_pt ? station_graph::node_type::kStop
                : type != end(area_types) && type->second.second == "Zugang"
                    ? station_graph::node_type::kEntrance
                    : station_graph::node_type::kGeneric;
      n.name_ = col(r, 15);
      if (is_current_pt) {
        n.ref_ = col(r, 25);
      }
      current = g.add_node(std::move(n));
      nodes.emplace(key(r, 0U, 4U), *current);
    } else if (current.has_value() && is_current_pt) {
      // linked stop: (supplier ref, stop number, track text)
      auto const id = global_ids.find(std::string{col(r, 2)} + "#" +
                                      std::string{col(r, 1)} + "#");
      if (id == end(global_ids)) {
        return;
      }
      auto& n = g.nodes_[*current];
      if (n.ref_.empty() || n.ref_ == id->second) {
        n.ref_ = id->second;
      } else {
        extra_refs.emplace_back(*current, id->second);
      }
    }
  });
  for (auto const& [area, ref] : extra_refs) {
    auto n = g.nodes_[area];
    n.ref_ = ref;
    n.name_.clear();
    auto const extra = g.add_node(std::move(n));
    g.add_edge(station_graph::edge{.from_ = area, .to_ = extra});
    g.add_edge(station_graph::edge{.from_ = extra, .to_ = area});
  }

  auto path_types = hash_map<std::string, std::string>{};
  for_each_row(d, kPathTypesFile, [&](row_t const& r) {
    path_types.emplace(std::string{col(r, 0)}, std::string{col(r, 3)});
  });

  auto attribute_flags = hash_map<std::string, std::uint8_t>{};
  auto attribute_codes = hash_map<std::string, std::string>{};
  for_each_row(d, kAttributesFile, [&](row_t const& r) {
    if (auto const flags = to_flags(col(r, 1), col(r, 4)); flags != 0U) {
      attribute_flags.emplace(std::string{col(r, 0)}, flags);
    }
    attribute_codes.emplace(std::string{col(r, 0)}, std::string{col(r, 1)});
  });
  auto values = hash_map<std::string, path_values>{};
  auto path_flags = hash_map<std::string, std::uint8_t>{};
  for_each_row(d, kPathAttributesFile, [&](row_t const& r) {
    if (auto const it = attribute_flags.find(std::string{col(r, 1)});
        it != end(attribute_flags)) {
      path_flags[std::string{col(r, 0)}] |= it->second;
    }
    auto const code = attribute_codes.find(std::string{col(r, 1)});
    auto const value = to_value(col(r, 2));
    if (code != end(attribute_codes) && value.has_value()) {
      add_value(values[std::string{col(r, 0)}], code->second, *value);
    }
  });

  // Paths: rows without speed profile + the profile closest to "mittel".
  auto speeds = std::vector<unsigned>{};
  for_each_row(d, kPathsFile, [&](row_t const& r) {
    if (auto const s = to_number(col(r, 0)); s.has_value()) {
      speeds.push_back(static_cast<unsigned>(*s));
    }
  });
  auto const preferred =
      speeds.empty()
          ? 0U
          : *std::min_element(begin(speeds), end(speeds),
                              [](unsigned const a, unsigned const b) {
                                auto const da = a > kPreferredSpeed
                                                    ? a - kPreferredSpeed
                                                    : kPreferredSpeed - a;
                                auto const db = b > kPreferredSpeed
                                                    ? b - kPreferredSpeed
                                                    : kPreferredSpeed - b;
                                return da < db;
                              });

  using fr = footpath_restriction;
  auto use_rows = true;
  auto speed = kDefaultWalkSpeed;
  auto n_edges = 0U;
  for_each_row(d, kPathsFile, [&](row_t const& r) {
    if (!col(r, 0).empty() || r.size() < 13U) {  // speed profile header
      auto const s = to_number(col(r, 0));
      use_rows = !s.has_value() || static_cast<unsigned>(*s) == preferred;
      speed = s.has_value() ? *s / 100.0 / 3.6 : kDefaultWalkSpeed;
      return;
    }
    if (!use_rows) {
      return;
    }
    auto const from = nodes.find(key(r, 2U, 4U));
    auto const to = nodes.find(key(r, 6U, 4U));
    if (from == end(nodes) || to == end(nodes) || from->second == to->second) {
      return;  // self loop: same-area transfer time
    }

    auto const type_it = path_types.find(std::string{col(r, 10)});
    auto const type =
        type_it == end(path_types) ? std::string_view{} : type_it->second;
    auto const time = to_number(col(r, 11));
    auto const length = to_number(col(r, 12));
    auto const seconds = static_cast<std::uint32_t>(
        std::ceil(time.has_value() ? *time : length.value_or(0.0) / speed));

    auto e = station_graph::edge{.from_ = from->second, .to_ = to->second};
    auto flags = std::uint8_t{0U};
    if (type == "Treppe") {
      e.mode_ = station_graph::edge_mode::kStairs;
      flags = fr::kStairs;
    } else if (type == "Rolltreppe") {
      e.mode_ = station_graph::edge_mode::kEscalator;
      flags = fr::kEscalator;
    } else if (type == "Aufzug") {
      e.mode_ = station_graph::edge_mode::kElevator;
      flags = fr::kElevator;
    } else if (type == "Rampe") {
      flags = fr::kRamp;
    }
    if (auto const it = path_flags.find(std::string{col(r, 1)});
        it != end(path_flags)) {
      flags |= it->second;
    }
    auto const is_ride = e.mode_ == station_graph::edge_mode::kEscalator ||
                         e.mode_ == station_graph::edge_mode::kElevator;
    (is_ride ? e.fixed_seconds_ : e.walk_seconds_) = seconds;
    auto const v = values.find(std::string{col(r, 1)});
    e.restriction_ =
        to_restriction(flags, v == end(values) ? nullptr : &v->second);
    e.signpost_ = col(r, 14);
    g.add_edge(std::move(e));
    ++n_edges;
  });

  log(log_lvl::info, "loader.isa.station_graph",
      "{} areas, {} paths (speed profile {}), {} stops linked by id",
      nodes.size(), n_edges, preferred, global_ids.size());
}

bool isa_station_graph_loader::applicable(dir const& d) const {
  return d.exists(kPathsFile) && d.exists(kAreasFile);
}

void isa_station_graph_loader::load(loader_config const&,
                                    source_idx_t const src,
                                    dir const& d,
                                    timetable& tt,
                                    import_context& ctx,
                                    assistance_times*,
                                    shapes_storage*) const {
  // Per-source data of a source without timetable.
  tt.n_sources_ = std::max(tt.n_sources_, to_idx(src + 1U));
  tt.fares_.emplace_back();
  tt.src_end_date_.push_back(date::sys_days::max());
  tt.route_ids_.emplace_back();
  load_station_graph(d, ctx.station_graph_);
}

std::string_view isa_station_graph_loader::name() const {
  return "isa-station-graph";
}

}  // namespace nigiri::loader::isa
