#include "nigiri/loader/hrd/stamm/station.h"

#include "utl/parser/arg_parser.h"
#include "utl/pipes.h"

#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/logging.h"

namespace nigiri::loader::hrd {

void parse_station_names(config const& c,
                         hash_map<eva_number, hrd_location>& stations,
                         std::string_view file_content) {
  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, unsigned const line_number) {
        if (line.len == 0 || line[0] == '%') {
          return;
        } else if (line.len < 13) {
          log(log_lvl::error, "loader.hrd.station.coordinates",
              "station name file unknown line format line={} content=\"{}\"",
              line_number, line.view());
          return;
        }

        auto name = line.substr(c.st_.names_.name_);
        auto const it = std::find(begin(name), end(name), '$');
        if (it != end(name)) {
          name.len = static_cast<size_t>(std::distance(begin(name), it));
        }

        auto const eva_num = parse_eva_number(line.substr(c.st_.names_.eva_));
        auto& s = stations[eva_num];
        s.name_ = iso_8859_1_to_utf8(name.view());
        s.id_ = eva_num;
      });
}

void parse_station_coordinates(config const& c,
                               hash_map<eva_number, hrd_location>& stations,
                               std::string_view file_content) {
  utl::for_each_line_numbered(file_content, [&](utl::cstr line,
                                                unsigned const line_number) {
    if (line.len == 0 || line[0] == '%') {
      return;
    } else if (line.len < 30) {
      log(log_lvl::error, "loader.hrd.station.coordinates",
          "station coordinate file unknown line format line={} content=\"{}\"",
          line_number, line.view());
      return;
    }

    stations[parse_eva_number(line.substr(c.st_.coords_.eva_))].pos_ = {
        utl::parse_verify<double>(line.substr(c.st_.coords_.lat_).trim()),
        utl::parse_verify<double>(line.substr(c.st_.coords_.lng_).trim())};
  });
}

void parse_equivilant_stations(config const& c,
                               hash_map<eva_number, hrd_location>& stations,
                               std::string_view file_content) {
  auto const is_5_20_26 = c.version_ == "hrd_5_20_26";

  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, unsigned const line_number) {
        if (line.length() < 16 || line[0] == '%' || line[0] == '*') {
          return;
        }

        if (line[7] == ':') {  // equivalent stations
          try {
            auto const eva =
                parse_eva_number(line.substr(c.meta_.meta_stations_.eva_));
            auto const station_it = stations.find(eva);
            if (station_it == end(stations)) {
              log(log_lvl::error, "loader.hrd.meta", "line {}: {} not found",
                  line_number, eva);
              return;
            }
            auto& station = station_it->second;
            utl::for_each_token(line.substr(8), ' ', [&](utl::cstr token) {
              if (token.empty() || (is_5_20_26 && token.starts_with("F"))) {
                return;
              }
              if (token.starts_with("H")  // Hauptmast
                  || token.starts_with("B")  // Bahnhofstafel
                  || token.starts_with("V")  // Virtueller Umstieg
                  || token.starts_with("S")  // Start-Ziel-Aequivalenz
                  || token.starts_with("F")  // Fußweg-Aequivalenz
              ) {
                return;
              }
              if (auto const meta = parse_eva_number(token); meta != 0) {
                station.equivalent_.emplace(meta);
              }
            });
          } catch (std::exception const& e) {
            log(log_lvl::error, "loader.hrd.equivalent",
                "could not parse line {}: {}", line_number, e.what());
          }
        } else {  // footpaths
        }
      });
}

void parse_footpaths(config const& c,
                     hash_map<eva_number, hrd_location>& stations,
                     std::string_view file_content) {
  auto const is_5_20_26 = c.version_ == "hrd_5_20_26";

  auto const add_footpath = [](hrd_location& l, eva_number const to,
                               u8_minutes const d) {
    if (auto const it = l.footpaths_out_.find(to);
        it != end(l.footpaths_out_)) {
      it->second = std::min(it->second, d);
    } else {
      l.footpaths_out_.emplace(to, d);
    }
  };

  utl::for_each_line_numbered(file_content, [&](utl::cstr line,
                                                unsigned const line_number) {
    if (line.length() < 16 || line[0] == '%' || line[0] == '*') {
      return;
    }

    if (line[7] == ':') {  // equivalent stations
    } else {  // footpaths
      auto const f_equal = is_5_20_26 && line.length() > 23 &&
                           line.substr(23, utl::size(1)) == "F";

      auto const from_eva =
          parse_eva_number(line.substr(c.meta_.footpaths_.from_));
      auto const from_it = stations.find(from_eva);
      if (from_it == end(stations)) {
        log(log_lvl::error, "loader.hrd.footpath",
            "footpath line={}: {} not found", line_number, to_idx(from_eva));
        return;
      }
      auto& from = from_it->second;

      auto const to_eva = parse_eva_number(line.substr(c.meta_.footpaths_.to_));
      auto const to_it = stations.find(to_eva);
      if (to_it == end(stations)) {
        log(log_lvl::error, "loader.hrd.footpath",
            "footpath line={}: {} not found", line_number, to_idx(to_eva));
        return;
      }
      auto& to = to_it->second;

      auto const duration_int =
          parse<int>(line.substr(c.meta_.footpaths_.duration_));
      utl::verify(duration_int <= std::numeric_limits<u8_minutes::rep>::max(),
                  "footpath duration {} > {}",
                  std::numeric_limits<u8_minutes::rep>::max());
      add_footpath(from, to.id_, u8_minutes{duration_int});

      if (f_equal) {
        from.equivalent_.erase(to.id_);
      }
    }
  });
}

struct hash {
  size_t operator()(location_id const& id) const {
    return cista::build_hash(id);
  }
};

location_map_t parse_stations(config const& c,
                              source_idx_t const src,
                              timetable& tt,
                              stamm& st,
                              std::string_view station_names_file,
                              std::string_view station_coordinates_file,
                              std::string_view station_metabhf_file) {
  auto const timer = scoped_timer{"parse stations"};

  auto empty_idx_vec = vector<location_idx_t>{};
  auto empty_footpath_vec = vector<footpath>{};

  location_map_t stations;
  parse_station_names(c, stations, station_names_file);
  parse_station_coordinates(c, stations, station_coordinates_file);
  parse_equivilant_stations(c, stations, station_metabhf_file);
  parse_footpaths(c, stations, station_metabhf_file);

  for (auto& [eva, s] : stations) {
    auto const eva_int = to_idx(eva);
    auto const id =
        location_id{.id_ = fmt::format("{:07}", eva_int), .src_ = src};
    auto const transfer_time = duration_t{eva_int < 1000000 ? 2 : 5};
    auto const idx = tt.locations_.register_location(
        location{id.id_, s.name_, s.pos_, src, location_type::kStation,
                 osm_node_id_t::invalid(), location_idx_t::invalid(),
                 st.get_tz(s.id_).first, transfer_time, it_range{empty_idx_vec},
                 std::span{empty_footpath_vec}, std::span{empty_footpath_vec}});
    s.idx_ = idx;
  }

  for (auto& [eva, s] : stations) {
    for (auto const& e : s.equivalent_) {
      if (auto const it = stations.find(e); it != end(stations)) {
        tt.locations_.equivalences_[s.idx_].emplace_back(it->second.idx_);
      } else {
        log(log_lvl::error, "loader.hrd.meta", "station {} not found", e);
      }
    }

    for (auto const& [target_eva, duration] : s.footpaths_out_) {
      auto const target_idx = stations.at(target_eva).idx_;
      auto const adjusted_duration =
          std::max({tt.locations_.transfer_time_[s.idx_],
                    tt.locations_.transfer_time_[target_idx], duration});
      tt.locations_.preprocessing_footpaths_out_[s.idx_].emplace_back(
          target_idx, adjusted_duration);
      tt.locations_.preprocessing_footpaths_in_[target_idx].emplace_back(
          s.idx_, adjusted_duration);
    }
  }

  return stations;
}

}  // namespace nigiri::loader::hrd
