#include "nigiri/loader/hrd/station.h"

#include "utl/parser/arg_parser.h"
#include "utl/pipes.h"

#include "nigiri/logging.h"

namespace nigiri::loader::hrd {

void parse_station_names(config const& c,
                         hash_map<eva_number, hrd_location>& stations,
                         std::string_view file_content) {
  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, int line_number) {
        if (line.len == 0 || line[0] == '%') {
          return;
        } else if (line.len < 13) {
          log(log_lvl::error, "nigiri.loader.hrd.station.coordinates",
              "station name file unknown line format line={} content=\"{}\"",
              line_number, line.view());
          return;
        }

        auto name = line.substr(c.st_.names_.name_);
        auto const it = std::find(begin(name), end(name), '$');
        if (it != end(name)) {
          name.len = std::distance(begin(name), it);
        }

        auto const eva_num = parse_eva_number(line.substr(c.st_.names_.eva_));
        auto& s = stations[eva_num];
        s.name_ = name.to_str();
        s.id_ = eva_num;
      });
}

void parse_station_coordinates(config const& c,
                               hash_map<eva_number, hrd_location>& stations,
                               std::string_view file_content) {
  utl::for_each_line_numbered(file_content, [&](utl::cstr line,
                                                int line_number) {
    if (line.len == 0 || line[0] == '%') {
      return;
    } else if (line.len < 30) {
      log(log_lvl::error, "nigiri.loader.hrd.station.coordinates",
          "station coordinate file unknown line format line={} content=\"{}\"",
          line_number, line.view());
      return;
    }

    stations[parse_eva_number(line.substr(c.st_.coords_.eva_))].pos_ = {
        utl::parse_verify<double>(line.substr(c.st_.coords_.lng_).trim()),
        utl::parse_verify<double>(line.substr(c.st_.coords_.lat_).trim())};
  });
}

void parse_equivilant_stations(config const& c,
                               hash_map<eva_number, hrd_location>& stations,
                               std::string_view file_content) {
  utl::for_each_line(file_content, [&](utl::cstr line) {
    if (line.length() < 16 || line[0] == '%' || line[0] == '*') {
      return;
    }

    if (line[7] == ':') {  // equivalent stations
      auto& station = stations.at(
          parse_eva_number(line.substr(c.meta_.meta_stations_.eva_)));
      utl::for_each_token(line.substr(8), ' ', [&](utl::cstr token) {
        if (token.empty() ||
            c.version_ == "hrd_5_20_26" && token.starts_with("F")) {
          return;
        }
        if (auto const eva = parse_eva_number(token); eva != 0) {
          station.equivalent_.emplace(eva);
        }
      });
    } else {  // footpaths
    }
  });
}

void parse_footpaths(config const& c,
                     hash_map<eva_number, hrd_location>& stations,
                     std::string_view file_content) {
  auto const add_footpath = [](hrd_location& l, eva_number const to,
                               duration_t const d) {
    if (auto const it = l.footpaths_out_.find(to);
        it != end(l.footpaths_out_)) {
      it->second = std::min(it->second, d);
    } else {
      l.footpaths_out_.emplace(to, d);
    }
  };

  utl::for_each_line(file_content, [&](utl::cstr line) {
    if (line.length() < 16 || line[0] == '%' || line[0] == '*') {
      return;
    }

    if (line[7] == ':') {  // equivalent stations
    } else {  // footpaths
      auto const f_equal = c.version_ == "hrd_5_00_8" && line.length() > 23 &&
                           line.substr(23, utl::size(1)) == "F";
      auto& from =
          stations.at(parse_eva_number(line.substr(c.meta_.footpaths_.from_)));
      auto& to =
          stations.at(parse_eva_number(line.substr(c.meta_.footpaths_.to_)));
      auto const duration =
          duration_t{parse<int>(line.substr(c.meta_.footpaths_.duration_))};

      add_footpath(from, to.id_, duration);

      if (f_equal) {
        from.equivalent_.erase(to.id_);
      }
    }
  });
}

hash_map<eva_number, hrd_location> parse_stations(
    config const& c,
    source_idx_t const src,
    timetable& tt,
    std::string_view station_names_file,
    std::string_view station_coordinates_file,
    std::string_view station_metabhf_file) {
  auto empty_idx_vec = vector<location_idx_t>{};
  auto empty_footpath_vec = vector<footpath>{};

  hash_map<eva_number, hrd_location> stations;
  parse_station_names(c, stations, station_names_file);
  parse_station_coordinates(c, stations, station_coordinates_file);
  parse_equivilant_stations(c, stations, station_metabhf_file);
  parse_footpaths(c, stations, station_metabhf_file);

  for (auto& [eva, s] : stations) {
    auto const id =
        location_id{.id_ = fmt::format("{:07}", to_idx(eva)), .src_ = src};
    auto const idx = tt.locations_.add(
        timetable::location{.id_ = id.id_,
                            .name_ = s.name_,
                            .pos_ = s.pos_,
                            .src_ = src,
                            .type_ = location_type::station,
                            .osm_id_ = osm_node_id_t::invalid(),
                            .parent_ = location_idx_t::invalid(),
                            .equivalences_ = it_range{empty_idx_vec},
                            .footpaths_out_ = it_range{empty_footpath_vec},
                            .footpaths_in_ = it_range{empty_footpath_vec}});
    s.idx_ = idx;
  }

  for (auto& [eva, s] : stations) {
    for (auto const& e : s.equivalent_) {
      tt.locations_.equivalences_[s.idx_].emplace_back(stations.at(e).idx_);
    }

    for (auto const& [target_eva, duration] : s.footpaths_out_) {
      auto const target_idx = stations.at(target_eva).idx_;
      tt.locations_.footpaths_out_[s.idx_].emplace_back(target_idx, duration);
      tt.locations_.footpaths_in_[target_idx].emplace_back(s.idx_, duration);
    }
  }

  return stations;
}

}  // namespace nigiri::loader::hrd
