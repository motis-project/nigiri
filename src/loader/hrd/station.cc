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
        stations[eva_num].name_ = name.to_str();
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

      from.footpaths_out_.emplace_back(to.id_, duration);
      to.footpaths_in_.emplace_back(from.id_, duration);

      // make footpaths symmetric
      to.footpaths_out_.emplace_back(from.id_, duration);
      from.footpaths_in_.emplace_back(to.id_, duration);

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
  hash_map<eva_number, hrd_location> stations;
  parse_station_names(c, stations, station_names_file);
  parse_station_coordinates(c, stations, station_coordinates_file);
  parse_equivilant_stations(c, stations, station_metabhf_file);
  parse_footpaths(c, stations, station_metabhf_file);

  for (auto const& [eva, s] : stations) {
    tt.locations_.location_id_to_idx_.emplace(
        location_id{.id_ = std::to_string(to_idx(s.id_)), .src_ = src},
        location_idx_t{tt.locations_.types_.size()});
    tt.locations_.types_.emplace_back(location_type::station);
  }

  return stations;
}

}  // namespace nigiri::loader::hrd
