#include "nigiri/loader/gtfs/location_geojson.h"

#include <utl/raii.h>

#include <tg.h>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/json/src.hpp>

#include <nigiri/timetable.h>

namespace nigiri::loader::gtfs {
location_geojson_map_t read_location_geojson(source_idx_t src,
                                             timetable& tt,
                                             std::string_view file_content) {
  location_geojson_map_t location_geojson{};

  boost::system::error_code ec;
  auto json = boost::json::parse(file_content, ec);
  if (ec) {
    log(log_lvl::error, "loader.gtfs.location_geojson",
        R"(Could not parse location.geojson!)");
    return location_geojson;
  }

  auto featureCollection = json.as_object();

  auto const features_entry = featureCollection.find("features");
  if (features_entry == featureCollection.end()) {
    log(log_lvl::error, "loader.gtfs.location_geojson",
        R"(Could not find entry with key "{}"!)", "features");
    return location_geojson;
  }
  auto const features = features_entry->value().as_array();
  for (auto feature = features.begin(); feature != features.end(); ++feature) {
    auto const feature_object = feature->as_object();
    auto const id_entry = feature_object.find("id");
    if (id_entry == feature_object.end()) {
      log(log_lvl::error, "loader.gtfs.location_geojson",
          R"(feature index {}: Could not find entry with key "{}"!)",
          std::distance(features.begin(), feature) + 1, "id");
      continue;
    }
    auto const boost_string_id = id_entry->value().as_string();
    auto const id = std::string(boost_string_id.data(), boost_string_id.size());
    auto const geometry_entry = feature_object.find("geometry");
    if (geometry_entry == feature_object.end()) {
      log(log_lvl::error, "loader.gtfs.location_geojson",
          R"(feature {}: Could not find entry with key "{}"!)", id, "geometry");
      continue;
    }
    auto const geometry_object = geometry_entry->value().as_object();

    auto const feature_content = boost::json::serialize(geometry_object);

    auto const geometry = tg_parse_geojson(feature_content.c_str());
    if (tg_geom_error(geometry)) {
      log(log_lvl::error, "loader.gtfs.location_geojson",
          R"(feature {}: Could not parse feature!)", id);
      continue;
    }

    const auto idx = tt.register_location_geojson(src, id, geometry);
    location_geojson.emplace(id, idx);
  }
  return location_geojson;
}

}  // namespace nigiri::loader::gtfs