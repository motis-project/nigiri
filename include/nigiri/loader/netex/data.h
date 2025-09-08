#pragma once

#include <optional>
#include <string>
#include <vector>

#include "geo/latlng.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/loader/register.h"

#include "nigiri/loader/netex/proj_transformers.h"

namespace nigiri::loader::netex {

struct netex_locale {
  std::string language_{};
  std::string tz_name_{};
  std::string tz_offset_{};
  std::string tz_summer_offset_{};
  timezone_idx_t tz_idx_{timezone_idx_t::invalid()};
};

struct netex_ctx {
  std::optional<netex_locale> locale_{};
  std::optional<std::string> default_crs_{};
};

struct alt_name {
  std::string name_;
  std::string language_;
};

struct quay {
  std::string id_;
  std::string name_;
  std::string public_code_;
  geo::latlng centroid_{};

  std::optional<std::string> parent_ref_{};
  netex_locale locale_{};

  location_idx_t location_idx_{location_idx_t::invalid()};
};

struct stop_place {
  std::string id_;
  std::string name_;
  std::string description_;
  geo::latlng centroid_{};
  std::vector<quay> quays_{};
  std::vector<alt_name> alt_names_{};

  std::vector<std::string> children_{};
  std::optional<std::string> parent_ref_{};

  netex_locale locale_{};

  location_idx_t location_idx_{location_idx_t::invalid()};
};

struct netex_data {
  hash_map<std::string, stop_place> stop_places_{};

  hash_map<std::string, quay> quays_with_missing_parents_{};
  hash_map<std::string, quay> standalone_quays_{};

  proj_transformers proj_transformers_{};
  gtfs::tz_map timezones_{};

  timetable& tt_;
  script_runner script_runner_;
};

}  // namespace nigiri::loader::netex
