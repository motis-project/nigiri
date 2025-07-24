#pragma once

#include <string>

#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri::loader::netex {

struct quay {
  std::string id_;
  std::string name_;
  geo::latlng centroid_{};
};

struct stop_place {
  std::string id_;
  std::string name_;
  std::string description_;
  geo::latlng centroid_{};
  std::vector<quay> quays_;

  location_idx_t location_idx_{location_idx_t::invalid()};
};

struct netex_data {
  hash_map<std::string, stop_place> stop_places_;
};

}  // namespace nigiri::loader::netex
