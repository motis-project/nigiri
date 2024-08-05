#pragma once

#include "geo/latlng.h"

#include "oh/parser.h"

#include "nigiri/timetable.h"

namespace nigiri::loader {

struct station {
  std::string name_;
  geo::latlng pos_;
  oh::rule availability_;
};

std::vector<station> read_availability(std::string_view);

}  // namespace nigiri::loader