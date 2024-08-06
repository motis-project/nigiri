#pragma once

#include <cstddef>
#include <filesystem>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "utl/parser/cstr.h"

#include "cista/containers/mmap_vec.h"
#include "cista/containers/vecvec.h"
#include "cista/mmap.h"

#include "geo/latlng.h"

#include "nigiri/types.h"

// #include "osmium/osm/location.hpp"
// #include "osr/types.h"

namespace nigiri::loader::gtfs {
namespace helper {

/* Code duplicated from 'osmium/osm/location.hpp' */
constexpr int32_t coordinate_precision{10000000};

constexpr int32_t double_to_fix(double const c) noexcept {
  return static_cast<int32_t>(std::round(c * coordinate_precision));
}

constexpr double fix_to_double(int32_t const c) noexcept {
  return static_cast<double>(c) / coordinate_precision;
}
}  // namespace helper

class shape {
public:
  using key_type = uint32_t;
  using value_type = std::vector<geo::latlng>;
  using shape_coordinate_type =
      std::remove_const<decltype(helper::coordinate_precision)>::type;
  struct coordinate {
    shape_coordinate_type lat, lon;
    bool operator==(coordinate const& other) const = default;
  };
  using mmap_vecvec = mm_vecvec<key_type, coordinate>;
  using id_type = utl::cstr;
  using stored_type = coordinate;
  using builder_t = std::function<std::optional<shape>(const id_type&)>;

  value_type get() const ;
  static builder_t get_builder();
  static builder_t get_builder(const std::string_view, mmap_vecvec*);
private:
  shape(mmap_vecvec*, key_type);
  mmap_vecvec* vecvec_{nullptr};
  key_type index_;
};

}  // namespace nigiri::loader::gtfs