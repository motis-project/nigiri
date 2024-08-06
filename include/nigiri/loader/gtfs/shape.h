#pragma once

#include <cstddef>
#include <filesystem>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "utl/parser/cstr.h"

#include "cista/containers/vecvec.h"

#include "geo/polyline.h"

#include "nigiri/types.h"

// #include "osmium/osm/location.hpp"
// #include "osr/types.h"

namespace nigiri::loader::gtfs {

class shape {
public:
  constexpr static int32_t coordinate_precision{10000000};

  using key_type = uint32_t;
  using value_type = geo::polyline;
  using shape_coordinate_type =
      std::remove_const<decltype(coordinate_precision)>::type;
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
  shape(mmap_vecvec::bucket);
  mmap_vecvec::bucket bucket_;
};

}  // namespace nigiri::loader::gtfs