#pragma once

#include <optional>

#include "utl/parser/cstr.h"

#include "cista/containers/vecvec.h"

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct shape {
public:
  using key_type = uint32_t;
  using value_type = geo::polyline;
  using coordinate = geo::latlng;
  using mmap_vecvec = mm_vecvec<key_type, coordinate>;
  using id_type = utl::cstr;
  using stored_type = coordinate;
  using builder_t = std::function<std::optional<const shape>(const id_type&)>;

  mmap_vecvec::bucket bucket_;

  value_type operator()() const;
  static builder_t get_builder();
  static builder_t get_builder(const std::string_view, mmap_vecvec*);
};

}  // namespace nigiri::loader::gtfs