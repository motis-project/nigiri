#pragma once

#include <cstdio>

#include "cista/reflection/comparable.h"
#include "cista/reflection/printable.h"

#include "geo/latlng.h"

#include "nigiri/database.h"
#include "nigiri/types.h"

namespace nigiri {

struct location {
  string name_;
  geo::latlng pos_;
};

struct section_info {
  CISTA_COMPARABLE()
  std::uint32_t category_idx_{0U};
  std::uint32_t attribute_idx_{0U};
  std::uint32_t provider_idx_{0U};
  std::uint32_t direction_idx_{0U};
  std::uint32_t line_idx_{0U};

  std::uint32_t train_nr_{0U};
  std::uint32_t clasz_{0U};
};

struct category {
  CISTA_COMPARABLE()
  string long_name_, short_name_;
};

struct attribute {
  CISTA_PRINTABLE(attribute)
  CISTA_COMPARABLE()
  string code_, text_;
};

struct provider {
  CISTA_COMPARABLE()
  string short_name_, long_name_;
};

using line_id_t = string;

using direction_t = cista::variant<location_idx_t, string>;

using info_db = database<location, category, attribute, provider, line_id_t,
                         direction_t, section_info>;

}  // namespace nigiri
