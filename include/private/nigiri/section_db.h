#pragma once

#include <cstdio>

#include "nigiri/types.h"

namespace nigiri {

struct section_info {
  std::uint32_t category_idx_{0U};
  std::uint32_t attribute_idx_{0U};
  std::uint32_t provider_idx_{0U};
  std::uint32_t direction_idx_{0U};
  std::uint32_t line_idx_{0U};

  std::uint32_t train_nr_{0U};
  std::uint32_t clasz_{0U};
};

struct category {
  string long_name_, short_name_;
};

struct attribute {
  string code_, text_;
};

struct provider {
  string long_name_, short_name_;
};

using line_id_t = string;

using direction_t = cista::variant<location_idx_t, string>;

}  // namespace nigiri
