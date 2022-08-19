#pragma once

#include "fmt/format.h"

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/section_db.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct category {
  CISTA_PRINTABLE(category, "name", "long_name", "output_rule", "clasz")
  friend bool operator==(category const&, category const&) = default;
  string name_, long_name_;
  std::uint8_t output_rule_;
  clasz clasz_{clasz::kAir};
};

using category_map_t = hash_map<std::string, category>;

category_map_t parse_categories(config const&, std::string_view file_content);

}  // namespace nigiri::loader::hrd
