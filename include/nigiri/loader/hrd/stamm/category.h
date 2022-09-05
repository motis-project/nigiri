#pragma once

#include "fmt/format.h"

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct category {
  friend std::ostream& operator<<(std::ostream& out, category const& c) {
    return out << "(name=" << c.name_ << ", long_name=" << c.long_name_
               << ", output_rule=" << static_cast<int>(c.output_rule_)
               << ", clasz=" << static_cast<int>(c.clasz_) << ")";
  }
  friend bool operator==(category const&, category const&) = default;
  std::string name_, long_name_;
  std::uint8_t output_rule_;
  clasz clasz_{clasz::kAir};
};

using category_map_t = hash_map<std::string, category>;

category_map_t parse_categories(config const&, std::string_view file_content);

}  // namespace nigiri::loader::hrd
