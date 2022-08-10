#include "nigiri/loader/hrd/category.h"

#include "utl/parser/arg_parser.h"

namespace nigiri::loader::hrd {

category_map_t parse_categories(config const& c,
                                std::string_view file_content) {
  bool ignore = false;
  category_map_t handle_map;
  utl::for_each_line_numbered(utl::cstr{file_content}, [&](utl::cstr line,
                                                           int line_number) {
    if (ignore || line.len <= 1 || line.str[0] == '#' || line.str[0] == '%') {
      return;
    } else if (line.starts_with("<")) {
      ignore = true;
      return;
    } else if (line.len < 20) {
      throw utl::fail("category line length <20 (line={})", line_number);
    }

    auto const code = line.substr(c.cat_.code_);
    auto const output_rule = utl::parse_verify<std::uint8_t>(
        line.substr(c.cat_.output_rule_).trim());
    auto const name = line.substr(c.cat_.name_).trim().view();

    handle_map[code.to_str()] = category{.name_ = code.trim().to_str(),
                                         .long_name_ = name,
                                         .output_rule_ = output_rule};
  });
  return handle_map;
}

}  // namespace nigiri::loader::hrd
