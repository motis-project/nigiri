#include "utl/parser/arg_parser.h"

#include "nigiri/clasz.h"
#include "nigiri/loader/hrd/stamm/category.h"
#include "nigiri/loader/hrd/util.h"

namespace nigiri::loader::hrd {

category_map_t parse_categories(config const& c,
                                std::string_view file_content) {
  auto const timer = nigiri::scoped_timer{"parse categories"};

  bool ignore = false;
  category_map_t handle_map;
  utl::for_each_line_numbered(
      utl::cstr{file_content}, [&](utl::cstr line, unsigned const line_number) {
        if (ignore || line.len <= 1 || line.str[0] == '#' ||
            line.str[0] == '%') {
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
        auto name = iso_8859_1_to_utf8(line.substr(c.cat_.name_).trim().view());
        auto const clasz = get_clasz(name);

        handle_map[code.to_str()] = category{.name_ = code.trim().to_str(),
                                             .long_name_ = std::move(name),
                                             .output_rule_ = output_rule,
                                             .clasz_ = clasz};
      });
  return handle_map;
}

}  // namespace nigiri::loader::hrd
