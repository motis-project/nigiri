#include "nigiri/loader/hrd/category.h"

#include "utl/parser/arg_parser.h"

namespace nigiri::loader::hrd {

hash_map<std::string, info_db::handle_t> parse_categories(
    config const& c, info_db& db, std::string_view file_content) {
  bool ignore = false;
  hash_map<std::string, info_db::handle_t> handle_map;
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

    auto const code = line.substr(c.cat_.code_).view();
    auto const output_rule = output_rule_t{utl::parse_verify<std::uint8_t>(
        line.substr(c.cat_.output_rule_).trim())};
    auto const name = line.substr(c.cat_.name_).trim().view();

    handle_map[code] = db.add(category{
        .short_name_ = code, .long_name_ = name, .output_rule_ = output_rule});
  });
  db.flush();
  return handle_map;
}

}  // namespace nigiri::loader::hrd
