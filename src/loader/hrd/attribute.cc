#include "nigiri/loader/hrd/attribute.h"

#include "nigiri/logging.h"

#include "utl/parser/cstr.h"

namespace nigiri::loader::hrd {

bool is_multiple_spaces(utl::cstr line) {
  return line.substr(2, utl::size(3)).trim().empty();
}

hash_map<std::string, info_db::handle_t> parse_attributes(
    config const& c, info_db& db, utl::cstr const& file_content) {
  scoped_timer timer("nigiri.loader.hrd.attribute");
  hash_map<std::string, info_db::handle_t> handle_map;
  for_each_line_numbered(file_content, [&](utl::cstr line,
                                           unsigned const line_number) {
    if (line.len == 0 || line.str[0] == '#') {
      return;
    } else if (line.len < 13 || (is_multiple_spaces(line) && line.len < 22)) {
      log(log_lvl::error, "nigiri.loader.hrd.attribute",
          "invalid attribute line - skipping {}", line_number);
      return;
    }

    if (auto const comment_start_pos = line.view().find('#');
        comment_start_pos != std::string_view::npos) {
      line = line.substr(0, comment_start_pos);
    }

    auto const code = line.substr(c.att_.code_);
    auto const text = is_multiple_spaces(line)
                          ? line.substr(c.att_.text_mul_spaces_)
                          : line.substr(c.att_.text_normal_);
    handle_map[code.to_str()] =
        db.add(attribute{.code_ = code.view(), .text_ = text.view()});
  });
  db.flush();
  return handle_map;
}

}  // namespace nigiri::loader::hrd
