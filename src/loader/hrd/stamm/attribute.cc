#include "nigiri/loader/hrd/stamm/attribute.h"

#include "nigiri/logging.h"

#include "nigiri/loader/hrd/util.h"
#include "utl/parser/cstr.h"

namespace nigiri::loader::hrd {

bool is_multiple_spaces(utl::cstr line) {
  return line.substr(2, utl::size(3)).trim().empty();
}

attribute_map_t parse_attributes(config const& c,
                                 timetable& tt,
                                 std::string_view file_content) {
  auto const timer = scoped_timer{"parse attributes"};
  attribute_map_t handle_map;
  utl::for_each_line_numbered(file_content, [&](utl::cstr line,
                                                unsigned const line_number) {
    if (line.len == 0 || line.str[0] == '#') {
      return;
    } else if (line.len < 13 || (is_multiple_spaces(line) && line.len < 22)) {
      log(log_lvl::error, "loader.hrd.attribute",
          "invalid attribute line - skipping {}", line_number);
      return;
    }

    auto const code = line.substr(c.att_.code_);
    auto text = is_multiple_spaces(line) ? line.substr(c.att_.text_mul_spaces_)
                                         : line.substr(c.att_.text_normal_);

    if (!text.empty() && text[text.length() - 1] != '\0') {
      text = text.substr(0, text.length() - 1);
    }

    auto const idx = attribute_idx_t{tt.attributes_.size()};
    tt.attributes_.emplace_back(attribute{
        .code_ = code.view(),
        .text_ = tt.register_translation(iso_8859_1_to_utf8(text.view()))});
    handle_map[code.to_str()] = idx;
  });
  return handle_map;
}

}  // namespace nigiri::loader::hrd
