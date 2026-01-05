#include "nigiri/loader/hrd/stamm/direction.h"

#include "utl/parser/cstr.h"
#include "utl/verify.h"

#include "nigiri/loader/hrd/util.h"
#include "nigiri/logging.h"

namespace nigiri::loader::hrd {

direction_map_t parse_directions(config const& c,
                                 timetable& tt,
                                 std::string_view file_content) {
  auto const timer = nigiri::scoped_timer{"parse directions"};
  direction_map_t directions;
  utl::for_each_line_numbered(file_content, [&](utl::cstr line,
                                                unsigned const line_number) {
    if (line.length() < 9 && line[7] == ' ') {
      throw utl::fail("parse_directions: invalid line format in line {}",
                      line_number);
    } else {
      directions[line.substr(c.dir_.eva_).to_str()] = tt.register_translation(
          iso_8859_1_to_utf8(line.substr(c.dir_.text_).view()));
    }
  });
  return directions;
}

}  // namespace nigiri::loader::hrd