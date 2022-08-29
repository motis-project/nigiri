#include "nigiri/loader/hrd/direction.h"

#include "utl/parser/cstr.h"
#include "utl/verify.h"

namespace nigiri::loader::hrd {

direction_map_t parse_directions(config const& c,
                                 std::string_view file_content) {
  direction_map_t directions;
  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, int line_number) {
        if (line.length() < 9 && line[7] == ' ') {
          throw utl::fail("parse_directions: invalid line format in line {}",
                          line_number);
        } else {
          directions[line.substr(c.dir_.eva_).to_str()] =
              line.substr(c.dir_.text_).to_str();
        }
      });
  return directions;
}

}  // namespace nigiri::loader::hrd