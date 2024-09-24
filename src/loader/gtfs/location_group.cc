#include "nigiri/loader/gtfs/location_group.h"

#include <utl/parser/csv_range.h>

namespace nigiri::loader::gtfs {

location_group_map_t read_location_group(std::string_view file_content) {
  struct location_group {
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_id")> location_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_group_name")> location_group_name_;
  };
}

}  // namespace nigiri::loader::gtfs
