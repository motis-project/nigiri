#include "nigiri/loader/gtfs/feed_info.h"

#include <utl/parser/csv_range.h>

namespace nigiri::loader::gtfs {

feed_info_map_t read_feed_info(std::string_view file_content) {
  struct feed_info {
    utl::csv_col<utl::cstr, UTL_NAME("feed_publisher_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("feed_publisher_url")> url_;
    utl::csv_col<utl::cstr, UTL_NAME("feed_lang")> language_;
    utl::csv_col<utl::cstr, UTL_NAME("feed_start_date")> start_date_;
    utl::csv_col<utl::cstr, UTL_NAME("feed_end_date")> end_date_;
    utl::csv_col<utl::cstr, UTL_NAME("feed_version")> version_;
    utl::csv_col<utl::cstr, UTL_NAME("feed_contact_email")> email_;
  };
}

}  // namespace nigiri::loader::gtfs
