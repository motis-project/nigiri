#include "nigiri/loader/gtfs/feed_info_test.h"

#include "utl/parser/csv_range.h"

#include "nigiri/loader/gtfs/parse_date.h"

namespace nigiri::loader::gtfs {

feed_info_test read_feed_info(std::string_view file_content) {
  struct feed_info_record {
    utl::csv_col<std::optional<unsigned>, UTL_NAME("feed_end_date")>
        feed_end_date_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("default_lang")>
        default_lang_;
  };

  auto ret = feed_info_test{
      .feed_end_date_ = {},
      .default_lang_ = "en",
  };
  utl::for_each_row<feed_info_record>(
      file_content, [&](feed_info_record const& r) {
        ret.feed_end_date_ = r.feed_end_date_->and_then(
            [](unsigned const x) { return std::optional{parse_date(x)}; });
        ret.default_lang_ = r.default_lang_->value_or("en").view();
      });
  return ret;
}

}  // namespace nigiri::loader::gtfs