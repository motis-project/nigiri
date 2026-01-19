#pragma once

#include "date/date.h"

namespace nigiri::loader::gtfs {

struct feed_info_test {
  std::optional<date::sys_days> feed_end_date_;
  std::string default_lang_;
};

feed_info_test read_feed_info(std::string_view);

}  // namespace nigiri::loader::gtfs