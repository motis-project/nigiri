#pragma once

#include "date/date.h"

namespace nigiri::loader::gtfs {

struct feed_info {
  std::optional<date::sys_days> feed_end_date_;
  std::optional<std::string> feed_lang_;
  std::string default_lang_;
};

feed_info read_feed_info(std::string_view);

}  // namespace nigiri::loader::gtfs
