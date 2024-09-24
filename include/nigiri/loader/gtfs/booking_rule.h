#pragma once

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {
  struct booking_rule {
    std::string id_;
    uint8_t type_;
    uint16_t prior_notice_duration_min_;
    uint16_t prior_notice_duration_max_;
    uint16_t prior_notice_last_day_;
    std::string message_;
    std::string phone_number_;
    std::string info_url_;
    std::string booking_url_;
  };

  using booking_rule_map_t = hash_map<std::string_view, std::unique_ptr<booking_rule>>;

  booking_rule_map_t read_booking_rule(std::string_view file_content);
}  // namespace nigiri::loader::gtfs

