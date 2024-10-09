#pragma once

#include "nigiri/types.h"

#include <bitset>

namespace nigiri::loader::gtfs_flex {
const uint16_t REAL_TIME_BOOKING = 0;
const uint16_t SAME_DAY_BOOKING = 1;
const uint16_t PRIOR_DAYS_BOOKING = 2;

struct td_booking_rule {
  uint8_t type_;                        //Required 0=Real-Time-Booking, 1=Same-Day-Booking, 2=Prior-Day-Booking
  uint16_t prior_notice_duration_min_;  //Conditionally Required If booking_type=1
  uint16_t prior_notice_duration_max_;  //Conditionally Forbidden For booking_type=0 And booking_type=2
  uint16_t prior_notice_last_day_;      //Conditionally Required If booking_type=2
  duration_t prior_notice_last_time_;   //Conditionally Required If prior_notice_last_day Is Defined
  uint16_t prior_notice_start_day_;     //Conditionally Forbidden For booking_type=0 And For booking_type=1 If prior_notice_duration_max Is Defined
  duration_t prior_notice_start_time_;  //Conditionally Required If prior_notice_start_day Is Defined
  std::string prior_notice_service_id_; //Conditionally Forbidden If booking_type=0 And booking_type=1
  std::string message_;                 //Optional
  std::string pickup_message_;          //Optional
  std::string drop_off_message_;        //Optional
  std::string phone_number_;            //Optional
  std::string info_url_;                //Optional
  std::string booking_url_;             //Optional
};

using td_booking_rule_map_t = hash_map<std::string_view, std::unique_ptr<td_booking_rule>>;

td_booking_rule_map_t read_td_booking_rules(std::string_view file_content);
}  // namespace nigiri::loader::gtfs_flex

