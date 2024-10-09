#pragma once
#include "td_agency.h"
#include "td_area.h"
#include "td_booking_rule.h"
#include "td_calendar.h"
#include "td_calendar_dates.h"
#include "td_locationGeojson.h"
#include "td_location_group.h"
#include "td_location_groups_stop.h"
#include "td_route.h"
#include "td_stop.h"
#include "td_stop_area.h"
#include "td_stop_time.h"
#include "td_trip.h"

namespace nigiri::loader::gtfs_flex {
  struct td_timetable_t {
    std::optional<uint8_t> pick_up_booking_rule_type_;
    std::optional<uint16_t> pick_up_booking_rule_prior_notice_duration_min_;
    std::optional<uint16_t> pick_up_booking_rule_prior_notice_duration_max_;
    std::optional<uint16_t> pick_up_booking_rule_prior_notice_last_day_;
    std::optional<duration_t> pick_up_booking_rule_prior_notice_last_time_;
    std::optional<uint16_t> pick_up_booking_rule_prior_notice_start_day_;
    std::optional<duration_t> pick_up_booking_rule_prior_notice_start_time_;
    std::optional<std::string> pick_up_booking_rule_prior_notice_service_id_;
    std::optional<std::string> pick_up_booking_rule_message_;


    std::optional<uint8_t> drop_off_booking_rule_type_;
    std::optional<uint16_t> drop_off_booking_rule_prior_notice_duration_min_;
    std::optional<uint16_t> drop_off_booking_rule_prior_notice_duration_max_;
    std::optional<uint16_t> drop_off_booking_rule_prior_notice_last_day_;
    std::optional<duration_t> drop_off_booking_rule_prior_notice_last_time_;
    std::optional<uint16_t> drop_off_booking_rule_prior_notice_start_day_;
    std::optional<duration_t> drop_off_booking_rule_prior_notice_start_time_;
    std::optional<std::string> drop_off_booking_rule_prior_notice_service_id_;
    std::string drop_off_booking_rule_message_;
    std::string drop_off_booking_rule_pickup_message_;


    std::optional<std::string> booking_rule_pickup_message_;
    std::optional<std::string> booking_rule_drop_off_message_;
    std::optional<std::string> booking_rule_phone_number_;
    std::optional<std::string> booking_rule_info_url_;
    std::optional<std::string> booking_rule_booking_url_;

    std::optional<std::bitset<7>> booking_rule_calendar_week_days_;
    std::optional<interval<date::sys_days>> booking_rule_calendar_interval_;
    std::optional<std::vector<calendar_date>> booking_rule_calendar_date_exceptions_;

    std::optional< std::bitset<7>> trip_calendar_week_days_;
    std::optional<interval<date::sys_days>> trip_calendar_interval_;
    std::optional<std::vector<calendar_date>> trip_calendar_date_exceptions_;

    std::optional<std::vector<std::string>> location_group_location_ids_;

    std::optional<std::vector<std::string>> location_group_stop_stop_ids_;

    std::optional<std::pair<GEOMETRY_TYPE, tg_geom*>> location_geojson_location;

    std::optional<std::string> stop_lat_;
    std::optional<std::string> stop_lon_;
    std::optional<uint8_t> stop_location_type_;
    std::optional<std::string> stop_parent_station_;

    std::optional<std::vector<std::string>> stop_area_stop_ids_;

    uint16_t stop_time_stop_sequence_;
    duration_t stop_time_start_pickup_drop_off_window_;
    duration_t stop_time_end_pickup_drop_off_window_;
    double_t stop_time_mean_duration_factor_;
    double_t stop_time_mean_duration_offset_;
    double_t stop_time_safe_duration_factor_;
    double_t stop_time_safe_duration_offset_;
    uint8_t stop_time_pickup_type_;
    uint8_t stop_time_drop_off_type_;
    std::optional<std::string> stop_time_pickup_booking_rule_id_;
    std::optional<std::string> stop_time_drop_off_booking_rule_id_;
  };
  using td_stop_id_t = std::string;
  using td_trip_id_t = std::string;
  using td_timetable_id_t = std::pair<td_stop_id_t, td_trip_id_t>;

  using td_timetable_map_t = hash_map<td_timetable_id_t, std::unique_ptr<td_timetable_t>>;

  td_timetable_map_t create_td_timetable( td_booking_rule_map_t booking_rules,
                                          td_calendar_map_t calendar,
                                          td_calendar_date_map_t calendar_dates,
                                          td_location_group_map_t location_groups,
                                          td_location_group_stop_map_t location_group_stops,
                                          td_location_geojson_map_t location_geojson,
                                          td_stop_map_t stops,
                                          td_stop_area_map_t stop_areas,
                                          td_stop_time_map_t stop_times,
                                          td_trip_map_t trips);
}