#include "nigiri/loader/gtfs-flex/td_timetable.h"

namespace nigiri::loader::gtfs_flex {
  td_timetable_map_t create_td_timetable(   td_booking_rule_map_t booking_rules,
                                            td_calendar_map_t calendar,
                                            td_calendar_date_map_t calendar_dates,
                                            td_location_group_map_t location_groups,
                                            td_location_group_stop_map_t location_group_stops,
                                            td_location_geojson_map_t location_geojson,
                                            td_stop_map_t stops,
                                            td_stop_area_map_t stop_areas,
                                            td_stop_time_map_t stop_times,
                                            td_trip_map_t trips) {
    td_timetable_map_t td_timetable{};
    for(auto stop_time : stop_times) {
      auto timetable_id = stop_time.first;
      td_stop_id_t stop_id = timetable_id.first;
      td_trip_id_t trip_id = timetable_id.second;




      std::string any_booking_rule_id = "";
      auto pick_up_booking_rule_id = stop_times[timetable_id]->pickup_booking_rule_id_;
      auto drop_off_booking_rule_id = stop_times[timetable_id]->pickup_booking_rule_id_;

      if(!pick_up_booking_rule_id.empty()) {
        any_booking_rule_id = pick_up_booking_rule_id;
      }
      else if(!drop_off_booking_rule_id.empty()) {
        any_booking_rule_id = drop_off_booking_rule_id;
      }

      auto trip_service_id = trips[trip_id]->service_id_;
      std::string booking_rule_service_id = "";
      if(!any_booking_rule_id.empty()) {
        booking_rule_service_id = booking_rules[any_booking_rule_id]->prior_notice_service_id_;
      }

      td_timetable.emplace(timetable_id,
      std::make_unique<td_timetable>{
        .pick_up_booking_rule_type_ = pick_up_booking_rule_id ? std::nullopt : booking_rules[pick_up_booking_rule_id]->type_,
        .pick_up_booking_rule_prior_notice_duration_min_ = pick_up_booking_rule_id.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->prior_notice_duration_min_,
        .pick_up_booking_rule_prior_notice_duration_max_ = stop_times[timetable_id]->pickup_booking_rule_id_.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->prior_notice_duration_max_,
        .pick_up_booking_rule_prior_notice_last_day_ = stop_times[timetable_id]->pickup_booking_rule_id_.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->prior_notice_last_day_,
        .pick_up_booking_rule_prior_notice_last_time_ = stop_times[timetable_id]->pickup_booking_rule_id_.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->prior_notice_last_time_,
        .pick_up_booking_rule_prior_notice_start_day_ = stop_times[timetable_id]->pickup_booking_rule_id_.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->prior_notice_start_day_,
        .pick_up_booking_rule_prior_notice_start_time_ = stop_times[timetable_id]->pickup_booking_rule_id_.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->prior_notice_start_time_,
        .pick_up_booking_rule_prior_notice_service_id_ = stop_times[timetable_id]->pickup_booking_rule_id_.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->prior_notice_service_id_,
        .pick_up_booking_rule_message_ = stop_times[timetable_id]->pickup_booking_rule_id_.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->message_,
        .booking_rule_pickup_message_ = stop_times[timetable_id]->pickup_booking_rule_id_.empty() ? std::nullopt : booking_rules[pick_up_booking_rule_id]->pickup_message_,

        .drop_off_booking_rule_type_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->type_,
        .drop_off_booking_rule_prior_notice_duration_min_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->prior_notice_duration_min_,
        .drop_off_booking_rule_prior_notice_duration_max_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->prior_notice_duration_max_,
        .drop_off_booking_rule_prior_notice_last_day_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->prior_notice_last_day_,
        .drop_off_booking_rule_prior_notice_last_time_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->prior_notice_last_time_,
        .drop_off_booking_rule_prior_notice_start_day_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->prior_notice_start_day_,
        .drop_off_booking_rule_prior_notice_start_time_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->prior_notice_start_time_,
        .drop_off_booking_rule_prior_notice_service_id_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->prior_notice_service_id_,
        .drop_off_booking_rule_message_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->message_,
        .booking_rule_drop_off_message_ = drop_off_booking_rule_id.empty() ? std::nullopt : booking_rules[drop_off_booking_rule_id]->drop_off_message_,

        .booking_rule_phone_number_ = any_booking_rule_id.empty() ? std::nullopt : booking_rules[any_booking_rule_id]->phone_number_,
        .booking_rule_info_url_ = any_booking_rule_id.empty() ? std::nullopt : booking_rules[any_booking_rule_id]->info_url_,
        .booking_rule_booking_url_ = any_booking_rule_id.empty() ? std::nullopt : booking_rules[any_booking_rule_id]->booking_url_,

        .booking_rule_calendar_week_days = booking_rule_service_id.empty() ? std::nullopt : calendar[booking_rule_service_id]->week_days_,
        .booking_rule_calendar_interval_ = booking_rule_service_id.empty() ? std::nullopt : calendar[booking_rule_service_id]->interval_,
        .booking_rule_calendar_date_exceptions = booking_rule_service_id.empty() ? std::nullopt : calendar_dates[booking_rule_service_id],

        .trip_calendar_week_days_ = trip_service_id.empty() ? std::nullopt : calendar[trip_service_id]->week_days_,
        .trip_calendar_interval_ = trip_service_id.empty() ? std::nullopt : calendar[trip_service_id]->interval_,
        .trip_calendar_date_exceptions_ = trip_service_id.empty() ? std::nullopt : calendar_dates[trip_service_id],

        .location_group_location_ids_ = location_groups.count(stop_id) > 0 ? location_groups[stop_id].location_ids_ : std::nullopt,

        .location_group_stop_stop_ids_ = location_group_stops.count(stop_id) > 0 ? location_group_stops[stop_id] : std::nullopt,

        .location_geojson_location = location_geojson.count(stop_id) > 0 ? location_geojson[stop_id] : std::nullopt,

        .stop_lat_ = stops.count(stop_id) > 0 ? stops[stop_id]->lat_ : std::nullopt,
        .stop_lon_ = stops.count(stop_id) > 0 ? stops[stop_id]->lon_ : std::nullopt,
        .stop_location_type_ = stops.count(stop_id) > 0 ? stops[stop_id]->location_type_ : std::nullopt,
        .stop_parent_station_ = stops.count(stop_id) > 0 ? stops[stop_id]->parent_station_ : std::nullopt,

        .stop_area_stop_ids_ = stop_areas.count(stop_id) > 0 ? stop_areas[stop_id] : std::nullopt,

        .stop_time_stop_sequence_ = stop_times[timetable_id]->stop_sequence_,
        .stop_time_start_pickup_drop_off_window_ = stop_times[timetable_id]->start_pickup_drop_off_window_,
        .stop_time_end_pickup_drop_off_window_ = stop_times[timetable_id]->end_pickup_drop_off_window_,
        .stop_time_mean_duration_factor_ = stop_times[timetable_id]->mean_duration_factor_,
        .stop_time_mean_duration_offset_ = stop_times[timetable_id]->mean_duration_offset_,
        .stop_time_safe_duration_factor_ = stop_times[timetable_id]->safe_duration_factor_,
        .stop_time_safe_duration_offset_ = stop_times[timetable_id]->safe_duration_offset_,
        .stop_time_pickup_type_ = stop_times[timetable_id]->pickup_type_,
        .stop_time_drop_off_type_ = stop_times[timetable_id]->drop_off_type_,
        .stop_time_pickup_booking_rule_id_ = stop_times[timetable_id]->pickup_booking_rule_id_,
        .stop_time_drop_off_booking_rule_id_ = stop_times[timetable_id]->drop_off_booking_rule_id_
      });
    }
    return td_timetable;
  }
}