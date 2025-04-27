#include "nigiri/loader/gtfs/flex.h"

#include <ranges>

#include "boost/json.hpp"

#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"

#include "geo/box.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

string_idx_t to_str(timetable& tt, auto const& col) {
  return col
      ->and_then([&](utl::cstr const& x) {
        return std::optional{tt.strings_.store(x.view())};
      })
      .value_or(string_idx_t::invalid());
}

flex_areas_t parse_flex_areas(timetable& tt, std::string_view file_content) {
  using tmp_ring_t = std::vector<geo::latlng>;

  auto const to_latlng = [](boost::json::array const& x) -> geo::latlng {
    return {x.at(1).as_double(), x.at(0).as_double()};
  };
  auto const to_ring = [&](boost::json::array const& x) -> tmp_ring_t {
    return utl::to_vec(x, [&](auto&& y) { return to_latlng(y.as_array()); });
  };

  auto outer = std::vector<tmp_ring_t>{};
  auto inner = std::vector<std::vector<tmp_ring_t>>{};

  auto map = flex_areas_t{};
  auto const json = boost::json::parse(file_content).as_object();
  for (auto const& x : json.at("features").as_array()) {
    try {
      outer.clear();
      inner.clear();

      auto const id = x.at("id").as_string();
      auto const geometry = x.at("geometry").as_object();
      auto const geometry_type = geometry.at("type").as_string();
      auto const rings = geometry.at("coordinates").as_array();

      utl::verify(geometry_type == "Polygon",
                  "only Polygon supported at the moment, type={}",
                  geometry_type);

      outer.emplace_back(to_ring(rings.at(0).as_array()));
      auto& inners = inner.emplace_back();
      for (auto i = 1U; i < rings.size(); ++i) {
        inners.emplace_back(to_ring(rings.at(i).as_array()));
      }

      auto const idx = flex_area_idx_t{tt.flex_area_outers_.size()};
      tt.flex_area_outers_.emplace_back(outer);
      tt.flex_area_inners_.emplace_back(inner);

      auto box = geo::box{};
      for (auto const& o : outer[0]) {
        box.extend(o);
      }
      tt.flex_area_rtree_.insert(box.min_.lnglat_float(),
                                 box.max_.lnglat_float(), idx);

      map.emplace(id, idx);
    } catch (std::exception const& e) {
      log(log_lvl::error, "loader.gtfs.flex.locations",
          "GeoJSON parsing error: {}, json: {}", e.what(),
          boost::json::serialize(x));
      continue;
    }
  }
  return map;
}

location_groups_t parse_location_groups(timetable& tt,
                                        std::string_view file_content) {
  struct location_group_record {
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("location_group_name")>
        location_group_name_;
  };

  auto map = location_groups_t{};
  utl::for_each_row<location_group_record>(
      file_content, [&](location_group_record const& r) {
        auto const idx = location_group_idx_t{tt.location_group_name_.size()};
        tt.location_group_name_.emplace_back(
            to_str(tt, r.location_group_name_));
        tt.location_group_.emplace_back_empty();
        map.emplace(r.location_group_id_->to_str(), idx);
      });
  return map;
}

void parse_location_group_stops(timetable& tt,
                                std::string_view file_content,
                                location_groups_t const& location_groups,
                                stops_map_t const& stops) {
  struct location_group_record {
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
  };
  utl::for_each_row<location_group_record>(
      file_content, [&](location_group_record const& r) {
        auto const location_group_it =
            location_groups.find(r.location_group_id_->view());
        if (location_group_it == end(location_groups)) {
          log(log_lvl::error, "nigiri.loader.gtfs.flex",
              "location_group_id={} not found", r.location_group_id_->view());
          return;
        }

        auto const stop_it = stops.find(r.stop_id_->view());
        if (stop_it == end(stops)) {
          log(log_lvl::error, "nigiri.loader.gtfs.flex",
              "location_group_id={} not found", r.location_group_id_->view());
          return;
        }

        tt.location_group_[location_group_it->second].push_back(
            stop_it->second);
      });
}

booking_rules_t parse_booking_rules(
    timetable& tt,
    std::string_view file_content,
    traffic_days_t const& traffic_days,
    hash_map<bitfield, bitfield_idx_t>& bitfield_indices) {
  using utl::csv_col;

  struct booking_rule_record {
    csv_col<utl::cstr, UTL_NAME("booking_rule_id")> booking_rule_id_;
    csv_col<std::uint8_t, UTL_NAME("booking_type")> booking_type_;
    csv_col<std::optional<std::int32_t>, UTL_NAME("prior_notice_duration_min")>
        prior_notice_duration_min_;
    csv_col<std::optional<std::int32_t>, UTL_NAME("prior_notice_duration_max")>
        prior_notice_duration_max_;
    csv_col<std::optional<std::uint16_t>, UTL_NAME("prior_notice_last_day")>
        prior_notice_last_day_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("prior_notice_last_time")>
        prior_notice_last_time_;
    csv_col<std::optional<std::uint16_t>, UTL_NAME("prior_notice_start_day")>
        prior_notice_start_day_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("prior_notice_start_time")>
        prior_notice_start_time_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("prior_notice_service_id")>
        prior_notice_service_id_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("message")> message_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("pickup_message")>
        pickup_message_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("drop_off_message")>
        drop_off_message_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("phone_number")> phone_number_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("info_url")> info_url_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("booking_url")> booking_url_;
  };

  auto map = booking_rules_t{};
  utl::for_each_row<
      booking_rule_record>(file_content, [&](booking_rule_record const& r) {
    auto const get_type = [&]() -> booking_rule::booking_type {
      switch (*r.booking_type_) {
        case 0: return booking_rule::real_time{};

        case 1: {
          auto prior_notice = booking_rule::prior_notice{};
          if (r.prior_notice_duration_min_->has_value()) {
            prior_notice.prior_notice_duration_min_ =
                i32_minutes{**r.prior_notice_duration_min_};
          }
          if (r.prior_notice_duration_max_->has_value()) {
            prior_notice.prior_notice_duration_max_ =
                i32_minutes{**r.prior_notice_duration_max_};
          }
          return prior_notice;
        }

        case 2: {
          auto prior_day = booking_rule::prior_day{};
          if (r.prior_notice_last_day_->has_value()) {
            prior_day.prior_notice_last_day_ = **r.prior_notice_last_day_;
          }
          if (r.prior_notice_last_time_->has_value()) {
            prior_day.prior_notice_last_time_ =
                hhmm_to_min(**r.prior_notice_last_time_);
          }
          if (r.prior_notice_start_day_->has_value()) {
            prior_day.prior_notice_start_day_ = **r.prior_notice_start_day_;
          }
          if (r.prior_notice_start_time_->has_value()) {
            prior_day.prior_notice_start_time_ =
                hhmm_to_min(**r.prior_notice_start_time_);
          }
          if (r.prior_notice_service_id_->has_value()) {
            auto const bitfield_it =
                traffic_days.find((*r.prior_notice_service_id_)->view());
            if (bitfield_it == end(traffic_days)) {
              log(log_lvl::error, "nigiri.loader.gtfs.flex",
                  "service_id={} not found, falling back to real-time booking",
                  (*r.prior_notice_service_id_)->view());
              return booking_rule::real_time{};
            }
            prior_day.prior_notice_bitfield_ = utl::get_or_create(
                bitfield_indices, *bitfield_it->second,
                [&]() { return tt.register_bitfield(*bitfield_it->second); });
          }
          return prior_day;
        }
      }
      return booking_rule::real_time{};
    };

    auto const idx = booking_rule_idx_t{tt.booking_rules_.size()};
    tt.booking_rules_.emplace_back(
        booking_rule{.id_ = tt.strings_.store(r.booking_rule_id_->view()),
                     .type_ = get_type(),
                     .message_ = to_str(tt, r.message_),
                     .pickup_message_ = to_str(tt, r.pickup_message_),
                     .drop_off_message_ = to_str(tt, r.drop_off_message_),
                     .phone_number_ = to_str(tt, r.phone_number_),
                     .info_url_ = to_str(tt, r.info_url_),
                     .booking_url_ = to_str(tt, r.booking_url_)});
    map.emplace(r.booking_rule_id_->to_str(), idx);
  });
  return map;
}

}  // namespace nigiri::loader::gtfs