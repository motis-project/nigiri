#include "nigiri/loader/gtfs/flex.h"

#include <ranges>

#include "boost/json.hpp"

#include "utl/get_or_create.h"
#include "utl/parser/csv_range.h"

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

flex_areas_t parse_flex_areas(timetable& tt,
                              source_idx_t const src,
                              std::string_view file_content) {
  if (file_content.empty()) {
    return {};
  }

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

      auto const& area = x.as_object();
      auto const& id = area.at("id").as_string();
      auto const geometry = area.at("geometry").as_object();
      auto const geometry_type = geometry.at("type").as_string();

      utl::verify(geometry_type == "Polygon",
                  "only Polygon supported at the moment, type={}",
                  std::string_view{geometry_type});

      auto const properties = area.find("properties");
      if (properties != area.end() && properties->value().is_object()) {
        auto const& props = properties->value().as_object();

        auto const name = props.find("stop_name");
        tt.flex_area_name_.emplace_back(
            (name != props.end() && name->value().is_string())
                ? name->value().as_string()
                : "");

        auto const desc = props.find("stop_desc");
        tt.flex_area_desc_.emplace_back(
            (desc != props.end() && desc->value().is_string())
                ? desc->value().as_string()
                : "");
      } else {
        tt.flex_area_name_.emplace_back("");
        tt.flex_area_desc_.emplace_back("");
      }

      auto const rings = geometry.at("coordinates").as_array();
      outer.emplace_back(to_ring(rings.at(0).as_array()));
      auto& inners = inner.emplace_back();
      for (auto i = 1U; i < rings.size(); ++i) {
        inners.emplace_back(to_ring(rings.at(i).as_array()));
      }

      auto const idx = flex_area_idx_t{tt.flex_area_outers_.size()};
      tt.flex_area_outers_.emplace_back(outer);
      tt.flex_area_inners_.emplace_back(inner);

      auto box = geo::box{};
      box.extend(outer[0]);
      tt.flex_area_id_.emplace_back(tt.strings_.store(id));
      tt.flex_area_src_.emplace_back(src);
      tt.flex_area_bbox_.emplace_back(box);
      tt.flex_area_transports_.emplace_back_empty();
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
        tt.location_group_id_.emplace_back(
            tt.strings_.store(r.location_group_id_->view()));
        tt.location_group_name_.emplace_back(
            to_str(tt, r.location_group_name_));
        tt.location_group_locations_.emplace_back_empty();
        tt.location_group_transports_.emplace_back_empty();
        map.emplace(r.location_group_id_->to_str(), idx);
      });
  return map;
}

void parse_location_group_stops(timetable& tt,
                                std::string_view file_content,
                                location_groups_t const& location_groups,
                                stops_map_t const& stops) {
  tt.location_location_groups_.resize(tt.n_locations());

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

        tt.location_location_groups_[stop_it->second].push_back(
            location_group_it->second);
        tt.location_group_locations_[location_group_it->second].push_back(
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

void expand_flex_trip(timetable& tt,
                      hash_map<bitfield, bitfield_idx_t>& bitfield_indices,
                      stop_seq_map_t& stop_seq,
                      noon_offset_hours_t const& noon_offsets,
                      interval<date::sys_days> const& selection,
                      trip const& trp) {
  if (trp.flex_time_windows_.empty()) {
    return;
  }

  auto const stop_seq_idx =
      utl::get_or_create(stop_seq, trp.flex_stops_, [&]() {
        auto idx = flex_stop_seq_idx_t{tt.flex_stop_seq_.size()};
        tt.flex_stop_seq_.emplace_back(trp.flex_stops_);
        return idx;
      });
  auto const tt_interval = tt.internal_interval_days();
  auto utc_time_traffic_days = hash_map<duration_t /* tz offset */, bitfield>{};
  for (auto day = tt_interval.from_; day != tt_interval.to_;
       day += date::days{1}) {
    if (!selection.contains(day)) {
      continue;
    }

    auto const gtfs_local_day_idx =
        static_cast<std::size_t>((day - tt_interval.from_).count());
    if (!trp.service_->test(gtfs_local_day_idx)) {
      continue;
    }

    auto const tz_offset =
        noon_offsets.at(tt.providers_[trp.route_->agency_].tz_)
            .value()
            .at(gtfs_local_day_idx);

    auto const first_dep_time = trp.flex_time_windows_.front().start_;
    auto const first_dep_utc = first_dep_time - tz_offset;
    auto const first_dep_day_offset = date::days{static_cast<date::days::rep>(
        std::floor(static_cast<double>(first_dep_utc.count()) / 1440))};
    auto const utc_traffic_day =
        (day - tt_interval.from_ + first_dep_day_offset).count();

    utc_time_traffic_days[tz_offset].set(
        static_cast<std::size_t>(utc_traffic_day));
  }

  for (auto const& [tz_offset, traffic_days] : utc_time_traffic_days) {
    using std::views::transform;
    auto const first_time = trp.flex_time_windows_.front().start_;
    auto const first_time_utc = first_time - tz_offset;
    auto const first_day_offset = date::days{static_cast<date::days::rep>(
        std::floor(static_cast<double>(first_time_utc.count()) / 1440))};

    auto const idx =
        flex_transport_idx_t{tt.flex_transport_traffic_days_.size()};

    tt.flex_transport_traffic_days_.push_back(utl::get_or_create(
        bitfield_indices, traffic_days,
        [&]() { return tt.register_bitfield(traffic_days); }));
    tt.flex_transport_stop_time_windows_.emplace_back(
        trp.flex_time_windows_ |
        transform([&](auto&& w) -> interval<duration_t> {
          return {w.start_ - tz_offset - first_day_offset,
                  w.end_ - tz_offset - first_day_offset};
        }));
    tt.flex_transport_pickup_booking_rule_.emplace_back(
        trp.flex_time_windows_ |
        transform([&](auto&& w) { return w.pickup_booking_rule_; }));
    tt.flex_transport_drop_off_booking_rule_.emplace_back(
        trp.flex_time_windows_ |
        transform([&](auto&& w) { return w.drop_off_booking_rule_; }));
    tt.flex_transport_stop_seq_.emplace_back(stop_seq_idx);
    tt.flex_transport_trip_.emplace_back(trp.trip_idx_);

    for (auto const& s : trp.flex_stops_) {
      if (holds_alternative<flex_area_idx_t>(s)) {
        auto transports = tt.flex_area_transports_[get<flex_area_idx_t>(s)];
        if (transports.empty() || transports.back() != idx) {
          transports.push_back(idx);
        }
      } else {
        auto transports =
            tt.location_group_transports_[get<location_group_idx_t>(s)];
        if (transports.empty() || transports.back() != idx) {
          transports.push_back(idx);
        }
      }
    }
  }
}

}  // namespace nigiri::loader::gtfs