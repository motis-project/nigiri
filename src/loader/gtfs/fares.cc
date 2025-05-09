#include "nigiri/loader/gtfs/fares.h"

#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"

#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

using namespace std::string_view_literals;

namespace nigiri::loader::gtfs {

constexpr auto const kFareMediaFile = "fare_media.txt"sv;
constexpr auto const kFareProductsFile = "fare_products.txt"sv;
constexpr auto const kFareLegRulesFile = "fare_leg_rules.txt"sv;
constexpr auto const kTimeframesFile = "timeframes.txt"sv;
constexpr auto const kStopAreasFile = "stop_areas.txt"sv;
constexpr auto const kRouteNetworksFile = "route_networks.txt"sv;
constexpr auto const kAreasFile = "areas.txt"sv;
constexpr auto const kNetworksFile = "networks.txt"sv;
constexpr auto const kFareLegJoinRulesFile = "fare_leg_join_rules.txt"sv;
constexpr auto const kFareTransferRulesFile = "fare_transfer_rules.txt"sv;
constexpr auto const kRiderCategoriesFile = "rider_categories.txt"sv;

using media_map_t = hash_map<std::string, fare_media_idx_t>;

template <typename T, typename Key>
auto find(T const& x, Key const& k) {
  auto const it = x.find(k);
  return it != end(x) ? std::optional{it->second} : std::nullopt;
}

template <typename T, typename Fn>
void for_each_row(std::string_view file_content, Fn&& fn) {
  utl::line_range{utl::make_buf_reader(file_content)}  //
      | utl::csv<T>()  //
      | utl::for_each(std::forward<Fn>(fn));
}

media_map_t parse_media(timetable& tt,
                        std::string_view file_content,
                        fares& f) {
  struct fare_media_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_media_id")> fare_media_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_media_name")> fare_media_name_;
    utl::csv_col<unsigned, UTL_NAME("fare_media_type")> fare_media_type_;
  };

  auto m = media_map_t{};
  for_each_row<fare_media_record>(
      file_content, [&](fare_media_record const& r) {
        m.emplace(r.fare_media_id_->view(), fare_media_idx_t{m.size()});
        f.fare_media_.push_back(
            {.name_ = tt.strings_.store(r.fare_media_name_->view()),
             .type_ = static_cast<fares::fare_media::fare_media_type>(
                 *r.fare_media_type_)});
      });
  return m;
}

hash_map<std::string, fare_product_idx_t> parse_products(
    timetable& tt,
    std::string_view file_content,
    fares& f,
    media_map_t const& media,
    hash_map<std::string, rider_category_idx_t> const& rider_categories) {
  struct fare_product_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_product_id")> fare_product_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_product_name")> fare_product_name_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_media_id")> fare_media_id_;
    utl::csv_col<float, UTL_NAME("amount")> amount_;
    utl::csv_col<utl::cstr, UTL_NAME("currency")> currency_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("rider_category_id")>
        rider_category_id_;
  };

  auto m = hash_map<std::string, fare_product_idx_t>{};
  for_each_row<fare_product_record>(
      file_content, [&](fare_product_record const& r) {
        m.emplace(r.fare_product_id_->view(), fare_product_idx_t{m.size()});
        f.fare_products_.push_back(
            {.amount_ = *r.amount_,
             .name_ = tt.strings_.store(r.fare_product_name_->view()),
             .media_ = find(media, r.fare_media_id_->view())
                           .value_or(fare_media_idx_t::invalid()),
             .currency_code_ = tt.strings_.store(r.currency_->view()),
             .rider_category_ =
                 r.rider_category_id_
                     ->and_then([&](utl::cstr const& x) {
                       return find(rider_categories, x.view());
                     })
                     .value_or(rider_category_idx_t::invalid())});
      });
  return m;
}

hash_map<std::string, leg_group_idx_t> parse_leg_rules(
    std::string_view file_content,
    fares& f,
    hash_map<std::string, network_idx_t> const& networks,
    hash_map<std::string, area_idx_t> const& areas,
    hash_map<std::string, timeframe_group_idx_t> const& timeframes,
    hash_map<std::string, fare_product_idx_t> const& products) {
  struct fare_leg_rule_record {
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("leg_group_id")>
        leg_group_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("network_id")> network_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("from_area_id")>
        from_area_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("to_area_id")> to_area_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("from_timeframe_group_id")>
        from_timeframe_group_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("to_timeframe_group_id")>
        to_timeframe_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_product_id")> fare_product_id_;
    utl::csv_col<std::optional<int>, UTL_NAME("rule_priority")> rule_priority_;
  };

  auto m = hash_map<std::string, leg_group_idx_t>{};
  for_each_row<fare_leg_rule_record>(
      file_content, [&](fare_leg_rule_record const& r) {
        if (r.leg_group_id_->has_value()) {
          m.emplace((*r.leg_group_id_)->view(), leg_group_idx_t{m.size()});
        }
        auto const fare_product = find(products, r.fare_product_id_->view());
        if (!fare_product.has_value()) {
          log(log_lvl::error, "gtfs.fares", "leg_rules: product {} not found",
              r.fare_product_id_->view());
          return;
        }
        f.fare_leg_rules_.push_back({
            .rule_priority_ = r.rule_priority_->value_or(0U),
            .network_ = r.network_id_
                            ->and_then([&](utl::cstr const& x) {
                              return find(networks, x.view());
                            })
                            .value_or(network_idx_t::invalid()),
            .from_area_ = r.from_area_id_
                              ->and_then([&](utl::cstr const& x) {
                                return find(areas, x.view());
                              })
                              .value_or(area_idx_t::invalid()),
            .to_area_ = r.to_area_id_
                            ->and_then([&](utl::cstr const& x) {
                              return find(areas, x.view());
                            })
                            .value_or(area_idx_t::invalid()),
            .from_timeframe_group_ =
                r.from_timeframe_group_id_
                    ->and_then([&](utl::cstr const& x) {
                      return find(timeframes, x.view());
                    })
                    .value_or(timeframe_group_idx_t::invalid()),
            .to_timeframe_group_ =
                r.to_timeframe_group_id_
                    ->and_then([&](utl::cstr const& x) {
                      return find(timeframes, x.view());
                    })
                    .value_or(timeframe_group_idx_t::invalid()),
            .fare_product_ = *fare_product,
            .leg_group_idx_ = r.leg_group_id_
                                  ->and_then([&](utl::cstr const& x) {
                                    return find(m, x.view());
                                  })
                                  .value_or(leg_group_idx_t::invalid()),
        });
      });
  return m;
}

hash_map<std::string, timeframe_group_idx_t> parse_timeframes(
    std::string_view file_content,
    fares& f,
    hash_map<std::string, std::unique_ptr<bitfield>> const& services) {
  struct timeframe_record {
    utl::csv_col<utl::cstr, UTL_NAME("timeframe_group_id")> timeframe_group_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("start_time")> start_time_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("end_time")> end_time_;
    utl::csv_col<utl::cstr, UTL_NAME("service_id")> service_id_;
  };

  auto m = hash_map<std::string, timeframe_group_idx_t>{};
  for_each_row<timeframe_record>(file_content, [&](timeframe_record const& r) {
    try {
      auto const traffic_days = *services.at(r.service_id_->view());
      auto const i =
          utl::get_or_create(m, r.timeframe_group_id_->view(), [&]() {
            auto const idx = timeframe_group_idx_t{f.timeframes_.size()};
            f.timeframes_.add_back_sized(0U);
            return idx;
          });
      f.timeframes_[i].push_back(fares::timeframe{
          .start_time_ = r.start_time_
                             ->and_then([](utl::cstr x) {
                               return std::optional{hhmm_to_min(x)};
                             })
                             .value_or(0_hours),
          .end_time_ = r.end_time_
                           ->and_then([](utl::cstr x) {
                             return std::optional{hhmm_to_min(x)};
                           })
                           .value_or(24_hours),
          .service_ = traffic_days});
    } catch (...) {
      log(log_lvl::error, "gtfs.fares", "timeframes: service {} not found",
          r.service_id_->view());
    }
  });
  return m;
}

hash_map<std::string, area_idx_t> parse_areas(timetable& tt,
                                              std::string_view file_content) {
  struct area_record {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("area_name")> area_name_;
  };

  auto m = hash_map<std::string, area_idx_t>{};
  for_each_row<area_record>(file_content, [&](area_record const& r) {
    m.emplace(r.area_id_->view(), area_idx_t{m.size()});
    tt.areas_.push_back(area{.name_ = r.area_name_
                                          ->and_then([&](utl::cstr const& x) {
                                            return std::optional{
                                                tt.strings_.store(x.view())};
                                          })
                                          .value_or(string_idx_t::invalid())});
  });
  return m;
}

hash_map<std::string, network_idx_t> parse_networks(
    timetable& tt, std::string_view file_content, fares& f) {
  struct network_record {
    utl::csv_col<utl::cstr, UTL_NAME("network_id")> network_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("network_name")>
        network_name_;
  };

  auto m = hash_map<std::string, network_idx_t>{};
  for_each_row<network_record>(file_content, [&](network_record const& r) {
    m.emplace(r.network_id_->view(), network_idx_t{m.size()});
    f.networks_.push_back(fares::network{
        .name_ = r.network_name_
                     ->and_then([&](utl::cstr const& x) {
                       return std::optional{tt.strings_.store(x.view())};
                     })
                     .value_or(string_idx_t::invalid())});
  });
  return m;
}

hash_map<std::string, route_idx_t> parse_route_networks(
    std::string_view file_content,
    fares& f,
    route_map_t const& routes,
    hash_map<std::string, network_idx_t> const& networks) {
  struct route_network_record {
    utl::csv_col<utl::cstr, UTL_NAME("network_id")> network_id_;
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
  };

  auto m = hash_map<std::string, route_idx_t>{};
  for_each_row<route_network_record>(
      file_content, [&](route_network_record const& r) {
        auto const network_idx = find(networks, r.network_id_->view());
        if (!network_idx.has_value()) {
          log(log_lvl::error, "gtfs.fares",
              "route_networks: network {} not found", r.network_id_->view());
          return;
        }

        auto const route_it = routes.find(r.route_id_->view());
        if (route_it == end(routes)) {
          log(log_lvl::error, "gtfs.fares",
              "route_networks: route {} not found", r.route_id_->view());
          return;
        }

        f.route_networks_.emplace(route_it->second->route_id_idx_,
                                  *network_idx);
      });
  return m;
}

hash_map<std::string, location_idx_t> parse_stop_areas(
    timetable& tt,
    std::string_view file_content,
    hash_map<std::string, area_idx_t> const& areas,
    locations_map const& stops) {
  struct stop_area_record {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
  };

  tt.location_areas_.resize(tt.n_locations());

  auto m = hash_map<std::string, location_idx_t>{};
  for_each_row<stop_area_record>(file_content, [&](stop_area_record const& r) {
    auto const l_idx = find(stops, r.stop_id_->view());
    if (!l_idx.has_value()) {
      log(log_lvl::error, "gtfs.fares", "stop_areas: stop {} not found",
          r.stop_id_->view());
      return;
    }

    auto const area_idx = find(areas, r.area_id_->view());
    if (!area_idx.has_value()) {
      log(log_lvl::error, "gtfs.fares", "stop_areas: area {} not found",
          r.area_id_->view());
      return;
    }

    tt.location_areas_[*l_idx].push_back(*area_idx);
  });
  return m;
}

void parse_fare_leg_join_rules(
    std::string_view file_content,
    fares& f,
    hash_map<std::string, network_idx_t> const& networks,
    locations_map const& stops) {
  struct fare_leg_join_rule_record {
    utl::csv_col<utl::cstr, UTL_NAME("from_network_id")> from_network_id_;
    utl::csv_col<utl::cstr, UTL_NAME("to_network_id")> to_network_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("from_stop_id")>
        from_stop_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("to_stop_id")> to_stop_id_;
  };

  for_each_row<fare_leg_join_rule_record>(
      file_content, [&](fare_leg_join_rule_record const& r) {
        auto const from_network_idx =
            find(networks, r.from_network_id_->view());
        if (!from_network_idx.has_value()) {
          log(log_lvl::error, "gtfs.fares",
              "fare_leg_join_rules: network '{}' not found",
              r.from_network_id_->view());
          return;
        }

        auto const to_network_idx = find(networks, r.to_network_id_->view());
        if (!to_network_idx.has_value()) {
          log(log_lvl::error, "gtfs.fares",
              "fare_leg_join_rules: network '{}' not found",
              r.to_network_id_->view());
          return;
        }

        auto const from_stop_idx = r.from_stop_id_
                                       ->and_then([&](utl::cstr const& x) {
                                         return find(stops, x.view());
                                       })
                                       .value_or(location_idx_t::invalid());
        auto const to_stop_idx = r.to_stop_id_
                                     ->and_then([&](utl::cstr const& x) {
                                       return find(stops, x.view());
                                     })
                                     .value_or(location_idx_t::invalid());

        f.fare_leg_join_rules_.push_back(
            fares::fare_leg_join_rule{.from_network_ = *from_network_idx,
                                      .to_network_ = *to_network_idx,
                                      .from_stop_ = from_stop_idx,
                                      .to_stop_ = to_stop_idx});
      });
}

void parse_fare_transfer_rules(
    std::string_view file_content,
    fares& f,
    hash_map<std::string, fare_product_idx_t> const& products,
    hash_map<std::string, leg_group_idx_t> const& leg_groups) {
  struct fare_transfer_rule_record {
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("from_leg_group_id")>
        from_leg_group_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("to_leg_group_id")>
        to_leg_group_id_;
    utl::csv_col<std::optional<int>, UTL_NAME("transfer_count")>
        transfer_count_;
    utl::csv_col<std::optional<int>, UTL_NAME("duration_limit")>
        duration_limit_;
    utl::csv_col<std::optional<int>, UTL_NAME("duration_limit_type")>
        duration_limit_type_;
    utl::csv_col<unsigned, UTL_NAME("fare_transfer_type")> fare_transfer_type_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("fare_product_id")>
        fare_product_id_;
  };

  for_each_row<fare_transfer_rule_record>(
      file_content, [&](fare_transfer_rule_record const& r) {
        if (r.transfer_count_->value_or(-1) == 0) {
          log(log_lvl::error, "gtfs.fares",
              "fare transfer rule with 0 transfers not allowed");
          return;
        }
        f.fare_transfer_rules_.push_back(fares::fare_transfer_rule{
            .from_leg_group_ = r.from_leg_group_id_
                                   ->and_then([&](utl::cstr const& x) {
                                     return find(leg_groups, x.view());
                                   })
                                   .value_or(leg_group_idx_t::invalid()),
            .to_leg_group_ = r.to_leg_group_id_
                                 ->and_then([&](utl::cstr const& x) {
                                   return find(leg_groups, x.view());
                                 })
                                 .value_or(leg_group_idx_t::invalid()),
            .transfer_count_ =
                static_cast<std::int8_t>(r.transfer_count_->value_or(-1)),
            .duration_limit_ =
                r.duration_limit_
                    ->and_then(
                        [](int i) { return std::optional{duration_t{i}}; })
                    .value_or(fares::fare_transfer_rule::kNoDurationLimit),
            .duration_limit_type_ =
                static_cast<fares::fare_transfer_rule::duration_limit_type>(
                    r.duration_limit_type_->value_or(0U)),
            .fare_transfer_type_ =
                static_cast<fares::fare_transfer_rule::fare_transfer_type>(
                    *r.fare_transfer_type_),
            .fare_product_ = r.fare_product_id_
                                 ->and_then([&](utl::cstr const& x) {
                                   return find(products, x.view());
                                 })
                                 .value_or(fare_product_idx_t::invalid())});
      });
}

hash_map<std::string, rider_category_idx_t> parse_rider_categories(
    std::string_view file_content, fares& f, timetable& tt) {
  struct rider_category_record {
    utl::csv_col<utl::cstr, UTL_NAME("rider_category_id")> rider_category_id_;
    utl::csv_col<utl::cstr, UTL_NAME("rider_category_name")>
        rider_category_name_;
    utl::csv_col<int, UTL_NAME("is_default_fare_category")>
        is_default_fare_category_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("eligibility_url")>
        eligibility_url_;
  };

  auto m = hash_map<std::string, rider_category_idx_t>{};
  for_each_row<rider_category_record>(
      file_content, [&](rider_category_record const& r) {
        m.emplace(r.rider_category_id_->view(), rider_category_idx_t{m.size()});
        f.rider_categories_.push_back(fares::rider_category{
            .name_ = tt.strings_.store(r.rider_category_name_->view()),
            .eligibility_url_ = r.eligibility_url_
                                    ->and_then([&](utl::cstr const& x) {
                                      return std::optional{
                                          tt.strings_.store(x.view())};
                                    })
                                    .value_or(string_idx_t::invalid()),
            .is_default_fare_category_ = *r.is_default_fare_category_ == 1});
      });
  return m;
}

void load_fares(timetable& tt,
                dir const& d,
                traffic_days_t const& services,
                route_map_t const& routes,
                locations_map const& stops) {
  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  auto& f = tt.fares_.emplace_back();
  auto const media = parse_media(tt, load(kFareMediaFile).data(), f);
  auto const rider_categories =
      parse_rider_categories(load(kRiderCategoriesFile).data(), f, tt);
  auto const products = parse_products(tt, load(kFareProductsFile).data(), f,
                                       media, rider_categories);
  auto const areas = parse_areas(tt, load(kAreasFile).data());
  auto const networks = parse_networks(tt, load(kNetworksFile).data(), f);
  auto const route_networks = parse_route_networks(
      load(kRouteNetworksFile).data(), f, routes, networks);
  auto const timeframes =
      parse_timeframes(load(kTimeframesFile).data(), f, services);
  auto const leg_groups = parse_leg_rules(
      load(kFareLegRulesFile).data(), f, networks, areas, timeframes, products);
  auto const stop_areas =
      parse_stop_areas(tt, load(kStopAreasFile).data(), areas, stops);
  parse_fare_leg_join_rules(load(kFareLegJoinRulesFile).data(), f, networks,
                            stops);
  parse_fare_transfer_rules(load(kFareTransferRulesFile).data(), f, products,
                            leg_groups);

  utl::sort(f.fare_leg_rules_,
            [](fares::fare_leg_rule const& a, fares::fare_leg_rule const& b) {
              return a.rule_priority_ < b.rule_priority_;
            });
  utl::sort(f.fare_leg_join_rules_);
}

}  // namespace nigiri::loader::gtfs