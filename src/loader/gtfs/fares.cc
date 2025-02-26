#include "nigiri/loader/gtfs/fares.h"

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

media_map_t parse_media(std::string_view file_content,
                        string_cache_t& cache,
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
            {.name_ =
                 f.strings_.register_string(cache, r.fare_media_name_->view()),
             .type_ = static_cast<fares::fare_media::fare_media_type>(
                 *r.fare_media_type_)});
      });
  return m;
}

hash_map<std::string, fare_product_idx_t> parse_products(
    std::string_view file_content,
    string_cache_t& cache,
    fares& f,
    media_map_t const& media) {
  struct fare_product_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_product_id")> fare_product_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_product_name")> fare_product_name_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_media_id")> fare_media_id_;
    utl::csv_col<double, UTL_NAME("amount")> amount_;
    utl::csv_col<utl::cstr, UTL_NAME("currency")> currency_;
  };

  auto m = hash_map<std::string, fare_product_idx_t>{};
  for_each_row<fare_product_record>(
      file_content, [&](fare_product_record const& r) {
        m.emplace(r.fare_product_id_->view(), fare_product_idx_t{m.size()});
        f.fare_products_.push_back(
            {.name_ = f.strings_.register_string(cache,
                                                 r.fare_product_name_->view()),
             .media_ = find(media, r.fare_media_id_->view())
                           .value_or(fare_media_idx_t::invalid()),
             .amount_ = static_cast<std::uint32_t>(*r.amount_ / 100U),
             .currency_code_ =
                 f.strings_.register_string(cache, r.currency_->view())});
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
    utl::csv_col<std::optional<unsigned>, UTL_NAME("rule_priority")>
        rule_priority_;
  };

  auto m = hash_map<std::string, leg_group_idx_t>{};
  for_each_row<fare_leg_rule_record>(
      file_content, [&](fare_leg_rule_record const& r) {
        if (r.leg_group_id_->has_value()) {
          m.emplace((*r.leg_group_id_)->view(), leg_group_idx_t{m.size()});
        }
        f.fare_leg_rules_.push_back(
            {.leg_group_idx_ = r.leg_group_id_
                                   ->and_then([&](utl::cstr const& x) {
                                     return find(m, x.view());
                                   })
                                   .value_or(leg_group_idx_t::invalid()),
             .network_id_ = r.network_id_
                                ->and_then([&](utl::cstr const& x) {
                                  return find(networks, x.view());
                                })
                                .value_or(network_idx_t::invalid()),
             .from_area_id_ = r.from_area_id_
                                  ->and_then([&](utl::cstr const& x) {
                                    return find(areas, x.view());
                                  })
                                  .value_or(area_idx_t::invalid()),
             .to_area_id_ = r.to_area_id_
                                ->and_then([&](utl::cstr const& x) {
                                  return find(areas, x.view());
                                })
                                .value_or(area_idx_t::invalid()),
             .from_timeframe_group_id_ =
                 r.from_timeframe_group_id_
                     ->and_then([&](utl::cstr const& x) {
                       return find(timeframes, x.view());
                     })
                     .value_or(timeframe_group_idx_t::invalid()),
             .to_timeframe_group_id_ =
                 r.to_timeframe_group_id_
                     ->and_then([&](utl::cstr const& x) {
                       return find(timeframes, x.view());
                     })
                     .value_or(timeframe_group_idx_t::invalid()),
             .fare_product_id_ =
                 find(products, r.fare_product_id_->view()).value(),
             .rule_priority_ = r.rule_priority_->value_or(0U)});
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
      auto const [it, _] = m.emplace(r.timeframe_group_id_->view(),
                                     timeframe_group_idx_t{m.size()});
      f.timeframes_[it->second].push_back(fares::timeframe{
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

hash_map<std::string, area_idx_t> parse_areas(std::string_view file_content,
                                              string_cache_t& cache,
                                              fares& f) {
  struct area_record {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("area_name")> area_name_;
  };

  auto m = hash_map<std::string, area_idx_t>{};
  for_each_row<area_record>(file_content, [&](area_record const& r) {
    m.emplace(r.area_id_->view(), area_idx_t{m.size()});
    f.areas_.push_back(fares::area{
        .name_ = r.area_name_
                     ->and_then([&](utl::cstr const& x) {
                       return std::optional{
                           f.strings_.register_string(cache, x.view())};
                     })
                     .value_or(string_idx_t::invalid())});
  });
  return m;
}

hash_map<std::string, network_idx_t> parse_networks(
    std::string_view file_content, string_cache_t& cache, fares& f) {
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
                       return std::optional{
                           f.strings_.register_string(cache, x.view())};
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
    timetable const& tt,
    std::string_view file_content,
    fares& f,
    hash_map<std::string, area_idx_t> const& areas,
    locations_map const& stops) {
  struct stop_area_record {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
  };

  f.location_areas_.resize(tt.n_locations());

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

    f.location_areas_[*l_idx].push_back(*area_idx);
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
            fares::fare_leg_join_rule{.from_network_id_ = *from_network_idx,
                                      .to_network_id_ = *to_network_idx,
                                      .from_stop_id_ = from_stop_idx,
                                      .to_stop_id_ = to_stop_idx});
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

void parse_fares(timetable const& tt,
                 dir const& d,
                 traffic_days_t const& services,
                 route_map_t const& routes,
                 locations_map const& stops) {
  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  auto f = fares{};
  auto c = string_cache_t{std::size_t{0U}, string_idx_hash{f.strings_.strings_},
                          string_idx_equals{f.strings_.strings_}};
  auto const media = parse_media(load(kFareMediaFile).data(), c, f);
  auto const products =
      parse_products(load(kFareProductsFile).data(), c, f, media);
  auto const areas = parse_areas(load(kAreasFile).data(), c, f);
  auto const networks = parse_networks(load(kNetworksFile).data(), c, f);
  auto const route_networks = parse_route_networks(
      load(kRouteNetworksFile).data(), f, routes, networks);
  auto const timeframes =
      parse_timeframes(load(kTimeframesFile).data(), f, services);
  auto const leg_groups = parse_leg_rules(
      load(kFareLegRulesFile).data(), f, networks, areas, timeframes, products);
  auto const stop_areas =
      parse_stop_areas(tt, load(kStopAreasFile).data(), f, areas, stops);
  parse_fare_leg_join_rules(load(kFareLegJoinRulesFile).data(), f, networks,
                            stops);
  parse_fare_transfer_rules(load(kFareTransferRulesFile).data(), f, products,
                            leg_groups);
}

}  // namespace nigiri::loader::gtfs