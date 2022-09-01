#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/service/read_services.h"
#include "nigiri/loader/hrd/service/ref_service.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

struct service_builder {
  explicit service_builder(stamm const& s, timetable& tt)
      : stamm_{s}, tt_{tt} {}

  template <typename ProgressFn>
  void add_services(config const& c,
                    char const* filename,
                    std::string_view file_content,
                    ProgressFn&& bytes_consumed) {
    scoped_timer write{"reading services"};

    auto const get_index = [&](vector<ref_service> const& route_services,
                               ref_service const& s) -> std::optional<size_t> {
      auto const index = static_cast<unsigned>(std::distance(
          begin(route_services),
          std::lower_bound(begin(route_services), end(route_services), s,
                           [&](ref_service const& a, ref_service const& b) {
                             return a.stops(store_).front().dep_.time_ % 1440 <
                                    b.stops(store_).front().dep_.time_ % 1440;
                           })));

      for (auto i = 0U; i != s.utc_times_.size(); ++i) {
        auto const is_earlier_eq =
            index > 0 && s.utc_times_[i] % 1440 <
                             route_services[index - 1].utc_times_.at(i) % 1440;
        auto const is_later_eq =
            index < route_services.size() &&
            s.utc_times_[i] % 1440 >
                route_services[index].utc_times_.at(i) % 1440;
        if (is_earlier_eq || is_later_eq) {
          return std::nullopt;
        }
      }

      return index;
    };

    auto const add_service = [&](ref_service const& s) {
      auto const stop_seq =
          to_vec(s.stops(store_), [&](service::stop const& x) {
            return timetable::stop(stamm_.locations_.at(x.eva_num_).idx_,
                                   x.dep_.in_out_allowed_,
                                   x.arr_.in_out_allowed_);
          });
      auto const sections_clasz = to_vec(
          s.sections(store_),
          [](service::section const& section) { return section.clasz_; });

      auto& routes = route_services_[{stop_seq, sections_clasz}];
      for (auto& r : routes) {
        auto const idx = get_index(r, s);
        if (idx.has_value()) {
          r.insert(begin(r) + *idx, s);
          return;
        }
      }

      // No matching route found - create new one.
      routes.emplace_back(vector<ref_service>({s}));
    };

    parse_services(c, filename, tt_.date_range_, store_, stamm_.bitfields_,
                   stamm_.timezones_, file_content,
                   std::forward<ProgressFn>(bytes_consumed), add_service);
  }

  void write_services(source_idx_t const src) {
    scoped_timer write{"writing services"};

    hash_map<bitfield, bitfield_idx_t> bitfield_indices;
    for (auto const& [key, sub_routes] : route_services_) {
      for (auto const& services : sub_routes) {
        auto const& [stop_seq, sections_clasz] = key;
        auto const route_idx = tt_.register_route(stop_seq, sections_clasz);
        for (auto const& s : services) {
          auto const ref = store_.get(s.ref_);
          try {
            auto const stops = s.stops(store_);
            auto const sections = s.sections(store_);
            auto const id = tt_.register_trip_id(
                trip_id{
                    .id_ = fmt::format(
                        "{}/{}/{:07}/{}/{:07}/{}/{}", ref.initial_admin_.view(),
                        ref.initial_train_num_, to_idx(stops.front().eva_num_),
                        s.utc_times_.front().count(),
                        to_idx(stops.back().eva_num_),
                        s.utc_times_.back().count(),
                        sections.front().line_information_.empty()
                            ? ""
                            : sections.front()
                                  .line_information_.front()
                                  .view()),
                    .src_ = src},
                ref.display_name(tt_, stamm_.categories_, stamm_.providers_),
                ref.origin_.str(), tt_.next_transport_idx(),
                {0U, stop_seq.size()});

            auto const section_attributes =
                // Warning! This currently ignores the traffic days.
                // TODO(felix) consider traffic day bitfields of attributes:
                // - watch out: attribute bitfields probably reference local
                // time!
                // - attributes that are relevant for routing: split service
                // - not routing relevant attributes: store in database
                to_vec(sections, [&](service::section const& sec) {
                  auto attribute_idx_combination = to_vec(
                      sec.attributes_, [&](service::attribute const& attr) {
                        return stamm_.attributes_.at(attr.code_.view());
                      });

                  return utl::get_or_create(
                      attribute_combinations_, attribute_idx_combination,
                      [&]() {
                        auto const combination_idx =
                            attribute_combination_idx_t{
                                tt_.attribute_combinations_.size()};
                        tt_.attribute_combinations_.emplace_back(
                            std::move(attribute_idx_combination));
                        return combination_idx;
                      });
                });

            auto const section_providers =
                to_vec(sections, [&](service::section const& sec) {
                  return stamm_.providers_.get(sec.admin_.view())
                      .value_or(provider_idx_t::invalid());
                });

            auto const section_directions =
                to_vec(sections, [&](service::section const& sec) {
                  if (sec.directions_.empty()) {
                    return trip_direction_idx_t::invalid();
                  }
                  return sec.directions_[0].apply(utl::overloaded{
                      [&](utl::cstr const& str) {
                        return utl::get_or_create(
                            string_directions_, str.view(), [&]() {
                              auto const it =
                                  stamm_.directions_.find(str.view());
                              if (it == end(stamm_.directions_)) {
                                return trip_direction_idx_t::invalid();
                              } else {
                                auto const dir_idx = trip_direction_idx_t{
                                    tt_.trip_directions_.size()};
                                tt_.trip_directions_.emplace_back(it->second);
                                return dir_idx;
                              }
                            });
                      },
                      [&](eva_number const eva) {
                        return utl::get_or_create(eva_directions_, eva, [&]() {
                          auto const it = stamm_.locations_.find(eva);
                          if (it == end(stamm_.locations_)) {
                            return trip_direction_idx_t::invalid();
                          } else {
                            auto const dir_idx = trip_direction_idx_t{
                                tt_.trip_directions_.size()};
                            tt_.trip_directions_.emplace_back(it->second.idx_);
                            return dir_idx;
                          }
                        });
                      }});
                });

            auto const section_lines =
                to_vec(sections, [&](service::section const& sec) {
                  if (sec.line_information_.empty()) {
                    return line_idx_t::invalid();
                  }
                  return utl::get_or_create(
                      lines_, sec.line_information_[0].view(), [&]() {
                        auto const idx = line_idx_t{tt_.lines_.size()};
                        tt_.lines_.emplace_back(
                            sec.line_information_[0].view());
                        return idx;
                      });
                });

            auto const merged_trip = tt_.register_merged_trip({id});
            tt_.add_transport(timetable::transport{
                .bitfield_idx_ = utl::get_or_create(
                    bitfield_indices, s.utc_traffic_days_,
                    [&]() {
                      return tt_.register_bitfield(s.utc_traffic_days_);
                    }),
                .route_idx_ = route_idx,
                .stop_times_ = s.utc_times_,
                .meta_data_ = vector<section_db_idx_t>(stop_seq.size() - 1),
                .external_trip_ids_ = vector<merged_trips_idx_t>(
                    stop_seq.size() - 1U, merged_trip),
                .section_attributes_ = section_attributes,
                .section_providers_ = section_providers,
                .section_directions_ = section_directions,
                .section_lines_ = section_lines});
          } catch (std::exception const& e) {
            log(log_lvl::error, "loader.hrd.service",
                "unable to load service {}: {}", ref.origin_, e.what());
            continue;
          }
        }
        tt_.finish_route();
      }
    }
    route_services_.clear();
    store_.clear();
  }

  stamm const& stamm_;
  timetable& tt_;
  hash_map<pair<vector<timetable::stop>, vector<clasz>>,
           vector<vector<ref_service>>>
      route_services_;
  service_store store_;
  hash_map<vector<attribute_idx_t>, attribute_combination_idx_t>
      attribute_combinations_;
  hash_map<string, line_idx_t> lines_;
  hash_map<string, trip_direction_idx_t> string_directions_;
  hash_map<eva_number, trip_direction_idx_t> eva_directions_;
};

}  // namespace nigiri::loader::hrd
