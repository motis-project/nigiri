#include "nigiri/loader/hrd/service/service_builder.h"

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"

namespace nigiri::loader::hrd {

void service_builder::add_services(config const& c,
                                   const char* filename,
                                   std::string_view file_content,
                                   progress_update_fn const& progress_update) {
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
    auto const stop_seq = to_vec(s.stops(store_), [&](service::stop const& x) {
      return timetable::stop(stamm_.resolve_location(x.eva_num_),
                             x.dep_.in_out_allowed_, x.arr_.in_out_allowed_);
    });
    auto const& ref = store_.get(s.ref_);
    auto const begin_to_end_cat = ref.begin_to_end_info_.category_;
    auto const sections_clasz =
        begin_to_end_cat.has_value()
            ? vector<clasz>{begin_to_end_cat.value() == nullptr
                                ? clasz::kOther
                                : begin_to_end_cat.value()->clasz_}
            : to_vec(s.sections(store_), [&](service::section const& sec) {
                assert(sec.category_.has_value());
                return sec.category_.value() == nullptr
                           ? clasz::kOther
                           : sec.category_.value()->clasz_;
              });

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

  auto const source_file_idx = tt_.register_source_file(filename);
  parse_services(c, filename, source_file_idx, tt_.date_range_, store_, stamm_,
                 file_content, progress_update, add_service);
}

void service_builder::write_services(const nigiri::source_idx_t src) {
  scoped_timer write{"writing services"};

  for (auto const& [key, sub_routes] : route_services_) {
    for (auto const& services : sub_routes) {
      auto const& [stop_seq, sections_clasz] = key;
      auto const route_idx = tt_.register_route(stop_seq, sections_clasz);
      for (auto const& s : services) {
        auto const ref = store_.get(s.ref_);
        try {
          auto const stops = s.stops(store_);
          auto const id = tt_.register_trip_id(
              trip_id{.id_ = fmt::format(
                          "{}/{:07}/{}/{:07}/{}/{}", ref.initial_train_num_,
                          to_idx(stops.front().eva_num_),
                          s.utc_times_.front().count(),
                          to_idx(stops.back().eva_num_),
                          s.utc_times_.back().count(), s.line_info(store_)),
                      .src_ = src},
              ref.display_name(tt_), ref.origin_.dbg_, tt_.next_transport_idx(),
              {0U, stop_seq.size()});

          vector<attribute_combination_idx_t> section_attributes{
              attribute_combination_idx_t::invalid()};
          auto const get_attribute_combination_idx =
              [&](std::vector<service::attribute> const& a,
                  std::vector<service::attribute> const& b) {
                vector<attribute_idx_t> combination;
                combination.resize(a.size() + b.size());

                auto i = 0;
                for (auto const& attr : a) {
                  combination[i++] = attr.code_;
                }
                for (auto const& attr : b) {
                  combination[i++] = attr.code_;
                }
                utl::erase_duplicates(combination);

                return utl::get_or_create(
                    attribute_combinations_, combination, [&]() {
                      auto const combination_idx = attribute_combination_idx_t{
                          tt_.attribute_combinations_.size()};
                      tt_.attribute_combinations_.emplace_back(
                          std::move(combination));
                      return combination_idx;
                    });
              };
          auto const has_no_attributes = [](service::section const& sec) {
            return !sec.attributes_.has_value();
          };
          if (utl::all_of(ref.sections_, has_no_attributes)) {
            if (ref.begin_to_end_info_.attributes_.has_value()) {
              section_attributes = {get_attribute_combination_idx(
                  ref.begin_to_end_info_.attributes_.value(), {})};
            }
          } else {
            section_attributes =
                to_vec(s.sections(store_), [&](service::section const& sec) {
                  return get_attribute_combination_idx(
                      ref.begin_to_end_info_.attributes_.value_or(
                          std::vector<service::attribute>{}),
                      sec.attributes_.value_or(
                          std::vector<service::attribute>{}));
                });
          }

          auto section_providers = vector<provider_idx_t>{};
          if (ref.begin_to_end_info_.admin_.has_value()) {
            section_providers = {ref.begin_to_end_info_.admin_.value()};
          } else {
            section_providers =
                to_vec(s.sections(store_), [&](service::section const& sec) {
                  return sec.admin_.value();
                });
          }

          auto section_directions = vector<trip_direction_idx_t>{};
          if (ref.begin_to_end_info_.direction_.has_value()) {
            section_directions = {ref.begin_to_end_info_.direction_.value()};
          } else if (utl::any_of(s.sections(store_),
                                 [](service::section const& sec) {
                                   return sec.direction_.has_value();
                                 })) {
            section_directions =
                to_vec(s.sections(store_), [&](service::section const& sec) {
                  return sec.direction_.value_or(
                      trip_direction_idx_t::invalid());
                });
          }

          auto const merged_trip = tt_.register_merged_trip({id});
          tt_.add_transport(timetable::transport{
              .bitfield_idx_ = utl::get_or_create(
                  bitfield_indices_, s.utc_traffic_days_,
                  [&]() { return tt_.register_bitfield(s.utc_traffic_days_); }),
              .route_idx_ = route_idx,
              .stop_times_ = s.utc_times_,
              .external_trip_ids_ = {merged_trip},
              .section_attributes_ = section_attributes,
              .section_providers_ = section_providers,
              .section_directions_ = section_directions});
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

}  // namespace nigiri::loader::hrd