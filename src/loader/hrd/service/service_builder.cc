#include "nigiri/loader/hrd/service/service_builder.h"

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/get_index.h"
#include "nigiri/loader/hrd/service/read_services.h"

namespace nigiri::loader::hrd {

void service_builder::add_service(ref_service&& s) {
  route_key_.first = std::move(s.stop_seq_);

  auto const begin_to_end_cat = store_.get(s.ref_).begin_to_end_info_.category_;
  if (begin_to_end_cat.has_value()) {
    route_key_.second.resize(1U);
    route_key_.second[0] = begin_to_end_cat.value() == nullptr
                               ? clasz::kOther
                               : begin_to_end_cat.value()->clasz_;
  } else {
    route_key_.second.clear();
    utl::transform_to(s.sections(store_), route_key_.second,
                      [&](service::section const& sec) {
                        assert(sec.category_.has_value());
                        return sec.category_.value() == nullptr
                                   ? clasz::kOther
                                   : sec.category_.value()->clasz_;
                      });
  }

  if (auto const it = route_services_.find(route_key_);
      it != end(route_services_)) {
    for (auto& r : it->second) {
      auto const idx = get_index(r, s);
      if (idx.has_value()) {
        r.insert(begin(r) + *idx, s);
        return;
      }
    }
    it->second.emplace_back(vector<ref_service>{std::move(s)});
  } else {
    route_services_.emplace(route_key_,
                            vector<vector<ref_service>>{{std::move(s)}});
  }
}

service_builder::service_builder(stamm& s, timetable& tt)
    : stamm_{s}, tt_{tt} {}

void service_builder::add_services(config const& c,
                                   const char* filename,
                                   std::string_view file_content,
                                   progress_update_fn const& progress_update) {
  auto const timer = scoped_timer{"loader.hrd.services.read"};
  auto const source_file_idx = tt_.register_source_file(filename);
  parse_services(c, filename, source_file_idx, stamm_.get_date_range(),
                 tt_.date_range_, store_, stamm_, file_content, progress_update,
                 [&](ref_service&& s) { add_service(std::move(s)); });
}

void service_builder::write_services(source_idx_t const src) {
  auto const timer = scoped_timer{"loader.hrd.services.write"};
  for (auto const& [key, sub_routes] : route_services_) {
    for (auto const& services : sub_routes) {
      auto const& [stop_seq, sections_clasz] = key;
      auto const route_idx = tt_.register_route(stop_seq, sections_clasz);

      for (auto const& s : stop_seq) {
        auto s_routes = location_routes_[stop{s}.location_idx()];
        if (s_routes.empty() || s_routes.back() != route_idx) {
          s_routes.emplace_back(route_idx);
        }
      }

      for (auto const& s : services) {
        auto const& ref = store_.get(s.ref_);
        try {
          auto const stops = s.stops(store_);

          trip_id_buf_.clear();
          fmt::format_to(trip_id_buf_, "{}/{:07}/{}/{:07}/{}/{}",
                         ref.initial_train_num_, to_idx(stops.front().eva_num_),
                         s.utc_times_.front().count(),
                         to_idx(stops.back().eva_num_),
                         s.utc_times_.back().count(), s.line_info(store_));

          auto const id = tt_.register_trip_id(
              trip_id_buf_, src, ref.display_name(tt_), ref.origin_.dbg_,
              ref.initial_train_num_, {});
          tt_.trip_transport_ranges_.emplace_back({transport_range_t{
              tt_.next_transport_idx(),
              interval<stop_idx_t>{0U,
                                   static_cast<stop_idx_t>(stop_seq.size())}}});

          auto const get_attribute_combination_idx =
              [&](std::optional<std::vector<service::attribute>> const& a,
                  std::optional<std::vector<service::attribute>> const& b) {
                attribute_combination_.resize((a.has_value() ? a->size() : 0U) +
                                              (b.has_value() ? b->size() : 0U));

                auto i = 0U;
                if (a.has_value()) {
                  for (auto const& attr : a.value()) {
                    attribute_combination_[i++] = attr.idx_;
                  }
                }
                if (b.has_value()) {
                  for (auto const& attr : b.value()) {
                    attribute_combination_[i++] = attr.idx_;
                  }
                }
                utl::erase_duplicates(attribute_combination_);

                return utl::get_or_create(
                    attribute_combinations_, attribute_combination_, [&]() {
                      auto const combination_idx = attribute_combination_idx_t{
                          tt_.attribute_combinations_.size()};
                      tt_.attribute_combinations_.emplace_back(
                          attribute_combination_);
                      return combination_idx;
                    });
              };
          if (!ref.sections_.empty() &&
              utl::all_of(ref.sections_, [](service::section const& sec) {
                return !sec.attributes_.has_value();
              })) {
            if (ref.begin_to_end_info_.attributes_.has_value()) {
              section_attributes_.resize(1U);
              section_attributes_[0] = get_attribute_combination_idx(
                  ref.begin_to_end_info_.attributes_, std::nullopt);
            } else {
              section_attributes_.clear();
            }
          } else if (!ref.sections_.empty()) {
            section_attributes_.clear();
            utl::transform_to(s.sections(store_), section_attributes_,
                              [&](service::section const& sec) {
                                return get_attribute_combination_idx(
                                    ref.begin_to_end_info_.attributes_,
                                    sec.attributes_);
                              });
          } else {
            section_attributes_.clear();
          }

          if (ref.begin_to_end_info_.admin_.has_value()) {
            section_providers_.resize(1U);
            section_providers_[0] = {ref.begin_to_end_info_.admin_.value()};
          } else if (!ref.sections_.empty()) {
            section_providers_.clear();
            utl::transform_to(s.sections(store_), section_providers_,
                              [&](service::section const& sec) {
                                return sec.admin_.value();
                              });
          } else {
            section_providers_.clear();
          }

          if (ref.begin_to_end_info_.direction_.has_value()) {
            section_directions_.resize(1U);
            section_directions_ = ref.begin_to_end_info_.direction_.value();
          } else if (!ref.sections_.empty() &&
                     utl::any_of(s.sections(store_),
                                 [](service::section const& sec) {
                                   return sec.direction_.has_value();
                                 })) {
            section_directions_.clear();
            utl::transform_to(s.sections(store_), section_directions_,
                              [&](service::section const& sec) {
                                return sec.direction_.value_or(
                                    trip_direction_idx_t::invalid());
                              });
          } else {
            section_directions_.clear();
          }

          if (ref.begin_to_end_info_.line_.has_value()) {
            section_lines_.resize(1U);
            section_lines_ = stamm_.resolve_line(
                ref.begin_to_end_info_.line_.value().view());
          } else if (!ref.sections_.empty() &&
                     utl::any_of(s.sections(store_),
                                 [&](service::section const& sec) {
                                   return sec.line_.has_value();
                                 })) {
            section_lines_.clear();
            utl::transform_to(
                s.sections(store_), section_lines_,
                [&](service::section const& sec) {
                  return sec.line_.has_value()
                             ? stamm_.resolve_line(sec.line_.value().view())
                             : trip_line_idx_t::invalid();
                });
          } else {
            section_lines_.clear();
          }

          auto const merged_trip = tt_.register_merged_trip({id});
          tt_.add_transport(timetable::transport{
              .bitfield_idx_ = utl::get_or_create(
                  bitfield_indices_, s.utc_traffic_days_,
                  [&]() { return tt_.register_bitfield(s.utc_traffic_days_); }),
              .route_idx_ = route_idx,
              .first_dep_offset_ = 0_minutes,
              .external_trip_ids_ = {merged_trip},
              .section_attributes_ = section_attributes_,
              .section_providers_ = section_providers_,
              .section_directions_ = section_directions_,
              .section_lines_ = section_lines_,
              .stop_seq_numbers_ = stop_seq_numbers_});
        } catch (std::exception const& e) {
          log(log_lvl::error, "loader.hrd.service",
              "unable to load service {}: {}", ref.origin_, e.what());
          continue;
        }
      }

      tt_.finish_route();

      // Event times are stored alternatingly in utc_times_:
      // departure (D), arrival (A), ..., arrival (A)
      // event type: D A D A D A D A
      // stop index: 0 1 1 2 2 3 3 4
      // event time: 0 1 2 3 4 5 6 7
      // --> A at stop i = i x 2 - 1
      // --> D at stop i = i x 2
      // Note: no arrival at the first stop and no departure at the last stop.
      //
      // Transform route from:
      //   [[D(0, 0), A(0, 1), D(0, 1), A(0, 2), ...],
      //    [D(1, 0), A(1, 1), D(1, 1), A(1, 2), ...],
      //    ... ]
      // to:
      //   [D(0, 0), D(1, 0), D(2, 0), ..., D(N, 0),
      //    A(0, 1), A(1, 1), A(2, 1), ..., A(N, 1),
      //    D(0, 1), D(1, 1), D(2, 1), ..., D(N, 1),
      //    A(0, 2), A(1, 2), A(2, 2), ..., A(N, 2),
      //    ...]
      //
      // Where D(x, y) is departure of transport x at stop index y in the route
      // location sequence and A(x, y) is the arrival.

      auto const stop_times_begin = tt_.route_stop_times_.size();
      for (auto const [from, to] :
           utl::pairwise(interval{std::size_t{0U}, stop_seq.size()})) {
        // Write departure times of all route services at stop i.
        for (auto const& s : services) {
          tt_.route_stop_times_.emplace_back(s.utc_times_[from * 2]);
        }

        // Write arrival times of all route services at stop i+1.
        for (auto const& s : services) {
          tt_.route_stop_times_.emplace_back(s.utc_times_[to * 2 - 1]);
        }
      }
      auto const stop_times_end = tt_.route_stop_times_.size();
      tt_.route_stop_time_ranges_.emplace_back(
          interval{stop_times_begin, stop_times_end});
    }
  }
  route_services_.clear();
  store_.clear();
}

void service_builder::write_location_routes() {
  for (auto l = tt_.location_routes_.size(); l != tt_.n_locations(); ++l) {
    tt_.location_routes_.emplace_back(location_routes_[location_idx_t{l}]);
    assert(tt_.location_routes_.size() == l + 1U);
  }
}

}  // namespace nigiri::loader::hrd
