#include "nigiri/loader/hrd/service/service_builder.h"

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"

namespace nigiri::loader::hrd {

std::optional<size_t> get_index(vector<ref_service> const& route_services,
                                ref_service const& s) {
  auto const index = static_cast<unsigned>(std::distance(
      begin(route_services),
      std::lower_bound(begin(route_services), end(route_services), s,
                       [&](ref_service const& a, ref_service const& b) {
                         return a.utc_times_.front() % 1440 <
                                b.utc_times_.front() % 1440;
                       })));

  for (auto i = 0U; i != s.utc_times_.size(); ++i) {
    auto const is_earlier_eq =
        index > 0 && s.utc_times_[i] % 1440 <
                         route_services[index - 1].utc_times_.at(i) % 1440;
    auto const is_later_eq =
        index < route_services.size() &&
        s.utc_times_[i] % 1440 > route_services[index].utc_times_.at(i) % 1440;
    if (is_earlier_eq || is_later_eq) {
      return std::nullopt;
    }
  }

  return index;
}

void service_builder::add_service(ref_service&& s) {
  route_key_.first.clear();
  utl::transform_to(
      s.stops(store_), route_key_.first, [&](service::stop const& x) {
        return timetable::stop(stamm_.resolve_location(x.eva_num_),
                               x.dep_.in_out_allowed_, x.arr_.in_out_allowed_)
            .value();
      });

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

void service_builder::add_services(config const& c,
                                   const char* filename,
                                   std::string_view file_content,
                                   progress_update_fn const& progress_update) {
  scoped_timer write{"reading services"};

  auto const source_file_idx = tt_.register_source_file(filename);
  parse_services(c, filename, source_file_idx, tt_.date_range_, store_, stamm_,
                 file_content, progress_update,
                 [&](ref_service&& s) { add_service(std::move(s)); });
}

void service_builder::write_services(const nigiri::source_idx_t src) {
  scoped_timer write{"writing services"};

  for (auto const& [key, sub_routes] : route_services_) {
    for (auto const& services : sub_routes) {
      auto const& [stop_seq, sections_clasz] = key;
      auto const route_idx = tt_.register_route(stop_seq, sections_clasz);
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
              tt_.next_transport_idx(),
              {0U, static_cast<unsigned>(stop_seq.size())});

          auto const get_attribute_combination_idx =
              [&](std::optional<std::vector<service::attribute>> const& a,
                  std::optional<std::vector<service::attribute>> const& b) {
                attribute_combination_.resize((a.has_value() ? a->size() : 0U) +
                                              (b.has_value() ? b->size() : 0U));

                auto i = 0U;
                if (a.has_value()) {
                  for (auto const& attr : a.value()) {
                    attribute_combination_[i++] = attr.code_;
                  }
                }
                if (b.has_value()) {
                  for (auto const& attr : b.value()) {
                    attribute_combination_[i++] = attr.code_;
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
          auto const has_no_attributes = [](service::section const& sec) {
            return !sec.attributes_.has_value();
          };
          if (!ref.sections_.empty() &&
              utl::all_of(ref.sections_, has_no_attributes)) {
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

          auto const merged_trip = tt_.register_merged_trip({id});
          tt_.add_transport(timetable::transport{
              .bitfield_idx_ = utl::get_or_create(
                  bitfield_indices_, s.utc_traffic_days_,
                  [&]() { return tt_.register_bitfield(s.utc_traffic_days_); }),
              .route_idx_ = route_idx,
              .stop_times_ = s.utc_times_,
              .external_trip_ids_ = {merged_trip},
              .section_attributes_ = section_attributes_,
              .section_providers_ = section_providers_,
              .section_directions_ = section_directions_});
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
