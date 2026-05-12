#include "nigiri/timetable.h"

#include <ranges>

#include "cista/io.h"

#include "utl/overloaded.h"
#include "utl/verify.h"
#include "utl/visit.h"

#include "nigiri/common/day_list.h"
#include "nigiri/logging.h"
#include "nigiri/rt/frun.h"
#include "nigiri/translations_view.h"

namespace nigiri {

timezone_idx_t timetable::register_timezone(timezone tz) {
  auto const idx =
      timezone_idx_t{static_cast<timezone_idx_t::value_t>(timezones_.size())};
  timezones_.emplace_back(std::move(tz));
  return idx;
}

std::string_view timetable::get_default_name(location_idx_t const l) const {
  return get_default_translation(locations_.names_[l]);
}

translated_str_t timetable::get(translation_idx_t const t) const {
  if (translations_[t].size() != 1U) {
    auto ret = std::vector<translation>{};
    ret.reserve(translations_[t].size());
    for (auto const [lang, text] : get_translation_view(*this, t)) {
      ret.emplace_back(languages_.get(lang), text);
    }
    return ret;
  } else {
    return get_default_translation(t);
  }
}

translation_idx_t timetable::register_translation(std::string const& s) {
  return register_translation(std::string_view{s});
}

translation_idx_t timetable::register_translation(std::string_view s) {
  auto idx = translations_.size();
  translations_.emplace_back(std::initializer_list<std::string_view>{s});
  translation_language_.emplace_back({kDefaultLang});
  assert(translations_.size() == translation_language_.size());
  return translation_idx_t{idx};
}

translation_idx_t timetable::register_translation(translated_str_t const& s) {
  auto idx = translations_.size();
  utl::visit(
      s,
      [&](std::vector<translation> const& x) {
        using std::views::transform;
        translations_.emplace_back(
            x | transform([](translation const& t) { return t.get_text(); }));
        translation_language_.emplace_back(
            x | transform([&](translation const& t) {
              return languages_.store(t.get_language());
            }));
      },
      [&](std::string_view const& x) { register_translation(x); });
  assert(idx == translations_.size() - 1);
  assert(translations_.size() == translation_language_.size());
  return translation_idx_t{idx};
}

std::string_view timetable::get_default_translation(
    translation_idx_t const t) const {
  return t == translation_idx_t::invalid() ? "" : translations_[t][0].view();
}

std::string_view timetable::translate(lang_t const& lang,
                                      translation_idx_t const t) const {
  if (!lang.has_value()) {
    return get_default_translation(t);
  }

  using std::views::filter;
  using std::views::transform;
  for (auto const preferred_lang :
       *lang  //
           | transform([&](auto&& x) { return languages_.find(x); })  //
           | filter([](auto&& x) { return x.has_value(); })  //
           | transform([](auto&& x) { return *x; })) {
    for (auto const [translation_lang, text] : get_translation_view(*this, t)) {
      if (translation_lang == preferred_lang) {
        return text;
      }
    }
  }

  return get_default_translation(t);
}

std::optional<location_idx_t> timetable::find(location_id const& id) const {
  auto const it = locations_.location_id_to_idx_.find(id);
  return it == end(locations_.location_id_to_idx_) ? std::nullopt
                                                   : std::optional{it->second};
}

void timetable::resolve() {
  for (auto& tz : timezones_) {
    if (holds_alternative<pair<string, void const*>>(tz)) {
      auto& [name, ptr] = tz.as<pair<string, void const*>>();
      ptr = date::locate_zone(name);
    }
  }
}

bitfield_idx_t timetable::register_bitfield(bitfield const& b) {
  auto const idx = bitfield_idx_t{bitfields_.size()};
  bitfields_.emplace_back(b);
  return idx;
}

route_idx_t timetable::register_route(
    basic_string<stop::value_type> const& stop_seq,
    basic_string<clasz> const& clasz_sections,
    bitvec const& bikes_allowed_per_section,
    bitvec const& cars_allowed_per_section) {
  assert(stop_seq.size() > 1U);
  assert(!clasz_sections.empty());

  auto const idx = route_location_seq_.size();

  route_transport_ranges_.emplace_back(
      transport_idx_t{transport_traffic_days_.size()},
      transport_idx_t::invalid());
  route_location_seq_.emplace_back(stop_seq);
  route_section_clasz_.emplace_back(clasz_sections);
  route_clasz_.emplace_back(clasz_sections[0]);

  auto const bike_sections = bikes_allowed_per_section.size();
  auto const sections_with_bikes_allowed = bikes_allowed_per_section.count();
  auto const bikes_allowed_on_all_sections =
      sections_with_bikes_allowed == bike_sections && bike_sections != 0;
  auto const bikes_allowed_on_some_sections = sections_with_bikes_allowed != 0U;
  route_bikes_allowed_.resize(route_bikes_allowed_.size() + 2U);
  route_bikes_allowed_.set(idx * 2, bikes_allowed_on_all_sections);
  route_bikes_allowed_.set(idx * 2 + 1, bikes_allowed_on_some_sections);

  route_bikes_allowed_per_section_.resize(idx + 1);
  if (bikes_allowed_on_some_sections && !bikes_allowed_on_all_sections) {
    auto bucket = route_bikes_allowed_per_section_[route_idx_t{idx}];
    for (auto i = 0U; i < bikes_allowed_per_section.size(); ++i) {
      bucket.push_back(bikes_allowed_per_section[i]);
    }
  }

  auto const car_sections = cars_allowed_per_section.size();
  auto const sections_with_cars_allowed = cars_allowed_per_section.count();
  auto const cars_allowed_on_all_sections =
      sections_with_cars_allowed == car_sections && car_sections != 0;
  auto const cars_allowed_on_some_sections = sections_with_cars_allowed != 0U;
  route_cars_allowed_.resize(route_cars_allowed_.size() + 2U);
  route_cars_allowed_.set(idx * 2, cars_allowed_on_all_sections);
  route_cars_allowed_.set(idx * 2 + 1, cars_allowed_on_some_sections);

  route_cars_allowed_per_section_.resize(idx + 1);
  if (cars_allowed_on_some_sections && !cars_allowed_on_all_sections) {
    auto bucket = route_cars_allowed_per_section_[route_idx_t{idx}];
    for (auto i = 0U; i < cars_allowed_per_section.size(); ++i) {
      bucket.push_back(cars_allowed_per_section[i]);
    }
  }

  return route_idx_t{idx};
}

void timetable::finish_route() {
  route_transport_ranges_.back().to_ =
      transport_idx_t{transport_traffic_days_.size()};
}

provider_idx_t timetable::get_provider_idx(std::string_view id,
                                           source_idx_t const src) const {
  auto const id_str_idx = strings_.find(id);
  if (!id_str_idx.has_value()) {
    return provider_idx_t::invalid();
  }
  auto const it = std::lower_bound(
      begin(provider_id_to_idx_), end(provider_id_to_idx_), *id_str_idx,
      [&](provider_idx_t const a, string_idx_t const b) {
        auto const& p = providers_[a];
        return std::tuple{p.src_, p.id_} < std::tuple{src, b};
      });
  if (it == end(provider_id_to_idx_) || providers_[*it].src_ != src ||
      *id_str_idx != providers_[*it].id_) {
    return provider_idx_t::invalid();
  }
  return *it;
}

merged_trips_idx_t timetable::register_merged_trip(
    basic_string<trip_idx_t> const& trip_ids) {
  auto const idx = merged_trips_.size();
  merged_trips_.emplace_back(trip_ids);
  return merged_trips_idx_t{static_cast<merged_trips_idx_t::value_t>(idx)};
}

source_file_idx_t timetable::register_source_file(std::string_view path) {
  auto const idx = source_file_idx_t{source_file_names_.size()};
  source_file_names_.emplace_back(path);
  return idx;
}

void timetable::add_transport(transport&& t) {
  transport_first_dep_offset_.emplace_back(t.first_dep_offset_);
  transport_traffic_days_.emplace_back(t.bitfield_idx_);
  transport_route_.emplace_back(t.route_idx_);
  transport_to_trip_section_.emplace_back(t.external_trip_ids_);
  transport_section_attributes_.emplace_back(t.section_attributes_);
  transport_section_providers_.emplace_back(t.section_providers_);
  transport_section_directions_.emplace_back(t.section_directions_);

  assert(transport_traffic_days_.size() == transport_route_.size());
  assert(transport_traffic_days_.size() == transport_to_trip_section_.size());
  assert(transport_section_directions_.back().size() == 0U ||
         transport_section_directions_.back().size() == 1U ||
         transport_section_directions_.back().size() ==
             route_location_seq_.at(transport_route_.back()).size() - 1U);
}

transport_idx_t timetable::next_transport_idx() const {
  return transport_idx_t{transport_traffic_days_.size()};
}

std::string reverse(std::string s) {
  std::reverse(s.begin(), s.end());
  return s;
}

std::ostream& operator<<(std::ostream& out, timetable const& tt) {
  for (auto const [id, idx] : tt.trip_id_to_idx_) {
    auto const str = tt.trip_id_strings_[id].view();
    out << str << ":\n";
    for (auto const& t : tt.trip_transport_ranges_.at(idx)) {
      out << "  " << t.first << ": " << t.second << " active="
          << day_list{tt.bitfields_[tt.transport_traffic_days_[t.first]],
                      tt.internal_interval_days().from_}
          << "\n";
    }
  }

  auto const internal = tt.internal_interval_days();
  auto const num_days =
      static_cast<size_t>((internal.to_ - internal.from_ + 1_days) / 1_days);
  for (auto i = 0U; i != tt.transport_traffic_days_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const num_stops =
        tt.route_location_seq_[tt.transport_route_[transport_idx]].size();
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRANSPORT=" << transport_idx << ", TRAFFIC_DAYS="
        << reverse(traffic_days.to_string().substr(kMaxDays - num_days))
        << "\n";
    for (auto d = internal.from_; d != internal.to_;
         d += std::chrono::days{1}) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t::value_t>((d - internal.from_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", d);
        out << " (day_idx=" << day_idx << ")\n";
        out << rt::frun{
            tt,
            nullptr,
            {.t_ = transport{transport_idx, day_idx},
             .stop_range_ = {0U, static_cast<stop_idx_t>(num_stops)}}};
        out << "\n";
      }
    }
    out << "---\n\n";
  }
  return out;
}

day_list timetable::days(bitfield const& bf) const {
  return day_list{bf, internal_interval_days().from_};
}

cista::wrapped<timetable> timetable::read(std::filesystem::path const& p) {
  return cista::read<timetable>(p);
}

void timetable::write(std::filesystem::path const& p) const {
  return cista::write(p, *this);
}

}  // namespace nigiri
