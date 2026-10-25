#include "nigiri/loader/hrd/stamm/transfer_times.h"

#include <algorithm>
#include <string>
#include <string_view>

#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

#include "fmt/format.h"

#include "nigiri/loader/hrd/service/service.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/loader/register.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/common/merge_sorted.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

// === file loading =========================================================

std::vector<file> load_transfer_time_files(config const& c, dir const& d) {
  auto const load = [&](std::vector<std::string> const& alt) {
    for (auto const& f : alt) {
      try {
        return d.get_file(c.prefix(d) / c.core_data_ / f);
      } catch (...) {
        continue;
      }
    }
    return file{};
  };
  auto ret = std::vector<file>{};
  ret.emplace_back(load(c.transfers_.station_));
  ret.emplace_back(load(c.transfers_.admin_));
  ret.emplace_back(load(c.transfers_.line_));
  ret.emplace_back(load(c.transfers_.trip_));
  return ret;
}

// === parsing ==============================================================

namespace {

constexpr auto const kDefaultEva = eva_number{9999999U};

// column layouts acc. to HRDF 5.40.41, ch. 8 (0-based)
namespace col {
// umsteigb: eva, time IC-IC, time other
constexpr auto const b_eva = utl::field{0, 7};
constexpr auto const b_time_other = utl::field{11, 2};
// umsteigv: eva|@@@@@@@, admin1, admin2, time
constexpr auto const v_eva = utl::field{0, 7};
constexpr auto const v_admin_1 = utl::field{8, 6};
constexpr auto const v_admin_2 = utl::field{15, 6};
constexpr auto const v_time = utl::field{22, 2};
// umsteigl: eva|@@@@@@@, (admin, category, line, dir) x2, time, '!'
constexpr auto const l_eva = utl::field{0, 7};
constexpr auto const l_admin_1 = utl::field{8, 6};
constexpr auto const l_cat_1 = utl::field{15, 3};
constexpr auto const l_line_1 = utl::field{19, 8};
constexpr auto const l_dir_1 = utl::field{28, 1};
constexpr auto const l_admin_2 = utl::field{30, 6};
constexpr auto const l_cat_2 = utl::field{37, 3};
constexpr auto const l_line_2 = utl::field{41, 8};
constexpr auto const l_dir_2 = utl::field{50, 1};
constexpr auto const l_time = utl::field{52, 3};
constexpr auto const l_guaranteed = utl::field{55, 1};
// umsteigz: eva, (trip number, admin) x2, time, '!', traffic day bitfield
constexpr auto const z_eva = utl::field{0, 7};
constexpr auto const z_nr_1 = utl::field{8, 6};
constexpr auto const z_admin_1 = utl::field{15, 6};
constexpr auto const z_nr_2 = utl::field{22, 6};
constexpr auto const z_admin_2 = utl::field{29, 6};
constexpr auto const z_time = utl::field{36, 3};
constexpr auto const z_guaranteed = utl::field{39, 1};
constexpr auto const z_bitfield = utl::field{41, 6};
}  // namespace col

// HAFAS precedence as specificity: a line pair ranks by its wildcards
constexpr auto const kAdminGlobal = transfer_rule_specificity_t{1U};
constexpr auto const kLineGlobal = transfer_rule_specificity_t{2U};  // +0..8
constexpr auto const kAdminStation = transfer_rule_specificity_t{11U};
constexpr auto const kLineStation = transfer_rule_specificity_t{12U};  // +0..8
constexpr auto const kTripStation = transfer_rule_specificity_t{21U};

constexpr std::uint8_t kFromListed = 1U;
constexpr std::uint8_t kToListed = 2U;

bool is_comment(utl::cstr const line) {
  return line.len == 0U || line[0] == '%' || line[0] == '*';
}

u8_minutes parse_time(utl::cstr const s) {
  auto const t = utl::parse<int>(s.trim(), 0);
  return u8_minutes{static_cast<u8_minutes::rep>(std::clamp(t, 0, 255))};
}

char normalize_dir(char const c) {
  switch (c) {
    case 'H': return '1';
    case 'R': return '2';
    default: return c;
  }
}

std::array<char, 8U> to_line_array(utl::cstr const s) {
  auto ret = std::array<char, 8U>{};
  auto const trimmed = s.trim();
  std::copy_n(trimmed.begin(), std::min(trimmed.len, std::size_t{8U}),
              ret.begin());
  return ret;
}

// a side with a wildcard everywhere qualifies nothing
bool is_listed(line_rule_side const& s) {
  return s.admin_ != provider_idx_t::invalid() || !s.cat_any_ || !s.line_any_ ||
         s.dir_ != '*';
}

std::uint8_t get_listed(line_rule const& r) {
  return static_cast<std::uint8_t>((is_listed(r.from_) ? kFromListed : 0U) |
                                   (is_listed(r.to_) ? kToListed : 0U));
}

transfer_rule_specificity_t get_line_specificity(
    transfer_rule_specificity_t const tier, line_rule const& r) {
  return static_cast<transfer_rule_specificity_t>(tier + 8U - r.n_wildcards_);
}

transfer_rule_idx_t add_rule(timetable& tt,
                             source_idx_t const src,
                             location_idx_t const station,
                             u8_minutes const time,
                             transfer_rule_specificity_t const specificity,
                             std::uint8_t const listed) {
  auto& rules = tt.transfer_rules_.rules_;
  rules.push_back(transfer_rule{.from_stop_ = station,
                                .to_stop_ = station,
                                .src_ = src,
                                .duration_ = duration_t{time.count()},
                                .specificity_ = specificity,
                                .listed_ = listed});
  return transfer_rule_idx_t{rules.size() - 1U};
}

bool matches(line_rule_side const& r, event_attrs const& e) {
  return e.valid_ &&
         (r.admin_ == provider_idx_t::invalid() || r.admin_ == e.admin_) &&
         (r.cat_any_ || r.cat_ == e.cat_) &&
         (r.line_any_ || r.line_ == e.line_) &&
         (r.dir_ == '*' || r.dir_ == e.dir_);
}

}  // namespace

void parse_station_transfer_times(transfer_times& tt,
                                  std::string_view file_content) {
  utl::for_each_line(utl::cstr{file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 13U) {
      return;
    }
    auto const eva = parse_eva_number(line.substr(col::b_eva));
    auto const time = parse_time(line.substr(col::b_time_other));
    if (eva == kDefaultEva) {
      tt.default_ = time;
    } else {
      tt.station_[eva] = time;
    }
  });
}

void parse_transfer_time_rules(transfer_times& r,
                               stamm& st,
                               timetable& tt,
                               source_idx_t const src,
                               std::string_view admin_file_content,
                               std::string_view line_file_content,
                               std::string_view trip_file_content) {
  if (admin_file_content.empty() && line_file_content.empty() &&
      trip_file_content.empty()) {
    return;
  }

  auto const timer = scoped_timer{"loader.hrd.transfer_times"};
  r.first_rule_ = transfer_rule_idx_t{tt.transfer_rules_.rules_.size()};

  auto const is_global = [](utl::cstr const eva_field) {
    return eva_field.starts_with("@@@@@@@");
  };
  auto const station_of = [&](utl::cstr const eva_field) {
    return st.resolve_location(parse_eva_number(eva_field));
  };

  // --- umsteigv ---
  utl::for_each_line(utl::cstr{admin_file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 24U) {
      return;
    }
    auto const rule =
        admin_rule{.from_ = st.resolve_provider(line.substr(col::v_admin_1)),
                   .to_ = st.resolve_provider(line.substr(col::v_admin_2)),
                   .time_ = parse_time(line.substr(col::v_time))};
    if (is_global(line)) {
      r.admin_global_from_[rule.from_].push_back(r.admin_global_.size());
      r.admin_global_to_.insert(rule.to_);
      r.admin_global_.push_back(rule);
      return;
    }
    auto const station = station_of(line.substr(col::v_eva));
    if (station == location_idx_t::invalid()) {
      return;
    }
    auto a = rule;
    a.rule_ = add_rule(tt, src, station, rule.time_, kAdminStation,
                       kFromListed | kToListed);
    r.admin_[parse_eva_number(line.substr(col::v_eva))].push_back(a);
  });

  // --- umsteigl ---
  auto const parse_side = [&](utl::cstr const admin, utl::cstr const cat,
                              utl::cstr const line_id, utl::cstr const dir,
                              std::uint8_t& n_wildcards) -> line_rule_side {
    auto side = line_rule_side{};
    if (admin.trim().view() == "*") {
      ++n_wildcards;
    } else {
      side.admin_ = st.resolve_provider(admin);
    }
    if (cat.trim().view() == "*") {
      ++n_wildcards;
    } else {
      side.cat_any_ = false;
      side.cat_ = st.resolve_category(cat);
    }
    if (line_id.trim().view() == "*") {
      ++n_wildcards;
    } else {
      side.line_any_ = false;
      side.line_ = to_line_array(line_id);
    }
    auto const d = normalize_dir(dir.len == 0U ? '*' : dir[0]);
    if (d == '*' || d == ' ') {
      ++n_wildcards;
    } else {
      side.dir_ = d;
    }
    return side;
  };
  auto const index_global_side = [&](line_rule_side const& side,
                                     std::size_t const i, bool const is_from) {
    if (side.admin_ == provider_idx_t::invalid()) {
      (is_from ? r.line_global_from_any_ : r.line_global_to_any_).push_back(i);
    } else {
      (is_from ? r.line_global_from_ : r.line_global_to_)[side.admin_]
          .push_back(i);
    }
  };

  utl::for_each_line(utl::cstr{line_file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 55U) {
      return;
    }
    auto rule = line_rule{};
    rule.from_ =
        parse_side(line.substr(col::l_admin_1), line.substr(col::l_cat_1),
                   line.substr(col::l_line_1), line.substr(col::l_dir_1),
                   rule.n_wildcards_);
    rule.to_ = parse_side(line.substr(col::l_admin_2),
                          line.substr(col::l_cat_2), line.substr(col::l_line_2),
                          line.substr(col::l_dir_2), rule.n_wildcards_);
    rule.time_ = parse_time(line.substr(col::l_time));
    rule.guaranteed_ = line.len > col::l_guaranteed.from &&
                       line[col::l_guaranteed.from] == '!';
    if (is_global(line)) {
      index_global_side(rule.from_, r.line_global_.size(), true);
      index_global_side(rule.to_, r.line_global_.size(), false);
      r.line_global_.push_back(rule);
      return;
    }
    auto const station = station_of(line.substr(col::l_eva));
    if (station == location_idx_t::invalid()) {
      return;
    }
    rule.rule_ =
        add_rule(tt, src, station, rule.time_,
                 get_line_specificity(kLineStation, rule), get_listed(rule));
    r.line_[parse_eva_number(line.substr(col::l_eva))].push_back(rule);
  });

  // --- umsteigz ---
  utl::for_each_line(utl::cstr{trip_file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 39U) {
      return;
    }
    auto const station = station_of(line.substr(col::z_eva));
    if (station == location_idx_t::invalid()) {
      return;
    }
    auto rule = trip_rule{
        .from_nr_ =
            utl::parse<std::uint32_t>(line.substr(col::z_nr_1).trim(), 0U),
        .from_admin_ = st.resolve_provider(line.substr(col::z_admin_1)),
        .to_nr_ =
            utl::parse<std::uint32_t>(line.substr(col::z_nr_2).trim(), 0U),
        .to_admin_ = st.resolve_provider(line.substr(col::z_admin_2)),
        .time_ = parse_time(line.substr(col::z_time)),
        .guaranteed_ = line.len > col::z_guaranteed.from &&
                       line[col::z_guaranteed.from] == '!',
        .bitfield_num_ =
            line.len >= col::z_bitfield.from + col::z_bitfield.size
                ? utl::parse<unsigned>(line.substr(col::z_bitfield).trim(), 0U)
                : 0U};
    rule.rule_ = add_rule(tt, src, station, rule.time_, kTripStation,
                          kFromListed | kToListed);
    r.trip_[parse_eva_number(line.substr(col::z_eva))].push_back(rule);
  });

  r.n_station_rules_ = tt.transfer_rules_.rules_.size() - to_idx(r.first_rule_);
  log(log_lvl::info, "loader.hrd.transfer_times",
      "transfer time rules: {} station times (default={}), admin={}+{} "
      "global, line={}+{} global, trip={}",
      r.station_.size(), r.default_.has_value() ? r.default_->count() : -1,
      r.admin_.size(), r.admin_global_.size(), r.line_.size(),
      r.line_global_.size(), r.trip_.size());
}

// === stop event attributes ================================================

namespace {

event_attrs from_section(service const& s, std::size_t const sec_idx) {
  auto e = event_attrs{};
  e.valid_ = true;

  auto const& b2e = s.begin_to_end_info_;
  auto const* sec =
      sec_idx < s.sections_.size() ? &s.sections_[sec_idx] : nullptr;

  e.train_nr_ = b2e.train_num_.has_value()
                    ? *b2e.train_num_
                    : (sec != nullptr && sec->train_num_.has_value()
                           ? *sec->train_num_
                           : s.initial_train_num_);
  e.admin_ = b2e.admin_.has_value() ? *b2e.admin_
                                    : (sec != nullptr && sec->admin_.has_value()
                                           ? *sec->admin_
                                           : s.initial_admin_);
  e.cat_ = b2e.category_.has_value()
               ? *b2e.category_
               : (sec != nullptr && sec->category_.has_value() ? *sec->category_
                                                               : nullptr);
  auto const line =
      b2e.line_.has_value()
          ? *b2e.line_
          : (sec != nullptr && sec->line_.has_value() ? *sec->line_
                                                      : utl::cstr{});
  if (line.valid()) {
    e.line_ = to_line_array(line);
  }
  e.dir_ = b2e.dir_flag_.has_value()
               ? *b2e.dir_flag_
               : (sec != nullptr && sec->dir_flag_.has_value() ? *sec->dir_flag_
                                                               : ' ');
  return e;
}

}  // namespace

stop_attrs get_stop_attrs(service const& s, std::size_t const stop_idx) {
  auto a = stop_attrs{};
  if (stop_idx > 0U) {
    a.arr_ = from_section(s, stop_idx - 1U);
  }
  if (stop_idx + 1U < s.stops_.size()) {
    a.dep_ = from_section(s, stop_idx);
  }
  return a;
}

// === route ids ============================================================

namespace {

route_type_t get_route_type(category const* cat) {
  if (cat == nullptr) {
    return route_type_t{2U};
  }
  switch (cat->clasz_) {
    case clasz::kAir: return route_type_t{1100U};
    case clasz::kHighSpeed: return route_type_t{101U};
    case clasz::kLongDistance: return route_type_t{102U};
    case clasz::kCoach: return route_type_t{200U};
    case clasz::kNight: return route_type_t{105U};
    case clasz::kRegional: return route_type_t{106U};
    case clasz::kSuburban: return route_type_t{109U};
    case clasz::kSubway: return route_type_t{1U};
    case clasz::kTram: return route_type_t{0U};
    case clasz::kBus: return route_type_t{3U};
    case clasz::kShip: return route_type_t{4U};
    case clasz::kFunicular: return route_type_t{7U};
    case clasz::kAerialLift: return route_type_t{6U};
    case clasz::kRideSharing:
    case clasz::kODM: return route_type_t{1700U};
    case clasz::kOther:
    case clasz::kNumClasses: return route_type_t{2U};
  }
  return route_type_t{2U};
}

std::string_view get_line(std::array<char, 8U> const& line) {
  return {line.data(),
          static_cast<std::size_t>(std::find(begin(line), end(line), '\0') -
                                   begin(line))};
}

}  // namespace

route_id_idx_t get_or_create_route_id(transfer_times& r,
                                      timetable& tt,
                                      source_idx_t const src,
                                      service const& ref,
                                      std::size_t const first_stop) {
  auto const a = get_stop_attrs(ref, first_stop).dep_;
  auto const line = get_line(a.line_);
  auto const cat =
      a.cat_ != nullptr ? std::string_view{a.cat_->name_} : std::string_view{};
  auto const id =
      fmt::format("{}/{}/{}/{}", to_idx(a.admin_), cat, line, a.dir_);
  return utl::get_or_create(r.route_ids_, id, [&]() {
    auto const long_name = a.cat_ != nullptr
                               ? std::string_view{a.cat_->long_name_}
                               : std::string_view{};
    auto const idx = register_route(
        tt, route{tt, src, id, tt.register_translation(line),
                  tt.register_translation(long_name), kEmptyTranslation,
                  get_route_type(a.cat_), route_color{}, a.admin_,
                  category_idx_t::invalid(), ticketing_link_idx_t::invalid()});
    if (r.route_attrs_.size() <= to_idx(idx)) {
      r.route_attrs_.resize(to_idx(idx) + 1U);
    }
    r.route_attrs_[idx] = a;
    return idx;
  });
}

// === rules per trip =======================================================

void add_trip_sides(transfer_times& r,
                    stamm const& st,
                    trip_idx_t const trip,
                    route_id_idx_t const route,
                    service const& ref,
                    interval<std::size_t> const stops) {
  for (auto i = stops.from_; i != stops.to_; ++i) {
    auto const eva = ref.stops_[i].eva_num_;
    auto const station = st.resolve_location(eva);
    auto const add = [&](transfer_rule_idx_t const rule, event_type const ev) {
      r.trip_sides_.push_back(
          {trip, station, transfer_rule_side_idx{rule, ev}});
    };
    auto attrs = get_stop_attrs(ref, i);
    attrs.arr_.valid_ = attrs.arr_.valid_ && i != stops.from_;
    attrs.dep_.valid_ = attrs.dep_.valid_ && i + 1U != stops.to_;
    if (auto const it = r.trip_.find(eva); it != end(r.trip_)) {
      for (auto const& t : it->second) {
        if (attrs.arr_.valid_ && t.from_nr_ == attrs.arr_.train_nr_ &&
            t.from_admin_ == attrs.arr_.admin_) {
          add(t.rule_, event_type::kArr);
        }
        if (attrs.dep_.valid_ && t.to_nr_ == attrs.dep_.train_nr_ &&
            t.to_admin_ == attrs.dep_.admin_) {
          add(t.rule_, event_type::kDep);
        }
      }
    }
    auto const has_own_time = r.station_.contains(eva);
    if (attrs.arr_.valid_) {
      r.arr_routes_.insert({station, route});
    }
    if (attrs.dep_.valid_) {
      r.dep_routes_.insert({station, route});
    }
    if (has_own_time) {
      continue;  // own transfer time: the global rules do not apply
    }
    auto const admin = r.route_attrs_[route].admin_;
    if (attrs.arr_.valid_ && r.admin_global_from_.contains(admin)) {
      r.arr_admins_.insert({station, admin});
    }
    if (attrs.dep_.valid_ && r.admin_global_to_.contains(admin)) {
      r.dep_admins_.insert({station, admin});
    }
  }
}

void write_transfer_rules(transfer_times& r,
                          stamm const& st,
                          timetable& tt,
                          source_idx_t const src) {
  auto const timer = scoped_timer{"loader.hrd.transfer_rules"};
  auto& tr = tt.transfer_rules_;

  auto const by_station =
      [](hash_set<pair<location_idx_t, route_id_idx_t>> const& s) {
        auto m = hash_map<location_idx_t, std::vector<route_id_idx_t>>{};
        for (auto const& [l, route] : s) {
          m[l].push_back(route);
        }
        return m;
      };
  auto const arr_routes = by_station(r.arr_routes_);
  auto const dep_routes = by_station(r.dep_routes_);
  auto const no_routes = std::vector<route_id_idx_t>{};
  auto const routes_at =
      [&](auto const& m,
          location_idx_t const l) -> std::vector<route_id_idx_t> const& {
    auto const it = m.find(l);
    return it == end(m) ? no_routes : it->second;
  };
  auto const add_route_sides = [&](transfer_rule_idx_t const rule,
                                   line_rule const& l,
                                   location_idx_t const station) {
    if (is_listed(l.from_)) {
      for (auto const route : routes_at(arr_routes, station)) {
        if (matches(l.from_, r.route_attrs_[route])) {
          r.route_sides_.push_back(
              {{src, route}, station, {rule, event_type::kArr}});
        }
      }
    }
    if (is_listed(l.to_)) {
      for (auto const route : routes_at(dep_routes, station)) {
        if (matches(l.to_, r.route_attrs_[route])) {
          r.route_sides_.push_back(
              {{src, route}, station, {rule, event_type::kDep}});
        }
      }
    }
  };

  // the line pair rules at a station
  for (auto const& [eva, rules] : r.line_) {
    auto const station = st.resolve_location(eva);
    if (station == location_idx_t::invalid()) {
      continue;
    }
    for (auto const& l : rules) {
      add_route_sides(l.rule_, l, station);
    }
  }

  // the global line pair rules: a copy per station without an own transfer
  // time where a route matches each listed side
  auto n_line_copies = 0U;
  if (!r.line_global_.empty()) {
    auto own_time = hash_set<location_idx_t>{};
    for (auto const& [eva, _] : r.station_) {
      own_time.insert(st.resolve_location(eva));
    }
    auto stations = hash_set<location_idx_t>{};
    for (auto const& [l, _] : arr_routes) {
      stations.insert(l);
    }
    for (auto const& [l, _] : dep_routes) {
      stations.insert(l);
    }
    auto const find = [](auto const& m, provider_idx_t const p) {
      auto const it = m.find(p);
      return it == end(m) ? nullptr : &it->second;
    };
    auto matched_from = hash_map<std::size_t, std::vector<route_id_idx_t>>{};
    auto matched_to = hash_map<std::size_t, std::vector<route_id_idx_t>>{};
    auto const match = [&](std::vector<route_id_idx_t> const& routes,
                           auto const& by_admin,
                           std::vector<std::size_t> const& any_admin,
                           bool const is_from, auto& matched) {
      for (auto const route : routes) {
        auto const& a = r.route_attrs_[route];
        auto const check = [&](std::size_t const g) {
          auto const& l = r.line_global_[g];
          if (matches(is_from ? l.from_ : l.to_, a)) {
            matched[g].push_back(route);
          }
        };
        if (auto const* idx = find(by_admin, a.admin_); idx != nullptr) {
          for (auto const g : *idx) {
            check(g);
          }
        }
        for (auto const g : any_admin) {
          check(g);
        }
      }
    };
    for (auto const station : stations) {
      if (own_time.contains(station)) {
        continue;
      }
      matched_from.clear();
      matched_to.clear();
      match(routes_at(arr_routes, station), r.line_global_from_,
            r.line_global_from_any_, true, matched_from);
      match(routes_at(dep_routes, station), r.line_global_to_,
            r.line_global_to_any_, false, matched_to);
      for (auto g = std::size_t{0U}; g != r.line_global_.size(); ++g) {
        auto const& l = r.line_global_[g];
        if ((is_listed(l.from_) && !matched_from.contains(g)) ||
            (is_listed(l.to_) && !matched_to.contains(g))) {
          continue;
        }
        auto const rule =
            add_rule(tt, src, station, l.time_,
                     get_line_specificity(kLineGlobal, l), get_listed(l));
        for (auto const route : matched_from[g]) {
          r.route_sides_.push_back(
              {{src, route}, station, {rule, event_type::kArr}});
        }
        for (auto const route : matched_to[g]) {
          r.route_sides_.push_back(
              {{src, route}, station, {rule, event_type::kDep}});
        }
        ++n_line_copies;
      }
    }
  }

  // the admin pair rules: the route ids of the two admins at the station
  auto const add_admin_route_sides = [&](transfer_rule_idx_t const rule,
                                         admin_rule const& a,
                                         location_idx_t const station) {
    for (auto const route : routes_at(arr_routes, station)) {
      if (r.route_attrs_[route].admin_ == a.from_) {
        r.route_sides_.push_back(
            {{src, route}, station, {rule, event_type::kArr}});
      }
    }
    for (auto const route : routes_at(dep_routes, station)) {
      if (r.route_attrs_[route].admin_ == a.to_) {
        r.route_sides_.push_back(
            {{src, route}, station, {rule, event_type::kDep}});
      }
    }
  };
  for (auto const& [eva, rules] : r.admin_) {
    auto const station = st.resolve_location(eva);
    for (auto const& a : rules) {
      add_admin_route_sides(a.rule_, a, station);
    }
  }

  // the global admin pair rules: a copy per station where both admins meet
  auto n_admin_copies = 0U;
  for (auto const& [station, a] : r.arr_admins_) {
    for (auto const g : r.admin_global_from_.at(a)) {
      auto const& rule = r.admin_global_[g];
      if (!r.dep_admins_.contains(pair{station, rule.to_})) {
        continue;
      }
      add_admin_route_sides(add_rule(tt, src, station, rule.time_, kAdminGlobal,
                                     kFromListed | kToListed),
                            rule, station);
      ++n_admin_copies;
    }
  }

  merge_sorted(tr.trip_rules_, r.trip_sides_);
  merge_sorted(tr.route_rules_, r.route_sides_);
  store_rule_lookups(
      tt, interval{r.first_rule_, transfer_rule_idx_t{tr.rules_.size()}});
  log(log_lvl::info, "loader.hrd.transfer_rules",
      "{} rules at stations, {} line rule copies, {} admin rule copies, {} "
      "trip sides, {} route sides, {} route ids",
      r.n_station_rules_, n_line_copies, n_admin_copies, r.trip_sides_.size(),
      r.route_sides_.size(), r.route_ids_.size());
}

}  // namespace nigiri::loader::hrd
