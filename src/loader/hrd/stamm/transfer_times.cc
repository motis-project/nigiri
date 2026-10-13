#include "nigiri/loader/hrd/stamm/transfer_times.h"

#include <algorithm>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

#include "nigiri/loader/hrd/service/service.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/loader/register.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

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

namespace {

constexpr auto const kDefaultEva = eva_number{9999999U};

// column layouts acc. to HRDF 5.40.41, ch. 8 (0-based)
namespace col {
constexpr auto const b_eva = utl::field{0, 7};
constexpr auto const b_time_other = utl::field{11, 2};
constexpr auto const v_eva = utl::field{0, 7};
constexpr auto const v_admin_1 = utl::field{8, 6};
constexpr auto const v_admin_2 = utl::field{15, 6};
constexpr auto const v_time = utl::field{22, 2};
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
constexpr auto const z_eva = utl::field{0, 7};
constexpr auto const z_nr_1 = utl::field{8, 6};
constexpr auto const z_admin_1 = utl::field{15, 6};
constexpr auto const z_nr_2 = utl::field{22, 6};
constexpr auto const z_admin_2 = utl::field{29, 6};
constexpr auto const z_time = utl::field{36, 3};
}  // namespace col

bool is_comment(utl::cstr const line) {
  return line.len == 0U || line[0] == '%' || line[0] == '*';
}

bool is_global(utl::cstr const line) { return line.starts_with("@@@@@@@"); }

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

bool matches(line_rule_side const& r, event_attrs const& e) {
  return (r.admin_ == provider_idx_t::invalid() || r.admin_ == e.admin_) &&
         (r.cat_any_ || r.cat_ == e.cat_) &&
         (r.line_any_ || r.line_ == e.line_) &&
         (r.dir_ == '*' || r.dir_ == e.dir_);
}

event_attrs get_attrs(service const& s, std::size_t const sec_idx) {
  auto const& b2e = s.begin_to_end_info_;
  auto const* sec =
      sec_idx < s.sections_.size() ? &s.sections_[sec_idx] : nullptr;
  auto const get = [&](auto const member) {
    return (b2e.*member).has_value()
               ? (b2e.*member)
               : (sec != nullptr
                      ? (sec->*member)
                      : std::remove_cvref_t<decltype(b2e.*member)>{});
  };

  auto e = event_attrs{};
  e.train_nr_ =
      get(&service::section::train_num_).value_or(s.initial_train_num_);
  e.admin_ = get(&service::section::admin_).value_or(s.initial_admin_);
  e.cat_ = get(&service::section::category_).value_or(nullptr);
  if (auto const line = get(&service::section::line_); line.has_value()) {
    e.line_ = to_line_array(*line);
  }
  e.dir_ = get(&service::section::dir_flag_).value_or(' ');
  return e;
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
                               std::string_view admin_file_content,
                               std::string_view line_file_content,
                               std::string_view trip_file_content) {
  auto const timer = scoped_timer{"loader.hrd.transfer_times"};

  utl::for_each_line(utl::cstr{admin_file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 24U) {
      return;
    }
    auto const rule =
        admin_rule{.from_ = st.resolve_provider(line.substr(col::v_admin_1)),
                   .to_ = st.resolve_provider(line.substr(col::v_admin_2)),
                   .time_ = parse_time(line.substr(col::v_time))};
    if (is_global(line)) {
      r.admin_global_.emplace_back(rule);
    } else {
      r.admin_[parse_eva_number(line.substr(col::v_eva))].emplace_back(rule);
    }
  });

  auto const parse_side = [&](utl::cstr const admin, utl::cstr const cat,
                              utl::cstr const line_id, utl::cstr const dir,
                              std::uint8_t& n_wildcards) {
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
    if (is_global(line)) {
      r.line_global_.emplace_back(rule);
    } else {
      r.line_[parse_eva_number(line.substr(col::l_eva))].emplace_back(rule);
    }
  });

  // UMSTEIGZ with a traffic day bitfield applies on all days of the trips here
  utl::for_each_line(utl::cstr{trip_file_content}, [&](utl::cstr const line) {
    if (is_comment(line) || line.len < 39U) {
      return;
    }
    r.trip_[parse_eva_number(line.substr(col::z_eva))].emplace_back(trip_rule{
        .from_nr_ =
            utl::parse<std::uint32_t>(line.substr(col::z_nr_1).trim(), 0U),
        .from_admin_ = st.resolve_provider(line.substr(col::z_admin_1)),
        .to_nr_ =
            utl::parse<std::uint32_t>(line.substr(col::z_nr_2).trim(), 0U),
        .to_admin_ = st.resolve_provider(line.substr(col::z_admin_2)),
        .time_ = parse_time(line.substr(col::z_time))});
  });

  log(log_lvl::info, "loader.hrd.transfer_times",
      "{} station times (default={}), admin @ {} stations + {} global, line @ "
      "{} stations + {} global, trip @ {} stations",
      r.station_.size(), r.default_.has_value() ? r.default_->count() : -1,
      r.admin_.size(), r.admin_global_.size(), r.line_.size(),
      r.line_global_.size(), r.trip_.size());
}

transfer_rule_builder::transfer_rule_builder(timetable& tt,
                                             transfer_times const& r,
                                             stamm const& st,
                                             source_idx_t const src)
    : tt_{tt}, r_{r}, st_{st}, src_{src} {
  auto const add = [](auto& by_admin, auto& any, provider_idx_t const admin,
                      global_row const row) {
    (admin == provider_idx_t::invalid() ? any : by_admin[admin]).push_back(row);
  };
  for (auto i = 0U; i != r.line_global_.size(); ++i) {
    auto const& x = r.line_global_[i];
    add(global_from_, global_from_any_, x.from_.admin_, {tier::kLineGlobal, i});
    add(global_to_, global_to_any_, x.to_.admin_, {tier::kLineGlobal, i});
  }
  for (auto i = 0U; i != r.admin_global_.size(); ++i) {
    auto const& x = r.admin_global_[i];
    add(global_from_, global_from_any_, x.from_, {tier::kAdminGlobal, i});
    add(global_to_, global_to_any_, x.to_, {tier::kAdminGlobal, i});
  }
}

namespace {

transfer_rule_specificity_t line_specificity(bool const is_station,
                                             line_rule const& x) {
  return static_cast<transfer_rule_specificity_t>((is_station ? 200U : 100U) -
                                                  x.n_wildcards_);
}

}  // namespace

route_id_idx_t transfer_rule_builder::get_group(
    service const& s, interval<std::size_t> const stops) {
  if (r_.line_global_.empty() && r_.admin_global_.empty()) {
    return route_id_idx_t::invalid();
  }

  // sections [from, to - 1): the same attributes everywhere?
  auto const e = get_attrs(s, stops.from_);
  auto const key = group_key{e.admin_, e.cat_, e.line_, e.dir_};
  for (auto j = stops.from_ + 1U; j + 1U < stops.to_; ++j) {
    auto const x = get_attrs(s, j);
    if (group_key{x.admin_, x.cat_, x.line_, x.dir_} != key) {
      ++n_varying_;
      return route_id_idx_t::invalid();
    }
  }

  return utl::get_or_create(groups_, key, [&]() {
    auto const id = fmt::format("{}|{}|{}|{}", to_idx(e.admin_),
                                e.cat_ == nullptr ? "" : e.cat_->name_,
                                std::string_view{e.line_.data(), 8U}, e.dir_);
    auto const route = register_route(
        tt_, loader::route{tt_, src_, id, kEmptyTranslation, kEmptyTranslation,
                           kEmptyTranslation, route_type_t{0U}, route_color{},
                           e.admin_, category_idx_t::invalid(),
                           ticketing_link_idx_t::invalid()});

    // the global rows this group matches
    auto const add = [&](global_row const row, event_type const ev) {
      auto const [tr, i] = row;
      auto const is_from = ev == event_type::kArr;
      auto const is_line = tr == tier::kLineGlobal;
      if (is_line &&
          !matches(is_from ? r_.line_global_[i].from_ : r_.line_global_[i].to_,
                   e)) {
        return;
      }
      auto const idx = utl::get_or_create(
          instance_idx_, std::tuple{tr, i, location_idx_t::invalid()}, [&]() {
            instances_.push_back(instance{
                .tier_ = tr,
                .row_ = i,
                .station_ = location_idx_t::invalid(),
                .time_ = is_line ? r_.line_global_[i].time_
                                 : r_.admin_global_[i].time_,
                .specificity_ =
                    is_line ? line_specificity(false, r_.line_global_[i])
                            : transfer_rule_specificity_t{50U}});
            return static_cast<std::uint32_t>(instances_.size() - 1U);
          });
      instances_[idx].has_side_[is_from ? 0U : 1U] = true;
      group_sides_.emplace_back(route, idx, ev);
    };
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      auto const& by_admin = ev == event_type::kArr ? global_from_ : global_to_;
      if (auto const it = by_admin.find(e.admin_); it != end(by_admin)) {
        for (auto const& row : it->second) {
          add(row, ev);
        }
      }
      for (auto const& row :
           ev == event_type::kArr ? global_from_any_ : global_to_any_) {
        add(row, ev);
      }
    }
    return route;
  });
}

void transfer_rule_builder::add_trip(trip_idx_t const t,
                                     service const& s,
                                     interval<std::size_t> const stops) {
  auto const add = [&](tier const tr, std::uint32_t const row,
                       location_idx_t const l, u8_minutes const time,
                       transfer_rule_specificity_t const specificity,
                       event_type const ev) {
    auto const idx =
        utl::get_or_create(instance_idx_, std::tuple{tr, row, l}, [&]() {
          instances_.push_back(instance{.tier_ = tr,
                                        .row_ = row,
                                        .station_ = l,
                                        .time_ = time,
                                        .specificity_ = specificity});
          return static_cast<std::uint32_t>(instances_.size() - 1U);
        });
    instances_[idx].has_side_[ev == event_type::kArr ? 0U : 1U] = true;
    trip_sides_.emplace_back(t, idx, ev);
  };

  for (auto j = stops.from_; j != stops.to_; ++j) {
    auto const& stp = s.stops_[j];
    auto const eva = stp.eva_num_;
    auto const is_arr = j != stops.from_ && stp.arr_.in_out_allowed_;
    auto const is_dep = j + 1U != stops.to_ && stp.dep_.in_out_allowed_;
    auto const trip_rules = r_.trip_.find(eva);
    auto const line_rules = r_.line_.find(eva);
    auto const admin_rules = r_.admin_.find(eva);
    if ((!is_arr && !is_dep) ||
        (trip_rules == end(r_.trip_) && line_rules == end(r_.line_) &&
         admin_rules == end(r_.admin_))) {
      continue;
    }
    auto const l = st_.resolve_location(eva);
    if (l == location_idx_t::invalid()) {
      continue;
    }

    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      auto const is_from = ev == event_type::kArr;
      if (!(is_from ? is_arr : is_dep)) {
        continue;
      }
      auto const e = get_attrs(s, is_from ? j - 1U : j);
      if (trip_rules != end(r_.trip_)) {
        for (auto i = 0U; i != trip_rules->second.size(); ++i) {
          auto const& x = trip_rules->second[i];
          if ((is_from ? x.from_nr_ : x.to_nr_) == e.train_nr_ &&
              (is_from ? x.from_admin_ : x.to_admin_) == e.admin_) {
            add(tier::kTrip, i, l, x.time_, 250U, ev);
          }
        }
      }
      if (line_rules != end(r_.line_)) {
        for (auto i = 0U; i != line_rules->second.size(); ++i) {
          auto const& x = line_rules->second[i];
          if (matches(is_from ? x.from_ : x.to_, e)) {
            add(tier::kLineStation, i, l, x.time_, line_specificity(true, x),
                ev);
          }
        }
      }
      if (admin_rules != end(r_.admin_)) {
        for (auto i = 0U; i != admin_rules->second.size(); ++i) {
          auto const& x = admin_rules->second[i];
          if ((is_from ? x.from_ : x.to_) == e.admin_) {
            add(tier::kAdminStation, i, l, x.time_, 150U, ev);
          }
        }
      }
    }
  }
}

void transfer_rule_builder::write() {
  auto& tr = tt_.transfer_rules_;

  // Only rules with trips on both sides. Equal specificity: the later rule
  // index wins, so the earlier row comes later (HAFAS: first row wins).
  auto order = std::vector<std::uint32_t>{};
  for (auto i = 0U; i != instances_.size(); ++i) {
    if (instances_[i].has_side_[0] && instances_[i].has_side_[1]) {
      order.push_back(i);
    }
  }
  std::ranges::sort(order, [&](std::uint32_t const a, std::uint32_t const b) {
    auto const& x = instances_[a];
    auto const& y = instances_[b];
    return std::tuple{x.station_, x.tier_, y.row_} <
           std::tuple{y.station_, y.tier_, x.row_};
  });

  auto rule_idx = std::vector<transfer_rule_idx_t>(
      instances_.size(), transfer_rule_idx_t::invalid());
  for (auto const i : order) {
    auto const& x = instances_[i];
    rule_idx[i] = transfer_rule_idx_t{tr.rules_.size()};
    tr.rules_.push_back(
        transfer_rule{.from_stop_ = x.station_,  // invalid: global
                      .to_stop_ = x.station_,
                      .src_ = src_,
                      .duration_ = duration_t{x.time_.count()},
                      .specificity_ = x.specificity_,
                      .trip_set_ = 3U});
  }

  for (auto const& [t, i, ev] : trip_sides_) {
    if (rule_idx[i] != transfer_rule_idx_t::invalid()) {
      tr.trip_rules_.push_back({t, transfer_rule_side_idx{rule_idx[i], ev}});
    }
  }
  utl::erase_duplicates(tr.trip_rules_);
  for (auto const& [route, i, ev] : group_sides_) {
    if (rule_idx[i] != transfer_rule_idx_t::invalid()) {
      tr.route_rules_.push_back(
          {route, transfer_rule_side_idx{rule_idx[i], ev}});
    }
  }
  utl::erase_duplicates(tr.route_rules_);

  tr.global_excluded_.resize(tt_.n_locations());
  for (auto const& [eva, _] : r_.station_) {
    if (auto const l = st_.resolve_location(eva);
        l != location_idx_t::invalid()) {
      tr.global_excluded_.set(to_idx(l), true);
    }
  }

  log(log_lvl::info, "loader.hrd.transfer_times",
      "{} transfer rules ({} global), {} trip rule sides, {} groups with {} "
      "route rule sides, {} trips with changing attributes (no global rules)",
      order.size(),
      std::ranges::count_if(order,
                            [&](std::uint32_t const i) {
                              return instances_[i].station_ ==
                                     location_idx_t::invalid();
                            }),
      tr.trip_rules_.size(), groups_.size(), tr.route_rules_.size(),
      n_varying_);
  instances_ = {};
  instance_idx_ = {};
  trip_sides_ = {};
  group_sides_ = {};
  groups_ = {};
}

}  // namespace nigiri::loader::hrd
