#include "nigiri/loader/hrd/service/service.h"

#include <algorithm>

#include "fmt/format.h"

#include "utl/parser/arg_parser.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/hrd/stamm/timezone.h"
#include "nigiri/loader/hrd/util.h"
#include "nigiri/clasz.h"
#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"

namespace nigiri::loader::hrd {

std::ostream& operator<<(std::ostream& out, parser_info const& pi) {
  return out << pi.filename_ << ":" << pi.dbg_.line_number_from_ << ":"
             << pi.dbg_.line_number_to_;
}

template <typename It, typename Predicate>
inline It find_nth(It begin, It end, std::size_t n, Predicate fun) {
  assert(n != 0);
  std::size_t num_elements_found = 0;
  auto it = begin;
  while (it != end && num_elements_found != n) {
    it = std::find_if(it, end, fun);
    ++num_elements_found;
    if (it != end && num_elements_found != n) {
      ++it;
    }
  }
  return it;
}

struct range {
  range() = default;
  range(std::vector<service::stop> const& stops,
        utl::cstr from_eva_or_idx,
        utl::cstr to_eva_or_idx,
        utl::cstr from_hhmm_or_idx,
        utl::cstr to_hhmm_or_idx) {
    if (from_eva_or_idx.trim().empty() && to_eva_or_idx.trim().empty() &&
        from_hhmm_or_idx.trim().empty() && to_hhmm_or_idx.trim().empty()) {
      from_idx_ = 0U;
      to_idx_ = stops.size() - 1U;
    } else {
      from_idx_ =
          get_index(stops, from_eva_or_idx, from_hhmm_or_idx, event_type::kDep);
      to_idx_ =
          get_index(stops, to_eva_or_idx, to_hhmm_or_idx, event_type::kArr);
    }
  }

  size_t from_idx() const { return from_idx_; }
  size_t to_idx() const { return to_idx_; }

private:
  static bool is_index(utl::cstr s) { return s[0] == '#'; }

  static size_t parse_index(utl::cstr s) {
    return parse_verify<size_t>(s.substr(1));
  }

  static size_t get_index(std::vector<service::stop> const& stops,
                          utl::cstr eva_or_idx,
                          utl::cstr hhmm_or_idx,
                          event_type const ev_type) {
    assert(!eva_or_idx.empty() && !hhmm_or_idx.empty());
    if (is_index(eva_or_idx)) {
      // eva_or_idx is an index which is already definite
      return parse_index(eva_or_idx);
    } else if (is_index(hhmm_or_idx) || hhmm_or_idx.trim().len == 0) {
      // eva_or_idx is not an index -> eva_or_idx is an eva number
      // hhmm_or_idx is empty -> search for first occurrence
      // hhmm_or_idx is an index -> search for nth occurrence
      const auto eva_num = parse<unsigned>(eva_or_idx);
      const auto n = is_index(hhmm_or_idx) ? parse_index(hhmm_or_idx) + 1 : 1;
      const auto it = find_nth(
          begin(stops), end(stops), n,
          [&](service::stop const& s) { return s.eva_num_ == eva_num; });
      utl::verify(it != end(stops),
                  "{}th occurrence of eva number {} not found", n, eva_num);
      return static_cast<std::size_t>(std::distance(begin(stops), it));
    } else {
      // hhmm_or_idx must be a time
      // -> return stop where eva number and time matches
      const auto eva_num = parse<unsigned>(eva_or_idx);
      const auto time = hhmm_to_min(parse<int>(hhmm_or_idx.substr(1)));
      if (ev_type == event_type::kDep) {
        const auto it =
            std::find_if(begin(stops), end(stops), [&](service::stop const& s) {
              return s.eva_num_ == eva_num &&
                     (ev_type == event_type::kDep ? s.dep_.time_
                                                  : s.arr_.time_) == time;
            });
        utl::verify(it != end(stops),
                    "event with time {} at eva number {} not found", time,
                    eva_num);
        return static_cast<std::size_t>(std::distance(begin(stops), it));
      } else {
        const auto it = std::find_if(
            stops.rbegin(), stops.rend(), [&](service::stop const& s) {
              return s.eva_num_ == eva_num &&
                     (ev_type == event_type::kDep ? s.dep_.time_
                                                  : s.arr_.time_) == time;
            });
        utl::verify(it != stops.rend(),
                    "event with time {} at eva number {} not found", time,
                    eva_num);
        return static_cast<std::size_t>(&*it - stops.data());
      }
    }
  }

  std::size_t from_idx_{}, to_idx_{};
};

template <typename Fn>
void compute_ranges(std::vector<utl::cstr> const& spec_lines,
                    std::vector<service::stop> const& stops,
                    range_parse_information const& parse_info,
                    Fn&& fn) {
  for (auto const& spec : spec_lines) {
    fn(std::make_pair(spec,
                      range(stops, spec.substr(parse_info.from_eva_or_idx_),
                            spec.substr(parse_info.to_eva_or_idx_),
                            spec.substr(parse_info.from_hhmm_or_idx_),
                            spec.substr(parse_info.to_hhmm_or_idx_))));
  }
}

template <typename TargetInformationType, typename TargetInformationParserFun>
void parse_range(std::vector<utl::cstr> const& spec_lines,
                 range_parse_information const& parse_info,
                 std::vector<service::stop> const& stops,
                 std::vector<service::section>& sections,
                 service::section& begin_to_end,
                 TargetInformationType service::section::*member,
                 TargetInformationParserFun parse_target_info) {
  compute_ranges(spec_lines, stops, parse_info, [&](auto const& r) {
    TargetInformationType target_info = parse_target_info(r.first, r.second);

    if (r.second.from_idx() == 0U && r.second.to_idx() == stops.size() - 1U) {
      (begin_to_end.*member) = std::move(target_info);
    } else {
      sections.resize(stops.size() - 1);
      for (auto i = r.second.from_idx(); i < r.second.to_idx(); ++i) {
        (sections[i].*member) = target_info;
      }
    }
  });
}

template <typename TargetInformationType, typename TargetInformationParserFun>
void parse_range(
    std::vector<utl::cstr> const& spec_lines,
    range_parse_information const& parse_info,
    std::vector<service::stop> const& stops,
    std::vector<service::section>& sections,
    service::section& begin_to_end,
    std::optional<std::vector<TargetInformationType>> service::section::*member,
    TargetInformationParserFun parse_target_info) {
  compute_ranges(spec_lines, stops, parse_info, [&](auto const& r) {
    TargetInformationType target_info = parse_target_info(r.first, r.second);

    if (r.second.from_idx() == 0U && r.second.to_idx() == stops.size() - 1U) {
      if (!(begin_to_end.*member).has_value()) {
        (begin_to_end.*member) = std::vector<TargetInformationType>{};
      }
      (begin_to_end.*member)->push_back(std::move(target_info));
    } else {
      sections.resize(stops.size() - 1);
      for (auto i = r.second.from_idx(); i < r.second.to_idx(); ++i) {
        if (!(sections[i].*member).has_value()) {
          (sections[i].*member) = std::vector<TargetInformationType>{};
        }
        (sections[i].*member)->push_back(target_info);
      }
    }
  });
}

service::stop parse_stop(utl::cstr stop) {
  return {eva_number{parse_verify<unsigned>(stop.substr(0, utl::size(7)))},
          {hhmm_to_min(
               parse<int>(stop.substr(30, utl::size(5)), service::kTimeNotSet)),
           stop[29] != '-'},
          {hhmm_to_min(
               parse<int>(stop.substr(37, utl::size(5)), service::kTimeNotSet)),
           stop[36] != '-'}};
}

std::uint32_t initial_train_num(specification const& spec) {
  return parse_verify<std::uint32_t>(
      spec.internal_service_.substr(3, utl::size(5)));
}

utl::cstr initial_admin(specification const& spec) {
  return spec.internal_service_.substr(9, utl::size(6));
}

utl::cstr stop_train_num(utl::cstr const& stop) {
  return stop.substr(43, utl::size(5)).trim();
}

utl::cstr stop_admin(utl::cstr const& stop) {
  return stop.substr(49, utl::size(6)).trim();
}

bool specification::is_empty() const { return !internal_service_; }

bool specification::valid() const {
  return ignore() || (!categories_.empty() && stops_.size() >= 2 &&
                      !traffic_days_.empty() && !is_empty());
}

bool specification::ignore() const {
  return !is_empty() && !internal_service_.starts_with("*Z");
}

void specification::reset() {
  internal_service_ = utl::cstr(nullptr, 0);
  traffic_days_.clear();
  categories_.clear();
  line_information_.clear();
  attributes_.clear();
  directions_.clear();
  stops_.clear();
}

bool specification::read_line(utl::cstr line,
                              char const* filename,
                              unsigned const line_number) {
  if (line.len == 0) {
    return false;
  }

  if (std::isdigit(line[0]) != 0) {
    stops_.push_back(line);
    return false;
  }

  if (line.len < 2 || line[0] != '*') {
    throw utl::fail("invalid line {}:{}: \"{}\"", filename, line_number,
                    line.view());
  }

  // ignore *I, *GR, *SH, *T, *KW, *KWZ
  bool potential_kurswagen = false;
  switch (line[1]) {
    case 'K': potential_kurswagen = true; [[fallthrough]];
    case 'Z': [[fallthrough]];
    case 'T':
      if (potential_kurswagen && line.len > 3 && line[3] == 'Z') {
        // ignore KWZ line
      } else if (is_empty()) {
        filename_ = filename;
        line_number_from_ = line_number;
        internal_service_ = line;
      } else {
        return true;
      }
      break;
    case 'A':
      if (line.starts_with("*A VE")) {
        traffic_days_.push_back(line);
      } else {  // *A based on HRD format version 5.00.8
        attributes_.push_back(line);
      }
      break;
    case 'G':
      if (!line.starts_with("*GR")) {
        categories_.push_back(line);
      }
      break;
    case 'L': line_information_.push_back(line); break;
    case 'R': directions_.push_back(line); break;
  }

  return false;
}

void parse_sections(stamm& st,
                    specification const& spec,
                    std::vector<service::section>& sections,
                    service::section& start_to_end) {
  auto const start_admin =
      st.resolve_provider(spec.internal_service_.substr(9, utl::size(6)));
  auto const start_train_num = initial_train_num(spec);

  auto const has_extra_info = [](utl::cstr const stop_str) {
    return !stop_train_num(stop_str).empty() || !stop_admin(stop_str).empty();
  };

  if (utl::none_of(spec.stops_, has_extra_info)) {
    start_to_end.admin_ = start_admin;
    start_to_end.train_num_ = start_train_num;
  } else {
    auto pred_admin = start_admin;
    auto pred_train_nr = start_train_num;
    for (auto i = 0U; i != spec.stops_.size() - 1; ++i) {
      auto const train_nr = stop_train_num(spec.stops_[i]);
      auto const admin = stop_admin(spec.stops_[i]);
      auto const& sec = sections.emplace_back(
          train_nr.empty() ? pred_train_nr
                           : utl::parse_verify<std::uint32_t>(train_nr),
          admin.empty() ? pred_admin : st.resolve_provider(admin));
      pred_admin = sec.admin_.value();
      pred_train_nr = sec.train_num_.value();
    }
  }
}

service::service(config const& c,
                 stamm& st,
                 source_file_idx_t const source_file_idx,
                 specification const& spec)
    : origin_{parser_info{
          spec.filename_,
          {source_file_idx, spec.line_number_from_, spec.line_number_to_}}},
      num_repetitions_{
          parse<unsigned>(spec.internal_service_.substr(22, utl::size(3)))},
      interval_{
          parse<unsigned>(spec.internal_service_.substr(26, utl::size(3)))},
      stops_{utl::to_vec(spec.stops_, parse_stop)},
      initial_admin_{st.resolve_provider(initial_admin(spec))},
      initial_train_num_{initial_train_num(spec)} {
  utl::verify(stops_.size() >= 2, "service with less than 2 stops");

  parse_sections(st, spec, sections_, begin_to_end_info_);

  parse_range(spec.attributes_, c.attribute_parse_info_, stops_, sections_,
              begin_to_end_info_, &section::attributes_,
              [&](utl::cstr line, range const&) {
                return attribute{
                    parse<unsigned>(line.substr(c.s_info_.traff_days_)),
                    st.resolve_attribute(line.substr(c.s_info_.att_code_))};
              });

  parse_range(spec.categories_, c.category_parse_info_, stops_, sections_,
              begin_to_end_info_, &section::category_,
              [&](utl::cstr line, range const&) {
                return st.resolve_category(line.substr(c.s_info_.cat_));
              });

  parse_range(spec.line_information_, c.line_parse_info_, stops_, sections_,
              begin_to_end_info_, &section::line_,
              [&](utl::cstr line, range const&) {
                return line.substr(c.s_info_.line_).trim();
              });

  parse_range(
      spec.traffic_days_, c.traffic_days_parse_info_, stops_, sections_,
      begin_to_end_info_, &section::traffic_days_,
      [&](utl::cstr line, range const&) {
        return parse_verify<unsigned>(line.substr(c.s_info_.traff_days_));
      });

  parse_range(spec.directions_, c.direction_parse_info_, stops_, sections_,
              begin_to_end_info_, &section::direction_,
              [&](utl::cstr line, range const& r) {
                if (line[5] == ' ') {
                  return st.resolve_direction({stops_[r.to_idx()].eva_num_});
                } else {
                  return st.resolve_direction({line.substr(c.s_info_.dir_)});
                }
              });
}

std::string service::display_name(nigiri::timetable& tt) const {
  static auto const unknown_catergory = category{.name_ = "UKN",
                                                 .long_name_ = "UNKNOWN",
                                                 .output_rule_ = 0U,
                                                 .clasz_ = clasz::kOther};

  constexpr auto const kOnlyCategory = std::uint8_t{0b0001};
  constexpr auto const kOnlyTrainNr = std::uint8_t{0b0010};
  constexpr auto const kNoOutput = std::uint8_t{0b0011};
  constexpr auto const kUseProvider = std::uint8_t{0b1000};

  auto const& cat =
      *(begin_to_end_info_.category_.has_value()
            ? begin_to_end_info_.category_.value()
            : sections_.front().category_.value_or(&unknown_catergory));
  auto const& provider = tt.providers_.at(
      begin_to_end_info_.admin_.has_value() ? begin_to_end_info_.admin_.value()
                                            : sections_.front().admin_.value());
  auto const is = [&](auto const flag) {
    return (cat.output_rule_ & flag) == flag;
  };

  if (is(kNoOutput)) {
    return "";
  } else {
    auto const train_nr = initial_train_num_;
    auto const line_id =
        begin_to_end_info_.line_
            .value_or(sections_.empty() ? ""
                                        : sections_.front().line_.value_or(""))
            .to_str();
    auto const first =
        is(kOnlyTrainNr)
            ? string{""}
            : (is(kUseProvider) ? provider.short_name_ : cat.name_);
    auto const second =
        is(kOnlyCategory)
            ? string{""}
            : (train_nr == 0U ? line_id : fmt::to_string(train_nr));
    return fmt::format("{}{}{}", first, first.empty() ? "" : " ", second);
  }
}

}  // namespace nigiri::loader::hrd
