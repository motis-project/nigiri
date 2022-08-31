#pragma once

#include <cassert>
#include <utility>
#include <variant>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/get_or_create.h"
#include "utl/overloaded.h"
#include "utl/parser/cstr.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/zip.h"

#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct parser_info {
  friend std::ostream& operator<<(std::ostream& out, parser_info const& pi);
  string str() const;
  char const* filename_;
  size_t line_number_from_;
  size_t line_number_to_;
};

struct specification {
  bool is_empty() const;
  bool valid() const;
  bool ignore() const;
  void reset();
  bool read_line(utl::cstr line,
                 char const* filename,
                 unsigned const line_number);

  char const* filename_{"unknown file"};
  size_t line_number_from_{0U};
  size_t line_number_to_{0U};
  utl::cstr internal_service_;
  std::vector<utl::cstr> traffic_days_;
  std::vector<utl::cstr> categories_;
  std::vector<utl::cstr> line_information_;
  std::vector<utl::cstr> attributes_;
  std::vector<utl::cstr> directions_;
  std::vector<utl::cstr> stops_;
};

struct service {
  static const constexpr auto kTimeNotSet = -1;  // NOLINT
  static category unknown_catergoy;
  static provider unknown_provider;

  struct event {
    int time_;
    bool in_out_allowed_;
  };

  struct stop {
    eva_number eva_num_;
    event arr_, dep_;
  };

  struct attribute {
    CISTA_COMPARABLE()

    attribute(int bitfield_num, utl::cstr code)
        : bitfield_num_{bitfield_num}, code_{code} {}

    int bitfield_num_;
    utl::cstr code_;
  };

  using direction_info =
      variant<utl::cstr /* custom string */, eva_number /* eva number */>;

  struct section {
    section() = default;
    section(int train_num, utl::cstr admin)
        : train_num_{train_num}, admin_{admin} {}

    int train_num_{0};
    utl::cstr admin_;
    std::vector<attribute> attributes_;
    std::vector<utl::cstr> category_;
    std::vector<utl::cstr> line_information_;
    std::vector<direction_info> directions_;
    std::vector<unsigned> traffic_days_;
    clasz clasz_{clasz::kAir};
  };

  service(service const& s, vector<duration_t> const& times, bitfield const& b)
      : service{s} {
    traffic_days_ = b;
    set_stop_times(times);
  }

  service(parser_info origin,
          int num_repetitions,
          int interval,
          std::vector<stop> stops,
          std::vector<section> sections,
          bitfield traffic_days,
          utl::cstr initial_admin,
          int initial_train_num);

  service(config const&, specification const&);

  void verify_service();

  std::vector<std::pair<int, utl::cstr>> get_ids() const;

  int find_first_stop_at(eva_number) const;

  int get_first_stop_index_at(eva_number) const;

  int event_time(unsigned stop_index, event_type evt) const;

  unsigned traffic_days_offset_at_stop(unsigned stop_index, event_type) const;

  bitfield traffic_days_at_stop(unsigned stop_index, event_type) const;

  vector<duration_t> get_stop_times() const;
  void set_stop_times(vector<duration_t> const&);
  vector<tz_offsets> get_stop_timezones(timezone_map_t const&) const;

  string display_name(timetable& tt,
                      category_map_t const& categories,
                      provider_map_t const& providers) const {
    try {
      constexpr auto const kOnlyCategory = std::uint8_t{0b0001};
      constexpr auto const kOnlyTrainNr = std::uint8_t{0b0010};
      constexpr auto const kNoOutput = std::uint8_t{0b0011};
      constexpr auto const kUseProvider = std::uint8_t{0b1000};

      auto const& cat = [&]() {
        auto const it =
            categories.find(sections_.front().category_.front().to_str());
        if (it != end(categories)) {
          return it->second;
        } else {
          log(log_lvl::error, "loader.hrd.service.name",
              "service {}: invalid category \"{}\"", origin_,
              sections_.front().category_.front().view());
          return unknown_catergoy;
        }
      }();

      auto const& provider = [&]() {
        auto const it = providers.find(initial_admin_.view());
        if (it != end(providers)) {
          return tt.providers_.at(it->second);
        } else {
          log(log_lvl::error, "loader.hrd.service.name",
              "service {}: invalid provider {}", origin_,
              initial_admin_.view());
          return unknown_provider;
        }
      }();

      auto const is = [&](auto const flag) {
        return (cat.output_rule_ & flag) == flag;
      };

      if (is(kNoOutput)) {
        return "";
      } else {
        auto const train_nr = initial_train_num_;
        auto const line_id =
            sections_.front().line_information_.empty()
                ? ""
                : sections_.front().line_information_.front().to_str();
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
    } catch (std::exception const& e) {
      log(log_lvl::error, "nigiri.loader.hrd.service.name",
          "unable to build service display name for {}: {}", origin_, e.what());
      return "";
    }
  }

  void set_sections_clasz();

  parser_info origin_{};
  int num_repetitions_{0};
  int interval_{0};
  std::vector<stop> stops_;
  std::vector<section> sections_;
  bitfield traffic_days_;
  utl::cstr initial_admin_;
  int initial_train_num_{0};
};

template <typename ConsumerFn, typename ProgressFn>
void parse_services(config const& c,
                    char const* filename,
                    interval<std::chrono::sys_days> const& interval,
                    bitfield_map_t const& bitfields,
                    timezone_map_t const& timezones,
                    std::string_view file_content,
                    ProgressFn&& bytes_consumed,
                    ConsumerFn&& consumer) {
  auto const expand_service = [&](service const& s) {
    expand_traffic_days(s, bitfields, [&](service&& s2) {
      expand_repetitions(s2, [&](service&& s3) {
        to_local_time(timezones, interval, s3, consumer);
      });
    });
  };

  specification spec;
  auto last_line = 0U;
  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, unsigned const line_number) {
        last_line = line_number;

        if (line_number % 1000 == 0) {
          bytes_consumed(static_cast<size_t>(line.c_str() - &file_content[0]));
        }

        if (line.len == 0 || line[0] == '%') {
          return;
        }

        auto const is_finished = spec.read_line(line, filename, line_number);

        if (!is_finished) {
          return;
        } else {
          spec.line_number_to_ = line_number - 1;
        }

        if (!spec.valid()) {
          log(log_lvl::error, "loader.hrd.service",
              "skipping invalid service at {}:{}", filename, line_number);
        } else if (!spec.ignore()) {
          // Store if relevant.
          try {
            expand_service(service{c, spec});
          } catch (std::runtime_error const& e) {
            log(log_lvl::error, "unable to build service at {}:{}: {}",
                filename, line_number, e.what());
          }
        }

        // Next try! Re-read first line of next service.
        spec.reset();
        spec.read_line(line, filename, line_number);
      });

  if (!spec.is_empty() && spec.valid() && !spec.ignore()) {
    spec.line_number_to_ = last_line;
    expand_service(service{c, spec});
  }
}

}  // namespace nigiri::loader::hrd
