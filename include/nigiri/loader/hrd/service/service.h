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
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

struct parser_info {
  friend std::ostream& operator<<(std::ostream& out, parser_info const& pi);
  char const* filename_;
  trip_debug dbg_;
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
  std::uint32_t line_number_from_{0U};
  std::uint32_t line_number_to_{0U};
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
  static category unknown_catergory;
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

    attribute(unsigned bitfield_num, attribute_idx_t const code)
        : bitfield_num_{bitfield_num}, code_{code} {}

    unsigned bitfield_num_;
    attribute_idx_t code_;
  };

  struct section {
    section() = default;
    section(int train_num, provider_idx_t admin)
        : train_num_{train_num}, admin_{admin} {}

    int train_num_{0};
    provider_idx_t admin_;
    std::vector<attribute> attributes_;
    category const* category_{nullptr};
    utl::cstr line_information_;
    trip_direction_idx_t direction_{trip_direction_idx_t::invalid()};
    unsigned traffic_days_;
  };

  service(config const&, stamm&, source_file_idx_t, specification const&);

  void verify_service();

  std::vector<std::pair<int, provider_idx_t>> get_ids() const;

  int find_first_stop_at(eva_number) const;

  int get_first_stop_index_at(eva_number) const;

  int event_time(unsigned stop_index, event_type evt) const;

  unsigned traffic_days_offset_at_stop(unsigned stop_index, event_type) const;

  bitfield traffic_days_at_stop(unsigned stop_index, event_type) const;

  vector<duration_t> get_stop_times() const;
  void set_stop_times(vector<duration_t> const&);
  vector<tz_offsets> get_stop_timezones(timezone_map_t const&) const;

  string display_name(timetable& tt) const {
    static auto const unknown_catergory = category{.name_ = "UKN",
                                                   .long_name_ = "UNKNOWN",
                                                   .output_rule_ = 0U,
                                                   .clasz_ = clasz::kOther};

    constexpr auto const kOnlyCategory = std::uint8_t{0b0001};
    constexpr auto const kOnlyTrainNr = std::uint8_t{0b0010};
    constexpr auto const kNoOutput = std::uint8_t{0b0011};
    constexpr auto const kUseProvider = std::uint8_t{0b1000};

    auto const& cat = sections_.front().category_ == nullptr
                          ? unknown_catergory
                          : *sections_.front().category_;
    auto const& provider = tt.providers_.at(initial_admin_);
    auto const is = [&](auto const flag) {
      return (cat.output_rule_ & flag) == flag;
    };

    if (is(kNoOutput)) {
      return "";
    } else {
      auto const train_nr = initial_train_num_;
      auto const line_id = sections_.front().line_information_.to_str();
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

  parser_info origin_{};
  unsigned num_repetitions_{0U};
  unsigned interval_{0U};
  std::vector<stop> stops_;
  std::vector<section> sections_;
  bitfield traffic_days_;
  provider_idx_t initial_admin_;
  int initial_train_num_{0};
};

}  // namespace nigiri::loader::hrd
