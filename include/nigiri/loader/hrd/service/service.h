#pragma once

#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/parser/cstr.h"

#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
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
  bool read_line(utl::cstr line, char const* filename, unsigned line_number);

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
        : bitfield_num_{bitfield_num}, idx_{code} {}

    unsigned bitfield_num_;
    attribute_idx_t idx_;
  };

  struct section {
    section() = default;
    section(int train_num, provider_idx_t admin)
        : train_num_{train_num}, admin_{admin} {}

    std::optional<std::uint32_t> train_num_;
    std::optional<provider_idx_t> admin_;
    std::optional<std::vector<attribute>> attributes_;
    std::optional<category const*> category_;
    std::optional<utl::cstr> line_;
    std::optional<trip_direction_idx_t> direction_;
    std::optional<unsigned> traffic_days_;
  };

  service(config const&, stamm&, source_file_idx_t, specification const&);

  std::string display_name(timetable& tt) const;

  parser_info origin_{};
  unsigned num_repetitions_{0U};
  unsigned interval_{0U};
  std::vector<stop> stops_;
  std::vector<section> sections_;
  section begin_to_end_info_;
  bitfield traffic_days_;
  provider_idx_t initial_admin_;
  std::uint32_t initial_train_num_{0U};
};

}  // namespace nigiri::loader::hrd
