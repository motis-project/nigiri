#pragma once

#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {

  struct td_agency {
    std::string name_;
    std::string tz_name_;
    std::string language_;
    std::string phone_number_;
    std::string url_;
  };

  using td_agency_map_t = hash_map<std::string, std::unique_ptr<td_agency>>;

  td_agency_map_t read_td_agencies(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex
