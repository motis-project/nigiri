#pragma once

#include <memory>

namespace nigiri::algo {

struct statistics {};

struct journey {};

struct journey_response {
  statistics stats;
  std::vector<journey> journeys_;
};

struct data {};

struct data_response {
  statistics stats;
  std::unique_ptr<data> data_;
};

}  // namespace nigiri::algo