#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"

namespace nigiri::loader::gtfs {

bool is_based(std::span<stop_idx_t> in,
              stop_idx_t const start,
              stop_idx_t const inc) {
  for (auto x = start, i = stop_idx_t{0U}; i != in.size(); x += inc, ++i) {
    if (in[i] != x) {
      return false;
    }
  }
  return true;
}

void encode_seq_numbers(std::span<stop_idx_t> in,
                        std::basic_string<stop_idx_t>& out) {
  out.clear();
  if (is_based(in, 0U, 1U)) {
    return;
  } else if (is_based(in, 1U, 1U)) {
    out.push_back(1);
  } else if (is_based(in, 10U, 10U)) {
    out.push_back(10);
  } else {
    out.resize(in.size());
    std::copy(begin(in), end(in), begin(out));
  }
}

}  // namespace nigiri::loader::gtfs