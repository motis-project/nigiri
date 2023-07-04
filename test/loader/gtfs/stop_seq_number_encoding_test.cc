#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/common/it_range.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, stop_seq_number_encoding) {
  auto const check = [](std::basic_string<stop_idx_t> seq) {
    std::basic_string<stop_idx_t> out;
    encode_seq_numbers(seq, out);

    std::basic_string<stop_idx_t> test;
    for (auto const x :
         stop_seq_number_range{out, static_cast<stop_idx_t>(seq.size())}) {
      test.push_back(x);
    }
    EXPECT_EQ(test, seq);

    test.clear();
    auto const r =
        stop_seq_number_range{out, static_cast<stop_idx_t>(seq.size())};
    for (auto const x : it_range{std::next(begin(r), 2), end(r)}) {
      test.push_back(x);
    }
    EXPECT_EQ(test, seq.substr(2));
  };

  check({0, 1, 2, 3, 4});
  check({1, 2, 3, 4});
  check({10, 20, 30, 40});

  check({0, 1, 2, 3, 4, 6});
  check({1, 2, 3, 4, 6});
  check({10, 20, 30, 40, 60});

  check({5, 4, 3, 2, 1, 0});
  check({4, 3, 2, 1});
  check({10, 80});
}
