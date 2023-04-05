#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/transfer.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;

stop_pair t(stop_map const& stops,
            std::string const& s1,
            std::string const& s2) {
  return std::pair{stops.at(s1).get(), stops.at(s2).get()};
}

TEST(gtfs, read_transfers_example_data) {
  auto const stops = read_stops(example_files().get_file(kStopFile).data());
  auto const transfers =
      read_transfers(stops, example_files().get_file(kTransfersFile).data());

  EXPECT_EQ(2U, transfers.size());

  EXPECT_EQ(5_minutes, transfers.at(t(stops, "S6", "S7")).minutes_);
  EXPECT_EQ(transfer::type::kMinimumChangeTime,
            transfers.at(t(stops, "S6", "S7")).type_);
  EXPECT_EQ(transfer::type::kNotPossible,
            transfers.at(t(stops, "S7", "S6")).type_);
}
