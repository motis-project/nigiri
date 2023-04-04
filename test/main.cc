#include <filesystem>

#include "gtest/gtest.h"

#include "utl/progress_tracker.h"

#include "test_dir.h"

#ifdef PROTOBUF_LINKED
#include "google/protobuf/stubs/common.h"
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  utl::get_active_progress_tracker_or_activate("test");
  fs::current_path(NIGIRI_TEST_EXECUTION_DIR);

  ::testing::InitGoogleTest(&argc, argv);
  auto test_result = RUN_ALL_TESTS();

  return test_result;
}
