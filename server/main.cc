#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/logging.h"
#include "utl/helpers/algorithm.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::hrd;

int main(int ac, char** av) {
  if (ac != 2) {
    fmt::print("usage: {} [TIMETABLE_PATH]\n",
               ac == 0U ? "nigiri-server" : av[0]);
    return 1;
  }

  auto const src = source_idx_t{0U};
  auto const tt_path = std::filesystem::path{av[1]};
  auto const d = make_dir(tt_path);

  auto const c =
      utl::find_if(configs, [&](auto&& c) { return applicable(c, *d); });
  utl::verify(c != end(configs), "no loader applicable to {}", tt_path);
  log(log_lvl::info, "main",
      "loading nigiri timetable with configuration {}", c->version_.view());

  auto tt = std::make_shared<timetable>();
  load_timetable(src, *c, *d, *tt);

  std::cerr << "DONE! FIN\n";
}