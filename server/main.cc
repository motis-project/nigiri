#include <vector>

#include "date/date.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/logging.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;

struct loader_interface {
  virtual ~loader_interface() = default;
  virtual bool applicable(dir const&) const = 0;
  virtual void load(source_idx_t, dir const&, timetable&) const = 0;
  virtual std::string_view name() const = 0;
};

struct hrd_loader : public loader_interface {
  explicit hrd_loader(nigiri::loader::hrd::config c) : config_{std::move(c)} {}
  bool applicable(dir const& d) const override {
    return nigiri::loader::hrd::applicable(config_, d);
  }
  void load(source_idx_t const src,
            dir const& d,
            timetable& tt) const override {
    return nigiri::loader::hrd::load_timetable(src, config_, d, tt);
  }
  nigiri::loader::hrd::config config_;
};

struct hrd_5_00_8_loader : public hrd_loader {
  hrd_5_00_8_loader() : hrd_loader{nigiri::loader::hrd::hrd_5_00_8} {}
  std::string_view name() const override { return "hrd_5_00_8"; }
};

struct hrd_5_20_26_loader : public hrd_loader {
  hrd_5_20_26_loader() : hrd_loader{nigiri::loader::hrd::hrd_5_00_8} {}
  std::string_view name() const override { return "hrd_5_20_26"; }
};

struct hrd_5_20_39_loader : public hrd_loader {
  hrd_5_20_39_loader() : hrd_loader{nigiri::loader::hrd::hrd_5_00_8} {}
  std::string_view name() const override { return "hrd_5_20_39"; }
};

struct hrd_5_20_avv_loader : public hrd_loader {
  hrd_5_20_avv_loader() : hrd_loader{nigiri::loader::hrd::hrd_5_00_8} {}
  std::string_view name() const override { return "hrd_5_20_avv"; }
};

struct gtfs_loader : public loader_interface {
  bool applicable(dir const& d) const override {
    return nigiri::loader::gtfs::applicable(d);
  }
  void load(source_idx_t const src,
            dir const& d,
            timetable& tt) const override {
    return nigiri::loader::gtfs::load_timetable(src, d, tt);
  }
  std::string_view name() const override { return "gtfs"; }
};

int main(int ac, char** av) {
  if (ac != 2) {
    fmt::print("usage: {} [TIMETABLE_PATH]\n",
               ac == 0U ? "nigiri-server" : av[0]);
    return 1;
  }

  auto loaders = std::vector<std::unique_ptr<loader_interface>>{};
  loaders.emplace_back(std::make_unique<gtfs_loader>());
  loaders.emplace_back(std::make_unique<hrd_5_00_8_loader>());
  loaders.emplace_back(std::make_unique<hrd_5_20_26_loader>());
  loaders.emplace_back(std::make_unique<hrd_5_20_39_loader>());
  loaders.emplace_back(std::make_unique<hrd_5_20_avv_loader>());

  auto const src = source_idx_t{0U};
  auto const tt_path = std::filesystem::path{av[1]};
  auto const d = make_dir(tt_path);

  auto const c =
      utl::find_if(loaders, [&](auto&& l) { return l->applicable(*d); });
  utl::verify(c != end(loaders), "no loader applicable to {}", tt_path);
  log(log_lvl::info, "main", "loading nigiri timetable with configuration {}",
      (*c)->name());

  timetable tt;
  tt.date_range_ = {date::sys_days{April / 1 / 2023},
                    date::sys_days{December / 1 / 2023}};
  (*c)->load(src, *d, tt);
}