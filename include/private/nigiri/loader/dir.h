#pragma once

#include <filesystem>
#include <memory>
#include <string_view>

namespace nigiri::loader {

struct file {
  struct content {
    virtual ~content() = default;
    virtual std::string_view get() = 0;
  };

  std::string name_;
  std::unique_ptr<content> content_;
};

struct dir {
  virtual ~dir();
  virtual file get_file(std::string_view path) = 0;
};

struct fs_dir final : public dir {
  ~fs_dir() override;
  file get_file(std::string_view path) override;
  std::filesystem::path path_;
};

struct zip_dir final : public dir {
  ~zip_dir() override;
  file get_file(std::string_view path) override;
};

struct mem_dir final : public dir {
  ~mem_dir() override;
  file get_file(std::string_view path) override;
};

}  // namespace nigiri::loader