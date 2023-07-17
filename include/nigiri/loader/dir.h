#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string_view>
#include <vector>

namespace nigiri::loader {

enum class dir_type { kFilesystem, kZip, kInMemory };

struct file {
  struct content {
    virtual ~content();
    virtual std::string_view get() const = 0;
  };

  bool has_value() const noexcept { return content_ != nullptr; }
  std::string_view data() const {
    return content_ == nullptr ? "" : content_->get();
  }
  char const* filename() const { return name_.c_str(); }

  std::string name_;
  std::unique_ptr<content> content_;
};

struct dir {
  dir(std::filesystem::path);
  dir(dir const&);
  dir(dir&&) noexcept;
  dir& operator=(dir const&);
  dir& operator=(dir&&) noexcept;
  virtual ~dir();
  virtual std::vector<std::filesystem::path> list_files(
      std::filesystem::path const&) const = 0;
  virtual file get_file(std::filesystem::path const&) const = 0;
  virtual bool exists(std::filesystem::path const&) const = 0;
  virtual std::size_t file_size(std::filesystem::path const&) const = 0;
  virtual dir_type type() const = 0;
  virtual std::uint64_t hash() const = 0;
  std::filesystem::path path() const { return path_; }

protected:
  std::filesystem::path path_;
};

struct fs_dir final : public dir {
  explicit fs_dir(std::filesystem::path);
  ~fs_dir() final;
  std::vector<std::filesystem::path> list_files(
      std::filesystem::path const&) const final;
  file get_file(std::filesystem::path const&) const final;
  bool exists(std::filesystem::path const&) const final;
  std::size_t file_size(std::filesystem::path const&) const final;
  dir_type type() const final;
  std::uint64_t hash() const final;
};

struct zip_dir final : public dir {
  ~zip_dir() final;
  explicit zip_dir(std::filesystem::path const&);
  explicit zip_dir(std::vector<std::uint8_t>);
  std::vector<std::filesystem::path> list_files(
      std::filesystem::path const&) const final;
  file get_file(std::filesystem::path const&) const final;
  bool exists(std::filesystem::path const&) const final;
  std::size_t file_size(std::filesystem::path const&) const final;
  dir_type type() const final;
  std::uint64_t hash() const final;
  struct impl;
  std::unique_ptr<impl> impl_;
};

struct mem_dir final : public dir {
  using dir_t = std::map<std::filesystem::path, std::string>;
  static mem_dir read(std::string_view);
  mem_dir(dir_t);
  ~mem_dir() final;
  mem_dir(mem_dir const&);
  mem_dir(mem_dir&&) noexcept;
  mem_dir& operator=(mem_dir const&);
  mem_dir& operator=(mem_dir&&) noexcept;
  mem_dir& add(std::pair<std::filesystem::path, std::string>);
  std::vector<std::filesystem::path> list_files(
      std::filesystem::path const&) const final;
  file get_file(std::filesystem::path const&) const final;
  bool exists(std::filesystem::path const&) const final;
  std::size_t file_size(std::filesystem::path const&) const final;
  dir_type type() const final;
  std::uint64_t hash() const final;
  dir_t dir_;
};

std::unique_ptr<dir> make_dir(std::filesystem::path const& p);

}  // namespace nigiri::loader
