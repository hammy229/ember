#pragma once

#include <cstddef>
#include <span>
#include <string>

#include "ember/status_or.h"

namespace ember {

// A whole file mapped read-only into memory. The operating system pages it in as it is read, so
// opening a model costs almost nothing until the weights are touched, and the pages are shared with
// any other process that maps the same file.
//
// If another program shortens the file while it is mapped, reading a page past the new end kills
// the process with SIGBUS. That is how memory mapping works everywhere, and the fix is not to edit
// a model file while something is running it.
//
// The pages are read-only. Writing through a pointer into them kills the process, which is why the
// loader only ever hands them out as TensorView. Move-only: the mapping goes away with the object.
class MappedFile {
 public:
  // Fails with kNotFound if the path does not exist, and kInvalidArgument if it is not a regular
  // file or is too big to map (only possible on 32-bit WebAssembly). An empty file opens fine and
  // has no bytes.
  static StatusOr<MappedFile> Open(const std::string& path);

  MappedFile() = default;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  std::span<const std::byte> bytes() const { return {data_, size_}; }
  std::size_t size() const { return size_; }
  // Tells the system these bytes of the file will not be needed again soon, so it can drop their
  // pages. Reading them later still works; they come back from disk.
  void Release(std::span<const std::byte> part) const;

 private:
  MappedFile(const std::byte* data, std::size_t size) : data_(data), size_(size) {}
  void Unmap();

  const std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace ember
