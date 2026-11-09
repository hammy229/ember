#include "ember/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace ember {

namespace {

// Closes the descriptor on every way out of Open. The mapping stays valid after the close.
class FileDescriptor {
 public:
  explicit FileDescriptor(int fd) : fd_(fd) {}
  ~FileDescriptor() {
    if (fd_ >= 0) ::close(fd_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  int get() const { return fd_; }

 private:
  int fd_;
};

std::string Describe(const std::string& path, int error) {
  return path + ": " + std::strerror(error);
}

}  // namespace

StatusOr<MappedFile> MappedFile::Open(const std::string& path) {
  const FileDescriptor fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
  if (fd.get() < 0) {
    const int error = errno;
    if (error == ENOENT) return Status::NotFound(Describe(path, error));
    return Status::InvalidArgument(Describe(path, error));
  }

  struct stat info {};
  if (::fstat(fd.get(), &info) != 0) return Status::InvalidArgument(Describe(path, errno));
  if (!S_ISREG(info.st_mode)) return Status::InvalidArgument(path + ": not a regular file");

  // off_t is 64 bits even where size_t is 32, so a big file could wrap around in the cast.
  const auto file_size = static_cast<std::uintmax_t>(info.st_size);
  if (file_size > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument(path + ": too big to map on this machine");
  }
  const auto size = static_cast<std::size_t>(file_size);
  // mmap refuses a length of zero, and an empty file has nothing to map anyway.
  if (size == 0) return MappedFile();

  void* data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.get(), 0);
  if (data == MAP_FAILED) return Status::InvalidArgument(Describe(path, errno));
  return MappedFile(static_cast<const std::byte*>(data), size);
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    Unmap();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

MappedFile::~MappedFile() { Unmap(); }

void MappedFile::Release(std::span<const std::byte> part) const {
  if (data_ == nullptr || part.empty()) return;
  // Whole pages only: round the start up and the end down.
  const auto page = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
  auto begin = reinterpret_cast<std::uintptr_t>(part.data());
  auto end = begin + part.size();
  begin = (begin + page - 1) / page * page;
  end = end / page * page;
  if (end > begin) {
    ::madvise(reinterpret_cast<void*>(begin), end - begin, MADV_DONTNEED);
  }
}

void MappedFile::Unmap() {
  if (data_ != nullptr) ::munmap(const_cast<std::byte*>(data_), size_);
  data_ = nullptr;
  size_ = 0;
}

}  // namespace ember
