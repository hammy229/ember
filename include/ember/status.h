#pragma once

#include <string>
#include <string_view>

namespace ember {

enum class StatusCode {
  kOk,
  kNotFound,
  kInvalidFormat,
  kInvalidArgument,
  kOutOfMemory,
};

// Result of an operation that can fail, such as opening a model file.
// A default-constructed Status is OK. Ignoring a returned Status is a compile error here,
// because the build treats warnings as errors.
class [[nodiscard]] Status {
 public:
  Status() = default;

  static Status Ok() { return Status(); }
  static Status NotFound(std::string_view message);
  static Status InvalidFormat(std::string_view message);
  static Status InvalidArgument(std::string_view message);
  static Status OutOfMemory(std::string_view message);

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

  // "OK", or "CODE_NAME: message" ("CODE_NAME" when the message is empty).
  std::string ToString() const;

 private:
  Status(StatusCode code, std::string_view message) : code_(code), message_(message) {}

  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

}  // namespace ember
