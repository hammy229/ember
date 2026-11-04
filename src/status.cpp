#include "ember/status.h"

namespace ember {

namespace {

const char* CodeName(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return "OK";
    case StatusCode::kNotFound:
      return "NOT_FOUND";
    case StatusCode::kInvalidFormat:
      return "INVALID_FORMAT";
    case StatusCode::kInvalidArgument:
      return "INVALID_ARGUMENT";
  }
  return "UNKNOWN";
}

}  // namespace

Status Status::NotFound(std::string_view message) { return Status(StatusCode::kNotFound, message); }

Status Status::InvalidFormat(std::string_view message) {
  return Status(StatusCode::kInvalidFormat, message);
}

Status Status::InvalidArgument(std::string_view message) {
  return Status(StatusCode::kInvalidArgument, message);
}

std::string Status::ToString() const {
  std::string out = CodeName(code_);
  if (!message_.empty()) {
    out += ": ";
    out += message_;
  }
  return out;
}

}  // namespace ember
