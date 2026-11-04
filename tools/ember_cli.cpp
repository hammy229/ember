#include <iostream>

#include "ember/version.h"

int main() {
  std::cout << "ember " << ember::version() << '\n';
  return 0;
}
