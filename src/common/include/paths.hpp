// src/common/include/paths.hpp

#pragma once
#include <string>

std::string getExecutablePath();  // absolute path to this binary
std::string getProjectDir();      // repo root (…/build/*/ → ../..)