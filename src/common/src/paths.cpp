// src/common/src/paths.cpp

#include "paths.hpp"
#include <filesystem>
#include <stdexcept>
#include <string>

std::string getProjectDir() {
    // The macro PROJECT_SOURCE_DIR is defined in the root CMakeLists.txt
    // We need to convert it from a macro to a C++ string.
    #define STRINGIFY(x) #x
    #define TOSTRING(x) STRINGIFY(x)
    std::string path = TOSTRING(PROJECT_SOURCE_DIR);

    // The path might have extra quotes around it, so we remove them.
    path.erase(std::remove(path.begin(), path.end(), '\"'), path.end());

    return path;
}

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <unistd.h>
  #include <limits.h>
#endif

namespace fs = std::filesystem;

/* ------------------------------------------------------------------ */
std::string getExecutablePath() {
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (!n || n == MAX_PATH) throw std::runtime_error("GetModuleFileNameA failed.");
    return std::string(buf, n);
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, PATH_MAX);
    if (n == -1) throw std::runtime_error("readlink failed.");
    return std::string(buf, n);
#endif
}

// std::string getProjectDir()
// {
//     fs::path exeDir = fs::path(getExecutablePath()).parent_path();
//     return exeDir.parent_path().parent_path().parent_path().string();
// }