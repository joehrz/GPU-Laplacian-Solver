// src/common/src/python.cpp

#include "python.hpp"
#include "paths.hpp"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
  static bool cmd_exists(const std::string& cmd)
  { return system(("where " + cmd + " >nul 2>&1").c_str()) == 0; }
#else
  static bool cmd_exists(const std::string& cmd)
  { return system(("which " + cmd + " >/dev/null 2>&1").c_str()) == 0; }
#endif

/* ------------------------------------------------------------------ */
std::string getPythonCommand()
{
#ifdef _WIN32
    std::vector<std::string> cand = {
        R"(C:\Python312\python.exe)", R"(C:\Python311\python.exe)",
        R"(C:\Python310\python.exe)", R"(C:\Python39\python.exe)",
        "python.exe", "python3.exe"
    };
#else
    std::vector<std::string> cand = {"python3", "python"};
#endif
    for(const auto& p : cand)
    {
#ifdef _WIN32
        bool ok = (p.find(':') != std::string::npos)
                    ? std::filesystem::exists(p) : cmd_exists(p);
#else
        bool ok = cmd_exists(p);
#endif
        if(ok) { std::cerr << "[Debug] Using Python: " << p << '\n'; return p; }
    }
    throw std::runtime_error("Python interpreter not found.");
}

void plot_solution(const std::string& python,
                   const std::string& tag,
                   const std::string& csv)
{
    const std::string script =
        (std::filesystem::path(getProjectDir())/"scripts"/"plot_solution.py").string();

    std::string cmd = q(python) + ' ' + q(script) + ' ' + q(tag) + ' ' + q(csv);

    std::cerr << "[Debug] " << cmd << '\n';
    system(cmd.c_str());
}