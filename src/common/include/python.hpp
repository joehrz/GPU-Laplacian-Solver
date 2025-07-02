// src/common/include/python.hpp
#pragma once
#include <string>

std::string getPythonCommand();              // find an interpreter
inline std::string q(const std::string& s) { return (s.find_first_of(" \t") == std::string::npos) ? s : '"' + s + '"'; }

void plot_solution(const std::string& python,
                   const std::string& tag,
                   const std::string& csv_path);