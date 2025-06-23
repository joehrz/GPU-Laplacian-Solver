// common/src/config_loader.cpp

#include "simulation_config.h"          
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

/* ────────────────────────────────────────────────────────────── */
/*  loadConfiguration                                             */
/* ────────────────────────────────────────────────────────────── */
FullConfig loadConfiguration(const std::string& filename)
{
    FullConfig config;   // populated with defaults below

    /* ---------- sensible built-in defaults -------------------- */
    config.bc.left   = 0.0f;
    config.bc.right  = 0.0f;
    config.bc.top    = 0.0f;
    config.bc.bottom = 100.0f;

    config.sim_params.width          = 256;
    config.sim_params.height         = 256;
    config.sim_params.tolerance      = 1e-5f;
    config.sim_params.max_iterations = 10000;
    config.sim_params.omega          = 1.9f;

    /* ---------- try to open and parse JSON -------------------- */
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open configuration file: "
                  << filename << "\nUsing built-in defaults.\n";
        return config;
    }

    try {
        json j; file >> j;

        /* ---- boundary_conditions ----------------------------- */
        if (j.contains("boundary_conditions")) {
            const auto& bcj = j["boundary_conditions"];
            config.bc.left   = bcj.value<float>("left",   config.bc.left);
            config.bc.right  = bcj.value<float>("right",  config.bc.right);
            config.bc.top    = bcj.value<float>("top",    config.bc.top);
            config.bc.bottom = bcj.value<float>("bottom", config.bc.bottom);
        }

        /* ---- simulation_parameters --------------------------- */
        if (j.contains("simulation_parameters")) {
            const auto& sp = j["simulation_parameters"];
            config.sim_params.width          = sp.value<int>("width",          config.sim_params.width);
            config.sim_params.height         = sp.value<int>("height",         config.sim_params.height);
            config.sim_params.tolerance      = sp.value<float>("tolerance",      config.sim_params.tolerance);
            config.sim_params.max_iterations = sp.value<int>("max_iterations", config.sim_params.max_iterations);
            config.sim_params.omega          = sp.value<float>("omega",          config.sim_params.omega);
        }
    }
    catch (const json::parse_error& e) {
        std::cerr << "JSON parse error in " << filename << ": " << e.what()
                  << "\nUsing built-in defaults.\n";
    }
    catch (const json::exception& e) {
        std::cerr << "JSON data error in " << filename << ": " << e.what()
                  << "\nUsing built-in defaults (partial).\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Generic error loading " << filename << ": " << e.what()
                  << "\nUsing built-in defaults (partial).\n";
    }

    /* ---------- echo final configuration ---------------------- */
    std::cout << "Configuration loaded from " << filename << ":\n"
              << "  BCs: L=" << config.bc.left
              << ", R=" << config.bc.right
              << ", T=" << config.bc.top
              << ", B=" << config.bc.bottom << '\n'
              << "  Sim Params: W=" << config.sim_params.width
              << ", H=" << config.sim_params.height
              << ", Tol=" << config.sim_params.tolerance
              << ", MaxIter=" << config.sim_params.max_iterations
              << ", Omega=" << config.sim_params.omega << '\n';

    return config;
}









