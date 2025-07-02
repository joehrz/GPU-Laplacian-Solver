// src/common/src/config_loader.cpp

#include "simulation_config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

FullConfig loadConfiguration(const std::string& file)
{
    FullConfig cfg;                   // defaults already set in struct
    std::ifstream f(file);
    if(!f.is_open()){
        std::cerr<<"[Config] "<<file<<" not found; using defaults\n";
        return cfg;
    }
    try{
        json j; f>>j;
        if(j.contains("boundary_conditions")){
            auto b=j["boundary_conditions"];
            cfg.bc.left   = b.value("left",   cfg.bc.left);
            cfg.bc.right  = b.value("right",  cfg.bc.right);
            cfg.bc.top    = b.value("top",    cfg.bc.top);
            cfg.bc.bottom = b.value("bottom", cfg.bc.bottom);
        }
        if(j.contains("simulation_parameters")){
            auto s=j["simulation_parameters"];
            cfg.sim_params.width          = s.value("width",          cfg.sim_params.width);
            cfg.sim_params.height         = s.value("height",         cfg.sim_params.height);
            cfg.sim_params.tolerance      = s.value("tolerance",      cfg.sim_params.tolerance);
            cfg.sim_params.max_iterations = s.value("max_iterations", cfg.sim_params.max_iterations);
            cfg.sim_params.omega          = s.value("omega",          cfg.sim_params.omega);
        }
    }catch(const json::exception& e){
        std::cerr<<"[Config] JSON error: "<<e.what()<<"\nUsing defaults\n";
    }
    return cfg;
}