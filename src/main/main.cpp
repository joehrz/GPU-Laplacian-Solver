// src/main/main.cpp

#include "simulation_config.h"
#include "grid_initialization.h"
#include "laplace_analytical_solution.h"
#include "solution_export.h"
#include "paths.hpp"
#include "python.hpp"
#include "timers.hpp"
#include "solver_registry.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <cassert>

namespace fs = std::filesystem;

// ---------- helpers --------------------------------------------------------

static float l2_error(const std::vector<float>& num,
                      const std::vector<float>& exact,
                      int W, int H)
{
    double s2 = 0.0;
    int n = 0;
    for (int j = 1; j < H - 1; ++j)
        for (int i = 1; i < W - 1; ++i)
        {
            float e = exact[j * W + i];
            if (std::abs(e) < 1e-12f) continue;
            double d = num[j * W + i] - e;
            s2 += d * d;
            ++n;
        }
    return (n > 0) ? static_cast<float>(std::sqrt(s2 / n)) : 0.0f;
}

// Expand “all” into the maximum set of tags we *might* want; we will
// verify availability at runtime.
static std::vector<std::string> expand_tag(const std::string& tag)
{
    if (tag == "all")
        return { "basic_cpu",
                 "red_black_cpu",
                 "basic_cuda",
                 "shared_cuda" };
    return { tag };
}

// ---------- main -----------------------------------------------------------

int main(int argc, char* argv[])
{
    try
    {
        // -------------------------------------------------------------------
        // 1. Parse CLI
        // -------------------------------------------------------------------
        std::string cfg_path =
            (fs::path(getProjectDir()) / "config" / "simulation_config.json").string();
        std::string tag_requested = "all";

        if (argc > 1)
        {
            std::string arg1 = argv[1];
            const std::vector<std::string> known =
                { "all", "basic_cpu", "red_black_cpu", "basic_cuda", "shared_cuda" };

            if (std::find(known.begin(), known.end(), arg1) == known.end())
            {
                cfg_path = std::move(arg1);            // custom config file
                if (argc > 2) tag_requested = argv[2];
            }
            else
            {
                tag_requested = std::move(arg1);       // first arg was a tag
            }
        }

        // -------------------------------------------------------------------
        // 2. Load configuration
        // -------------------------------------------------------------------
        FullConfig C = loadConfiguration(cfg_path);
        const int W = C.sim_params.width;
        const int H = C.sim_params.height;

        std::vector<float> U_numerical(W * H, 0.0f);      // host grid
        UniversalFourierSolution exact_solution(C.bc, 50);
        std::vector<float> U_exact = exact_solution.compute(W, H);

        const std::string python_cmd = getPythonCommand();
        const fs::path    out_dir    = fs::path(getProjectDir()) / "results";
        fs::create_directories(out_dir);

        std::cout << "--- Running Laplace solver(s) ---\n";

        // -------------------------------------------------------------------
        // 3. Loop over each candidate tag, but skip if unavailable
        // -------------------------------------------------------------------
        for (const std::string& tag : expand_tag(tag_requested))
        {
            // Initialise host grid first (so constructors copy BC-filled data)
            initializeGrid(U_numerical.data(), W, H, C.bc);

            std::unique_ptr<Solver> solver;

            try
            {
                auto vec = make_solvers(tag, U_numerical.data(), W, H);
                if (vec.empty())
                    throw std::runtime_error("registry returned 0 solvers");

                assert(vec.size() == 1);               // option-B creates one
                solver = std::move(vec.front());
            }
            catch (const std::exception& e)
            {
                std::cout << "Skipping tag " << tag
                          << "not available in this build ("
                          << e.what() << ").\n";
                continue;                              // try next tag
            }

            // -------------------- run the solver ---------------------------
            DefaultTimer timer;
            timer.start();
            solver->solve(C.sim_params);
            double seconds = timer.stop();

            std::cout << "\n[" << solver->getName() << "] execution finished.\n";
            std::cout << "  Timing: " << seconds * 1000.0 << " ms\n";

            const fs::path csv_path =
                out_dir / ("solution_" + solver->getName() + ".csv");

            if (solver->isOnDevice())
            {
                exportDeviceSolutionToCSV(solver->deviceData(),
                                          W, H, csv_path.string(),
                                          solver->getName());
            }
            else
            {
                exportHostDataToCSV(U_numerical.data(),
                                    W, H, csv_path.string(),
                                    solver->getName());
                std::cout << "  L2 error (vs. analytical): "
                          << l2_error(U_numerical, U_exact, W, H) << '\n';
            }

            plot_solution(python_cmd, solver->getName(), csv_path.string());
        }

        std::cout << "\n--- All tasks complete ---\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}