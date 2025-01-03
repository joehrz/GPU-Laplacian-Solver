// include/analytical_solution.h

#ifndef ANALYTICAL_SOLUTION_H
#define ANALYTICAL_SOLUTION_H

#include <vector>

class AnalyticalSolution {
public:
    virtual ~AnalyticalSolution() {}
    
    virtual std::vector<double> compute(int width, int height) const = 0;

};

#endif // ANALYTICAL_SOLUTION_H
