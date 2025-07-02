// src/common/src/solution_export.cpp

#include "solution_export.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef SOLVER_ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

void exportHostDataToCSV(const float* h,int W,int H,
                         const std::string& fn,const std::string& tag)
{
    std::ofstream f(fn);
    if(!f.is_open()){
        std::cerr<<'['<<tag<<"] cannot open "<<fn<<'\n'; return; }
    f<<std::fixed<<std::setprecision(6);
    for(int j=0;j<H;++j){
        for(int i=0;i<W;++i){
            f<<h[j*W+i]; if(i+1!=W) f<<','; }
        f<<'\n';
    }
    std::cout<<'['<<tag<<"] wrote "<<fn<<'\n';
}

#ifdef SOLVER_ENABLE_CUDA
void exportDeviceSolutionToCSV(const float* d,int W,int H,
                               const std::string& fn,const std::string& tag)
{
    std::vector<float> h(W*H);
    cudaMemcpy(h.data(),d,W*H*sizeof(float),cudaMemcpyDeviceToHost);
    exportHostDataToCSV(h.data(),W,H,fn,tag);
}
#else
void exportDeviceSolutionToCSV(const float*,int,int,
                               const std::string&,const std::string& tag)
{
    std::cerr<<'['<<tag<<"] CUDA not enabled, cannot export device data\n";
}
#endif