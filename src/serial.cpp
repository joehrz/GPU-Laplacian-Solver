#include <iostream>
#include <vector>
#include <cmath>



using namespace std;

void sorMethod(vector<vector<double>> &phi, const vector<vector<double>> &a,
                const vector<vector<double>> &b, const vector<vector<double>> &c,
                const vector<vector<double>> &d, const vector<vector<double>> &e,
                const vector<vector<double>> &f, double omega, double tol, int maxIter){


    int M = phi.size();
    int N = phi[0].size();

    vector<vector<double>> r(M, vector<double>(N, 0.0));

    for (int n = 0; n < maxIter; n++){
        for (int i = 1; i < M-1; i++){
            for (int j = 1; j < N - 1; ++j){

                r[i][j] = f[i][j] - (a[i][j] * phi[i+1][j] + b[i][j] * phi[i-1][j] +
                                    c[i][j] * phi[i][j+1] + d[i][j] * phi[i][j-1] + e[i][j] * phi[i][j]);

                phi[i][j] += omega * r[i][j] / e[i][j];

            }


        }
    

        double norm_r = 0.0;
        double norm_f = 0.0;

        for (int i = 0; i < M; i++){
            for (int j = 0; j < N; ++j){
                norm_r += r[i][j] * r[i][j];
                norm_f += f[i][j] * f[i][j];
            }
        }

        if (sqrt(norm_r) / sqrt(norm_f) <  tol){
            cout << "Converged after " << n + 1 << " iterations." << endl;
            return;
        }
        cout << "Reached maximum iterations." << endl;
    }
                }
int main() {
    int M = 100, N = 100; // Dimensions
    double omega = 1.5; // Over-relaxation factor
    double tol = 1e-6; // Tolerance
    int max_ite = 10000; // Maximum number of iterations

    vector<vector<double>> phi(M, vector<double>(N, 0)), f(M, vector<double>(N, 1.0));
    vector<vector<double>> a(M, vector<double>(N, -1)), b(M, vector<double>(N, -1));
    vector<vector<double>> c(M, vector<double>(N, -1)), d(M, vector<double>(N, -1));
    vector<vector<double>> e(M, vector<double>(N, 4));

    sorMethod(phi, a, b, c, d, e, f, omega, tol, max_ite);
    return 0;
}