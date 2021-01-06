#include <unistd.h>
#include <fcntl.h>
#include "ortools/linear_solver/linear_solver.h"

using namespace operations_research;


void usage() {
    std::cout << "./emd <read_pipe> <write_pipe> <image_size>" << std::endl;
}


int main(int argc, char const *argv[])
{
    // Usage check
    if (argc != 4) {
        usage();
        exit(1);
    }

    unsigned int image_size = atoi(argv[3]);

    // Connect to read named pipe
    int readFd, writeFd;
    if ((readFd = open(argv[1], O_RDONLY)) == -1) {
        perror("Could not open read pipe");
        exit(1);
    }

    // Connect to write named pipe
    if ((writeFd = open(argv[2], O_WRONLY)) == -1) {
        perror("Could not open write pipe");
        close(readFd);
        exit(1);
    }

    double d[image_size][image_size], wi[image_size], wj[image_size];
    double result;
    // Start reading and calculating EMDs
    while (read(readFd, d, sizeof(d)) > 0 && read(readFd, wi, sizeof(wi)) > 0 && read(readFd, wj, sizeof(wj)) > 0) {
        // Create a linear solver
        MPSolver* linearSolver = MPSolver::CreateSolver("GLOP");

        //absl::Status st = linearSolver->SetNumThreads(8);
        //std::cout << linearSolver->GetSolverSpecificParametersAsString() << std::endl;

        double infinity = linearSolver->infinity();

        // Define the flow variables (fij)
        MPVariable *flow[image_size][image_size];
        for (unsigned int i = 0; i < image_size; i++) {
            for (unsigned int j = 0; j < image_size; j++) {
                flow[i][j] = linearSolver->MakeNumVar(0, infinity, "flow_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
        
        // Define the constraints
        MPConstraint *c0[image_size];
        for (unsigned int i = 0;i < image_size;i++) {
            c0[i] = linearSolver->MakeRowConstraint(wi[i], wi[i], "c0_" + std::to_string(i));
            // Set the coefficients
            for (int j = 0;j < image_size;j++) {
                c0[i]->SetCoefficient(flow[i][j], 1);
            }
        }

        MPConstraint *c1[image_size];
        for (unsigned int j = 0;j < image_size;j++) {
            c0[j] = linearSolver->MakeRowConstraint(wj[j], wj[j], "c1_" + std::to_string(j));
            // Set the coefficients
            for (int i = 0;i < image_size;i++) {
                c0[j]->SetCoefficient(flow[i][j], 1);
            }
        }

        // Define minimization objective
        MPObjective* objective = linearSolver->MutableObjective();
        for (unsigned int i = 0; i < image_size; i++) {
            for (unsigned int j = 0; j < image_size; j++) {
                objective->SetCoefficient(flow[i][j], d[i][j]);
            }
        }

        objective->SetMinimization();

        // Solve
        MPSolver::ResultStatus result_status = linearSolver->Solve();

        result = objective->Value();

        // Send back the result
        if(write(writeFd, &result, sizeof(result)) <= 0) {
            perror("write");
            break;
        }

        delete linearSolver;
    }
    
    // Close pipes
    close(writeFd);
    close(readFd);
    
    exit(0);
}
