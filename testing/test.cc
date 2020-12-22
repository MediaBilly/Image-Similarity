#include "ortools/linear_solver/linear_solver.h"

using namespace operations_research;

int main(int argc, char const *argv[]) {
    // Create the linear solver with the GLOP backend.
    MPSolver* solver = MPSolver::CreateSolver("GLOP");

    const double infinity = solver->infinity();
    // Create the variables x and y.
    MPVariable* const x = solver->MakeNumVar(0.0, 1.0, "x");
    MPVariable* const y = solver->MakeNumVar(0.0, 2.0, "y");

    LOG(INFO) << "Number of variables = " << solver->NumVariables();

    // x + y <= 2.0.
    MPConstraint* const c0 = solver->MakeRowConstraint(-infinity, 2.0, "c0");
    c0->SetCoefficient(x, 1);
    c0->SetCoefficient(y, 1);

    LOG(INFO) << "Number of constraints = " << solver->NumConstraints();

    // Maximize 3*x + y.
    MPObjective* const objective = solver->MutableObjective();
    objective->SetCoefficient(x, 3);
    objective->SetCoefficient(y, 1);
    objective->SetMaximization();

    const MPSolver::ResultStatus result_status = solver->Solve();
    // Check that the problem has an optimal solution.
    if (result_status != MPSolver::OPTIMAL) {
    LOG(FATAL) << "The problem does not have an optimal solution!";
    }
    LOG(INFO) << "Solution:" << std::endl;
    LOG(INFO) << "Objective value = " << objective->Value();
    LOG(INFO) << "x = " << x->solution_value();
    LOG(INFO) << "y = " << y->solution_value();

    LOG(INFO) << "\nAdvanced usage:";
    LOG(INFO) << "Problem solved in " << solver->wall_time() << " milliseconds";
    LOG(INFO) << "Problem solved in " << solver->iterations() << " iterations";
    LOG(INFO) << "Problem solved in " << solver->nodes() << " branch-and-bound nodes";

    return 0;
}
