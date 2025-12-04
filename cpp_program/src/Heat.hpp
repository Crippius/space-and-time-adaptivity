#ifndef HEAT_HPP
#define HEAT_HPP

// Import necessary headers from deal.II library
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/distributed/solution_transfer.h>

// Include necessary standard libraries
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class Heat
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the mu coefficient.
  class FunctionMu : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };

struct SpectralDim {
    std::vector<double> amplitudes;
    std::vector<double> wavenumbers; // For space: n*pi, For time: omega
};

class SpectralSumSolution : public Function<dim>
{
public:
    // We store coefficients for X, Y, Z (and potentially more) + Time
    std::vector<SpectralDim> space_dims;
    SpectralDim time_dim;

    SpectralSumSolution(unsigned int n_modes)
    : Function<dim>(1)
    {

        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist_amp(-1.0, 1.0);

        // --- 1. SETUP SPATIAL MODES (X, Y, Z) ---
        // We use prime numbers for frequencies to avoid repeating patterns
        std::vector<double> base_freqs = {1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0};

        space_dims.resize(dim);
        for(unsigned int d=0; d<dim; ++d) {
            for(unsigned int i=0; i<n_modes; ++i) {
                // Random amplitude
                space_dims[d].amplitudes.push_back(dist_amp(rng));

                // Wavenumber k = n * pi
                // We pick from prime list to make it "messy"
                double n = base_freqs[i % base_freqs.size()];
                // Add some random high-frequency noise for stress testing
                if (i > n_modes/2) n += (dist_amp(rng) + 1.0) * 2.0;

                space_dims[d].wavenumbers.push_back(n * numbers::PI);
            }
        }

        // --- 2. SETUP TIME MODES ---
        // Slower dynamics than space so the solver can catch up
        for(unsigned int i=0; i<n_modes; ++i) {
            time_dim.amplitudes.push_back(dist_amp(rng));
            // Time frequencies: random values between 1 and 10
            time_dim.wavenumbers.push_back(std::abs(dist_amp(rng)) * 10.0 + 1.0);
        }
    }

    // Helper to evaluate Sum( a * sin(k*val) )
    double eval_1d(double val, const SpectralDim& s) const {
        double sum = 0.0;
        for(size_t i=0; i<s.amplitudes.size(); ++i) {
            sum += s.amplitudes[i] * std::sin(s.wavenumbers[i] * val);
        }
        return sum;
    }

    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
        // u = X(x) * Y(y) * Z(z) * T(t)
        double spatial_part = 1.0;
        for(unsigned int d=0; d<dim; ++d) {
            spatial_part *= eval_1d(p[d], space_dims[d]);
        }

        double time_part = eval_1d(this->get_time(), time_dim);

        return spatial_part * time_part;
    }

};

  // Function for the forcing term (9 sources, relay activation in groups).
  class ForcingTerm : public Function<dim>
  {
  public:
    const SpectralSumSolution& exact_solution;
    double alpha;
    // Constructor
      ForcingTerm(const SpectralSumSolution& exact, double alpha_val)
      : Function<dim>(1), exact_solution(exact), alpha(alpha_val) {}

    // Forcing term: f(x,t) = (∑ₖ Aₖ sin(2π νₖ t + φₖ)) * (∑ᵢ exp(-||x-xᵢ||²/σ_spatial²))

    // Helper: evaluate Sum( a * sin(k*val) )
    double eval_func(double val, const SpectralDim& s) const {
        double sum = 0.0;
        for(size_t i=0; i<s.amplitudes.size(); ++i) {
            sum += s.amplitudes[i] * std::sin(s.wavenumbers[i] * val);
        }
        return sum;
    }

    // Helper: evaluate derivative Sum( a * k * cos(k*val) )
    double eval_deriv(double val, const SpectralDim& s) const {
        double sum = 0.0;
        for(size_t i=0; i<s.amplitudes.size(); ++i) {
            sum += s.amplitudes[i] * s.wavenumbers[i] * std::cos(s.wavenumbers[i] * val);
        }
        return sum;
    }

    // Helper: evaluate 2nd derivative Sum( -a * k^2 * sin(k*val) )
    double eval_2nd_deriv(double val, const SpectralDim& s) const {
        double sum = 0.0;
        for(size_t i=0; i<s.amplitudes.size(); ++i) {
            double k = s.wavenumbers[i];
            sum += -s.amplitudes[i] * k * k * std::sin(k * val);
        }
        return sum;
    }

    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {

        double t = this->get_time();
        // 1. Precompute parts for X, Y, Z
        //    We need Function (F) and 2nd Derivative (F'') for each dimension
        double X  = eval_func(p[0], exact_solution.space_dims[0]);
        double X_xx = eval_2nd_deriv(p[0], exact_solution.space_dims[0]);

        double Y  = eval_func(p[1], exact_solution.space_dims[1]);
        double Y_yy = eval_2nd_deriv(p[1], exact_solution.space_dims[1]);

        double Z = 1.0, Z_zz = 0.0;
        if(dim == 3) {
            Z  = eval_func(p[2], exact_solution.space_dims[2]);
            Z_zz = eval_2nd_deriv(p[2], exact_solution.space_dims[2]);
        }

        // 2. Precompute Time parts
        double T  = eval_func(t, exact_solution.time_dim);
        double T_t = eval_deriv(t, exact_solution.time_dim);

        // 3. Assemble components
        // u = X * Y * Z * T

        // Time Derivative Term: u_t = (X Y Z) * T'
        double u_t = (X * Y * Z) * T_t;

        // Laplacian Term: laplace(u) = (X'' Y Z + X Y'' Z + X Y Z'') * T
        double laplace_u = (X_xx * Y * Z) + (X * Y_yy * Z);
        if(dim == 3) {
            laplace_u += (X * Y * Z_zz);
        }
        laplace_u *= T;

        // Final Source: f = u_t - alpha * laplace(u)
        return u_t - (alpha * laplace_u);
    }

  };

  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // ParameterHandler declaration.
  static void
  declare_parameters(ParameterHandler &prm);

  // Constructor for the Heat class, initializes the problem with parameters.
  Heat(ParameterHandler &prm);


  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

protected:
  // Metodo per leggere i parametri dall'handler
  void
  parse_parameters(ParameterHandler &prm);

  // Metodo per calcolare e stampare le metriche di performance
  void
  compute_and_print_metrics() const;
  
  // Create the cube mesh
  void
  create_mesh();

  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices();

  // Assemble the right-hand side of the problem.
  void
  assemble_rhs(const double &time);

  // Solve the problem for one time step.
  void
  solve_time_step();

  // Refine the grid based on solution error estimation.
  void
  refine_grid();

  // Estimate the time discretization error for the current step
  double estimate_time_error(const double &time,
                             const TrilinosWrappers::MPI::Vector &prev_solution_owned,
                             double trial_deltat);

  // Adapt the time step based on error 
  bool adapt_time_step(const double &current_time, 
                       const TrilinosWrappers::MPI::Vector &solution_at_tn,
                       double &next_deltat);

  // Output.
  void
  output(const unsigned int &time_step) const;

  // Problem definition. ///////////////////////////////////////////////////////

  FunctionMu mu;
    SpectralSumSolution exact_solution;
    ForcingTerm forcing_term;

  FunctionU0 u_0;
  
  // Discretization. ///////////////////////////////////////////////////////////
  unsigned int r;
  double       T;
  double       deltat;
  double       theta;

  // MPI parallel metadata.
  const unsigned int mpi_size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout{std::cout, mpi_rank == 0};

  // Mesh.
  parallel::distributed::Triangulation<dim> mesh{MPI_COMM_WORLD};

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Parallel index sets.
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  // Constraints for hanging nodes
  AffineConstraints<double> constraints;

  // System matrices and vectors (Trilinos, MPI-aware)
  TrilinosWrappers::SparseMatrix mass_matrix;
  TrilinosWrappers::SparseMatrix stiffness_matrix;
  TrilinosWrappers::SparseMatrix lhs_matrix;
  TrilinosWrappers::SparseMatrix rhs_matrix;
  TrilinosWrappers::MPI::Vector system_rhs;
  TrilinosWrappers::MPI::Vector solution_owned; // without ghosts
  TrilinosWrappers::MPI::Vector solution;       // with ghosts (for output)

  // Space Adaptativity Parameters. ///////////////////////////////////////////////////////////
  unsigned int n_global_refinements;
  unsigned int refinement_interval;
  double       refinement_percent;
  double       coarsening_percent;
  
  bool enable_space_adaptivity;
  bool enable_time_adaptivity;

  // Time Adaptativity Parameters. ///////////////////////////////////////////////////////////
  unsigned int time_adapt_interval;
  double       time_error_lower_bound;
  double       time_error_upper_bound;
  double       min_deltat;
  double       max_deltat;

  // Performance Tracking. /////////////////////////////////////////////////////
  std::chrono::duration<double> time_total{0.0};
  std::chrono::duration<double> time_refine{0.0};
  std::chrono::duration<double> time_assemble_matrices{0.0};
  std::chrono::duration<double> time_assemble_rhs{0.0};
  std::chrono::duration<double> time_solve_step{0.0};
  unsigned int n_time_steps{0};

};

#endif
