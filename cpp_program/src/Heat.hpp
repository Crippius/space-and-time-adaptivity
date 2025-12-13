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

  // Exact Solution: u(x,t) = T(t) * S(x)
  // T(t) = sum( A_k * sin(2*pi*nu_k*t + phi_k) )
  // S(x) = sum( exp( -|x-c_i|^2 / sigma_i^2 ) )
  template <int dim>
  class PulsatingGaussianSolution : public Function<dim>
  {
  public:
    std::vector<Point<dim>> centers;
    std::vector<double> sigmas;
    std::vector<double> A, nu, phi; // Temporal parameters

    PulsatingGaussianSolution(unsigned int n_peaks, unsigned int seed = 42)
      : Function<dim>(1)
    {
      std::mt19937 rng(seed);
      std::uniform_real_distribution<double> dist_coord(0.2, 0.8); // Keep away from boundaries slightly
      std::uniform_real_distribution<double> dist_sigma(0.05, 0.15);
      
      // Temporal parameters (fixed set of 3 modes for simplicity, or could be random)
      A = {1.0, 0.5, 0.25};
      nu = {1.0, 2.0, 3.0};
      phi = {0.0, M_PI/4.0, M_PI/2.0};

      centers.resize(n_peaks);
      sigmas.resize(n_peaks);

      for(unsigned int i=0; i<n_peaks; ++i) {
        for(unsigned int d=0; d<dim; ++d)
          centers[i][d] = dist_coord(rng);
        sigmas[i] = dist_sigma(rng);
      }
    }

    double get_temporal_part(double t) const {
      double val = 0.0;
      for(size_t k=0; k<A.size(); ++k) {
        val += A[k] * std::sin(2.0 * M_PI * nu[k] * t + phi[k]);
      }
      // Add a constant offset to make it interesting (e.g., mostly positive)
      return val + 2.0; 
    }

    double get_spatial_part(const Point<dim> &p) const {
      double val = 0.0;
      for(size_t i=0; i<centers.size(); ++i) {
        val += std::exp( -p.distance_square(centers[i]) / (sigmas[i]*sigmas[i]) );
      }
      return val;
    }

    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return get_temporal_part(this->get_time()) * get_spatial_part(p);
    }
  };

  // Forcing Term: f = u_t - laplacian(u)
  // u_t = T'(t) * S(x)
  // laplacian(u) = T(t) * laplacian(S(x))
  template <int dim>
  class PulsatingGaussianForcing : public Function<dim>
  {
    const PulsatingGaussianSolution<dim>& exact_solution;

  public:
    PulsatingGaussianForcing(const PulsatingGaussianSolution<dim>& exact)
      : Function<dim>(1), exact_solution(exact) {}

    double get_temporal_deriv(double t) const {
      double val = 0.0;
      for(size_t k=0; k<exact_solution.A.size(); ++k) {
        val += exact_solution.A[k] * (2.0 * M_PI * exact_solution.nu[k]) * 
               std::cos(2.0 * M_PI * exact_solution.nu[k] * t + exact_solution.phi[k]);
      }
      return val;
    }

    double get_laplacian_spatial(const Point<dim> &p) const {
      double val = 0.0;
      for(size_t i=0; i<exact_solution.centers.size(); ++i) {
        double dist_sq = p.distance_square(exact_solution.centers[i]);
        double sigma2 = exact_solution.sigmas[i] * exact_solution.sigmas[i];
        double sigma4 = sigma2 * sigma2;
        double exp_val = std::exp(-dist_sq / sigma2);
        
        // Laplacian of exp(-r^2/s^2) is exp(...) * (4r^2/s^4 - 2*dim/s^2)
        val += exp_val * ( (4.0 * dist_sq / sigma4) - (2.0 * dim / sigma2) );
      }
      return val;
    }

    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      double t = this->get_time();
      double T = exact_solution.get_temporal_part(t);
      double T_prime = get_temporal_deriv(t);
      double S = exact_solution.get_spatial_part(p);
      double laplacian_S = get_laplacian_spatial(p);

      // f = u_t - delta u = T'S - T(delta S)
      return T_prime * S - T * laplacian_S;
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
  output(const unsigned int &time_step, const double &time);

private:
  // Helper functions to extract parameters for the constructor
  static unsigned int
  get_n_peaks_from_prm(ParameterHandler &prm);
  static unsigned int
  get_random_seed_from_prm(ParameterHandler &prm);

  // Problem definition. ///////////////////////////////////////////////////////

  // Pulsating Field Parameters must be declared before exact_solution
  unsigned int n_peaks;
  unsigned int random_seed;

  FunctionMu mu;
  PulsatingGaussianSolution<dim> exact_solution;
  PulsatingGaussianForcing<dim> forcing_term;
  FunctionU0 u_0;
  
  // Discretization. ///////////////////////////////////////////////////////////
  unsigned int r;
  double       T;
  double       deltat;
  double       theta;
  unsigned int output_interval;

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
  unsigned int max_n_dofs{0}; // Track maximum DOFs during simulation

};

#endif