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

  // Exact Solution: u(x,t) = X(x) * T(t)
  // X(x) = sum( a_i * exp( -|x-c_i|^2 / (2*sigma_i^2) ) )
  // T(t) = sum( b_j * exp( -(t-tau_j)^2 / (2*delta_j^2) ) )
  template <int dim>
  class PulsatingGaussianSolution : public Function<dim>
  {
  public:
    // Spatial parameters (N_x)
    std::vector<double> a;       // amplitudes
    std::vector<Point<dim>> c;   // centers
    std::vector<double> sigma;   // widths

    // Temporal parameters (N_t)
    std::vector<double> b;       // amplitudes
    std::vector<double> tau;     // centers
    std::vector<double> delta;   // widths

    PulsatingGaussianSolution(unsigned int n_spatial_peaks, unsigned int n_temporal_peaks, 
                              double
                              T_final, double amplitude_min, double amplitude_max, double delta_min, double delta_max, double sigma_min, double sigma_max,
                              unsigned int seed = 42)
      : Function<dim>(1)
    {
      std::mt19937 rng(seed);
      //std::uniform_real_distribution<double> dist_amp(amplitude_min, amplitude_max);
      //std::uniform_real_distribution<double> dist_coord(0.2, 0.8);
      //std::uniform_real_distribution<double> dist_sigma(0.05, 0.15);
      
      // Spatial setup
      a.resize(n_spatial_peaks);
      c.resize(n_spatial_peaks);
      sigma.resize(n_spatial_peaks);

      double step_amp = (amplitude_max - amplitude_min)/(n_spatial_peaks-1);
      double dist_x = 1.0 /  10313;
      double step_sigma = (sigma_max - sigma_min) / (n_spatial_peaks-1);
      unsigned int prev = 1;

      for(unsigned int i=0; i<n_spatial_peaks; ++i) {
        a[i] = amplitude_min + i * step_amp;
        for(unsigned int d=0; d<dim; ++d){
          unsigned int tmp = ((i*prev*10353 +1+ d))%10313;
          c[i][d] = dist_x * tmp;
          prev = tmp;

        }
        sigma[i] = sigma_min + i*step_sigma;
      }

      // Temporal setup
      b.resize(n_temporal_peaks);
      tau.resize(n_temporal_peaks);
      delta.resize(n_temporal_peaks);
      
      //std::uniform_real_distribution<double> dist_time(0.1 * T_final, 0.9 * T_final);
      //std::uniform_real_distribution<double> dist_delta(delta_min, delta_max);

      double step_time = 8.0/n_temporal_peaks;
      step_amp = (amplitude_max - amplitude_min)/(n_temporal_peaks-1);
      double step_delta = (delta_max - delta_min)/(n_temporal_peaks-1);
      for(unsigned int j=0; j<n_temporal_peaks; ++j) {
        b[j] = amplitude_min + j * step_amp;
        tau[j] = 0.1 + j * step_time;
        delta[j] = delta_min + j * step_delta;
      }

      // force the last peak to be at final time.
      tau[n_temporal_peaks-1] = T_final+delta[n_temporal_peaks-1]/2;
    }

    double get_spatial_part(const Point<dim> &p) const {
      double val = 0.0;
      for(size_t i=0; i<a.size(); ++i) {
        double dist_sq = p.distance_square(c[i]);
        val += a[i] * std::exp( -dist_sq / (2.0 * sigma[i] * sigma[i]) );
      }
      return val;
    }

    double get_temporal_part(double t) const {
      double val = 0.0;
      for(size_t j=0; j<b.size(); ++j) {
        double diff = t - tau[j];
        val += b[j] * std::exp( -(diff * diff) / (2.0 * delta[j] * delta[j]) );
      }
      return val;
    }

    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return get_spatial_part(p) * get_temporal_part(this->get_time());
    }
  };

  // Forcing Term: f = X * T' - alpha * (laplacian X) * T
  template <int dim>
  class PulsatingGaussianForcing : public Function<dim>
  {
    const PulsatingGaussianSolution<dim>& exact_solution;
    double alpha;

  public:
    PulsatingGaussianForcing(const PulsatingGaussianSolution<dim>& exact, double alpha_val = 1.0)
      : Function<dim>(1), exact_solution(exact), alpha(alpha_val) {}

    double get_temporal_deriv(double t) const {
      double val = 0.0;
      const auto& b = exact_solution.b;
      const auto& tau = exact_solution.tau;
      const auto& delta = exact_solution.delta;

      for(size_t j=0; j<b.size(); ++j) {
        double diff = t - tau[j];
        double delta2 = delta[j] * delta[j];
        // T'(t) term: - b_j * (t - tau_j) / delta_j^2 * exp(...)
        val += - (b[j] * diff / delta2) * std::exp( -(diff * diff) / (2.0 * delta2) );
      }
      return val;
    }

    double get_laplacian_spatial(const Point<dim> &p) const {
      double val = 0.0;
      const auto& a = exact_solution.a;
      const auto& c = exact_solution.c;
      const auto& sigma = exact_solution.sigma;

      for(size_t i=0; i<a.size(); ++i) {
        double dist_sq = p.distance_square(c[i]);
        double sigma2 = sigma[i] * sigma[i];
        double sigma4 = sigma2 * sigma2;
        double exp_val = std::exp( -dist_sq / (2.0 * sigma2) );
        
        // Laplacian term: a_i * [ r^2 / sigma^4 - dim / sigma^2 ] * exp(...)
        val += a[i] * ( (dist_sq / sigma4) - (static_cast<double>(dim) / sigma2) ) * exp_val;
      }
      return val;
    }

    virtual double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      double t = this->get_time();
      double xp = exact_solution.get_spatial_part(p);
      double tp = exact_solution.get_temporal_part(t);
      double T_prime = get_temporal_deriv(t);
      double laplacian_X = get_laplacian_spatial(p);

      // f = X * T' - alpha * (laplacian X) * T
      return xp * T_prime - alpha * laplacian_X * tp;
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
  static double
  get_final_time_from_prm(ParameterHandler &prm);
  static unsigned int
  get_n_spatial_peaks_from_prm(ParameterHandler &prm);
  static unsigned int
  get_n_temporal_peaks_from_prm(ParameterHandler &prm);
  static unsigned int
  get_random_seed_from_prm(ParameterHandler &prm);
  static double
  get_delta_min_from_prm(ParameterHandler &prm);
  static double
  get_delta_max_from_prm(ParameterHandler &prm);
  static double
  get_sigma_min_from_prm(ParameterHandler &prm);
  static double
  get_sigma_max_from_prm(ParameterHandler &prm);
  static double
  get_amplitude_min_from_prm(ParameterHandler &prm);
  static double
  get_amplitude_max_from_prm(ParameterHandler &prm);

  // Problem definition. ///////////////////////////////////////////////////////

  // Pulsating Field Parameters must be declared before exact_solution
  unsigned int n_peaks;
  unsigned int n_temporal_peaks;
  unsigned int random_seed;
  double delta_min;
  double delta_max;
  double amplitude_min;
  double amplitude_max;
  double sigma_min;
  double sigma_max;

  // Discretization. ///////////////////////////////////////////////////////////
  unsigned int r;
  double       T;
  double       deltat;
  double       theta;
  unsigned int output_interval;

  // Forcing term and Analytical solution
  FunctionMu mu;
  PulsatingGaussianSolution<dim> exact_solution;
  PulsatingGaussianForcing<dim> forcing_term;
  FunctionU0 u_0;


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
  bool enable_logging;

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
