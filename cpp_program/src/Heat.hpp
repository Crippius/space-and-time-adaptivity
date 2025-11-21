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

  // Function for the forcing term (9 sources, relay activation in groups).
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor
    explicit ForcingTerm(const double T_final)
      : Function<dim>(),
        T(T_final),
        // Heat sources parameters
        A{1.0, 0.5, 0.25},
        nu{1.0, 3.0, 5.0},
        phi{0.0, M_PI / 2.0, M_PI}
    {

      centers = {// Group 1: Short peaks
                 Point<dim>(0.1, 0.1, 0.1),
                 Point<dim>(0.9, 0.1, 0.9),
                 Point<dim>(0.5, 0.9, 0.5),
                 // Groupo 2: Medium peaks
                 Point<dim>(0.2, 0.8, 0.2),
                 Point<dim>(0.8, 0.2, 0.8),
                 Point<dim>(0.2, 0.2, 0.8),
                 // Group 3: Wide peaks
                 Point<dim>(0.5, 0.5, 0.5),
                 Point<dim>(0.1, 0.9, 0.1),
                 Point<dim>(0.9, 0.9, 0.1)};

      sigmas = {// Group 1: Short peaks
                0.015,
                0.020,
                0.018,
                // Group 2: Medium peaks
                0.08,
                0.07,
                0.09,
                // Group 3: Wide peaks
                0.20,
                0.22,
                0.18};
    }

    // Forcing term: f(x,t) = (∑ₖ Aₖ sin(2π νₖ t + φₖ)) * (∑ᵢ exp(-||x-xᵢ||²/σ_spatial²))
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      const double t = this->get_time();

      
      double temporal_wave = 0.0;
      for (unsigned int k = 0; k < A.size(); ++k)
        temporal_wave += A[k] * std::sin(2.0 * M_PI * nu[k] * t + phi[k]) + A[k];

      double spatial_term = 0.0;

      // Activate sources in groups based on time t
      // 3 sources per group, total 9 sources
      if (t < T / 3.0)
        {
          for (unsigned int i = 0; i < 3; ++i)
            spatial_term += std::exp(-p.distance_square(centers[i]) /
                                     (sigmas[i] * sigmas[i]));
        }
      else if (t < 2.0 * T / 3.0)
        {
          for (unsigned int i = 3; i < 6; ++i)
            spatial_term += std::exp(-p.distance_square(centers[i]) /
                                     (sigmas[i] * sigmas[i]));
        }
      else
        {
          for (unsigned int i = 6; i < 9; ++i)
            spatial_term += std::exp(-p.distance_square(centers[i]) /
                                     (sigmas[i] * sigmas[i]));
        }

      return temporal_wave * spatial_term;
    }

  public: 
    const double            T; 
    std::vector<double>     A, nu, phi;
    std::vector<Point<dim>> centers;
    std::vector<double>     sigmas;
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
