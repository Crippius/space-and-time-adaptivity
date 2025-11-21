#include "Heat.hpp"
#include <chrono>
#include <iomanip>


// ParameterHandler declaration.
void
Heat::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Adaptivity Control");
  {
    prm.declare_entry("Enable space adaptivity", "true", Patterns::Bool(), 
                      "If true, enables adaptive mesh refinement.");
    prm.declare_entry("Enable time adaptivity", "true", Patterns::Bool(), 
                      "If true, enables adaptive time stepping.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Discretization");
  {
    prm.declare_entry("Degree", "1", Patterns::Integer(0), "Polynomial degree of FE");
    prm.declare_entry("Global refinements", "2", Patterns::Integer(0), 
                      "Number of global refinements for uniform meshes.");
    prm.declare_entry("Final time", "2.0", Patterns::Double(0.0), "Final time T");
    prm.declare_entry("Initial deltat", "0.05", Patterns::Double(0.0), "Initial time step size");
    prm.declare_entry("Theta", "0.5", Patterns::Double(0.0, 1.0), "Theta for the time-stepping scheme");
  }
  prm.leave_subsection();

  prm.enter_subsection("Space Adaptivity");
  {
    prm.declare_entry("Refinement interval", "5", Patterns::Integer(1), "Steps between mesh refinements");
    prm.declare_entry("Refinement fraction", "0.1", Patterns::Double(0.0, 1.0), "Top fraction of cells to refine");
    prm.declare_entry("Coarsening fraction", "0.9", Patterns::Double(0.0, 1.0), "Bottom fraction of cells to coarsen");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time Adaptivity");
  {
    prm.declare_entry("Adaptivity interval", "1", Patterns::Integer(1), "Steps between time error checks");
    prm.declare_entry("Error lower bound", "0.0005", Patterns::Double(0.0), "Lower bound for time error");
    prm.declare_entry("Error upper bound", "0.002", Patterns::Double(0.0), "Upper bound for time error");
    prm.declare_entry("Min deltat", "1e-4", Patterns::Double(0.0), "Minimum allowed time step");
    prm.declare_entry("Max deltat", "0.2", Patterns::Double(0.0), "Maximum allowed time step");
  }
  prm.leave_subsection();
}

// Actual implementation of the parse_parameters method.
void
Heat::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Adaptivity Control");
  {
    enable_space_adaptivity = prm.get_bool("Enable space adaptivity");
    enable_time_adaptivity  = prm.get_bool("Enable time adaptivity");
  }
  prm.leave_subsection();

  prm.enter_subsection("Discretization");
  {
    r      = prm.get_integer("Degree");
    n_global_refinements = prm.get_integer("Global refinements");
    T      = prm.get_double("Final time");
    deltat = prm.get_double("Initial deltat");
    theta  = prm.get_double("Theta");
  }
  prm.leave_subsection();

  prm.enter_subsection("Space Adaptivity");
  {
    refinement_interval = prm.get_integer("Refinement interval");
    refinement_percent  = prm.get_double("Refinement fraction");
    coarsening_percent  = prm.get_double("Coarsening fraction");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time Adaptivity");
  {
    time_adapt_interval    = prm.get_integer("Adaptivity interval");
    time_error_lower_bound = prm.get_double("Error lower bound");
    time_error_upper_bound = prm.get_double("Error upper bound");
    min_deltat             = prm.get_double("Min deltat");
    max_deltat             = prm.get_double("Max deltat");
  }
  prm.leave_subsection();
}


Heat::Heat(ParameterHandler &prm)
  : forcing_term(0.0) // Temp declaration, will be updated later
{
  // Read the parameters from the ParameterHandler
  parse_parameters(prm);
  
  // Update forcing term with the final time
  const_cast<double&>(forcing_term.T) = T;
}


void
Heat::create_mesh()
{
  // Create the cube mesh.
  // This method generates a hypercube mesh in the unit cube [0,1]^dim.

  pcout << "Creating cube mesh" << std::endl;
  
  GridGenerator::hyper_cube(mesh, 0.0, 1.0);
  mesh.refine_global(n_global_refinements);
  
  pcout << "  Number of elements = " << mesh.n_active_cells()
        << std::endl;
}

void
Heat::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;
    create_mesh();
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_Q<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity, constraints, /*keep_constrained_dofs=*/false);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void
Heat::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      fe_values.reinit(cell);

      cell_mass_matrix      = 0.0;
      cell_stiffness_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double mu_loc = mu.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) *
                                            fe_values.JxW(q);

                  cell_stiffness_matrix(i, j) +=
                    mu_loc * fe_values.shape_grad(i, q) *
                    fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      // Apply constraints while assembling into global matrices
      constraints.distribute_local_to_global(cell_mass_matrix, dof_indices, mass_matrix);
      constraints.distribute_local_to_global(cell_stiffness_matrix, dof_indices, stiffness_matrix);
    }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  // LHS = M/deltat + theta*K
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix *= (1.0 / deltat);
  lhs_matrix.add(theta, stiffness_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution un).
  // RHS = M/deltat - (1-theta)*K
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix *= (1.0 / deltat);
  rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
}

void
Heat::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      fe_values.reinit(cell);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          forcing_term.set_time(time);
          const double f_new_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          // Compute f(tn)
          forcing_term.set_time(time - deltat);
          const double f_old_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      constraints.distribute_local_to_global(cell_rhs, dof_indices, system_rhs);
    }
  system_rhs.compress(VectorOperation::add);
  rhs_matrix.vmult_add(system_rhs, solution_owned);
}

void
Heat::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  // Enforce hanging-node constraints on the solution
  constraints.distribute(solution_owned);
  
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
  
  solution = solution_owned;
}

void
Heat::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "u");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./",
                                      "output_",
                                      time_step,
                                      MPI_COMM_WORLD,
                                      3);
}

void
Heat::refine_grid()
{
  pcout << "\n[Space adaptivity] Performing adaptive mesh refinement" << std::endl;

  pcout << " Refining grid based on error estimation" << std::endl;
  pcout << "  Number of active cells before refinement: "
        << mesh.n_active_cells() << std::endl;

  Vector<float> estimated_error_per_cell(mesh.n_active_cells());
  // ESTIMATE
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(r + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell);

  // MARK (use parallel-aware refinement marking)
  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
    mesh,
    estimated_error_per_cell,
    refinement_percent,
    coarsening_percent);

  // REFINE + TRANSFER (MPI)
  parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
    sol_trans(dof_handler);

  // Use ghosted solution for prepare to improve interface accuracy
  mesh.prepare_coarsening_and_refinement();
  sol_trans.prepare_for_coarsening_and_refinement(solution);
  mesh.execute_coarsening_and_refinement();

  // Redistribute DoFs and rebuild LA objects
  dof_handler.distribute_dofs(*fe);
  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  pcout << "  Number of DoFs after refinement = " << dof_handler.n_dofs() << std::endl;

  // Constraints
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // Rebuild sparsity + matrices
  {
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity, constraints, /*keep_constrained_dofs=*/false);
    sparsity.compress();

    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);
  }

  // Reinit vectors
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

  // Interpolate solution onto new mesh (owned vector)
  sol_trans.interpolate(solution_owned);
  constraints.distribute(solution_owned);
  solution = solution_owned;

  pcout << "  Reassembling matrices for the refined mesh" << std::endl;
  assemble_matrices();
}

double Heat::estimate_time_error(const double &time, const TrilinosWrappers::MPI::Vector &prev_solution_owned, double trial_deltat)
{
  // Save current state
  TrilinosWrappers::MPI::Vector backup_solution = solution_owned;
  double backup_deltat = deltat;
  double eps = 1e-8;
  
  // Temporarily store old matrices to avoid full reassembly
  TrilinosWrappers::SparseMatrix backup_lhs, backup_rhs;
  backup_lhs.copy_from(lhs_matrix);
  backup_rhs.copy_from(rhs_matrix);
  
  // Big Step Solution
  deltat = trial_deltat;
  // Update the time-dependent matrices
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix *= (1.0 / deltat);
  lhs_matrix.add(theta, stiffness_matrix);
  
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix *= (1.0 / deltat);
  rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
  
  solution_owned = prev_solution_owned;
  solution = solution_owned;
  assemble_rhs(time + trial_deltat);
  solve_time_step();
  TrilinosWrappers::MPI::Vector sol_big_step = solution_owned;

  // Two Half Steps Solution
  deltat = trial_deltat / 2.0;
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix *= (1.0 / deltat);
  lhs_matrix.add(theta, stiffness_matrix);
  
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix *= (1.0 / deltat);
  rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
  
  solution_owned = prev_solution_owned;
  solution = solution_owned;
  assemble_rhs(time + deltat);
  solve_time_step();
  assemble_rhs(time + 2.0 * deltat);
  solve_time_step();
  TrilinosWrappers::MPI::Vector sol_two_half_steps = solution_owned;

  // Compute the error estimate (Richardson extrapolation)
  sol_big_step -= sol_two_half_steps;
  const double num = sol_big_step.l2_norm();
  const double den = sol_two_half_steps.l2_norm() + eps;
  double error = num / den;
  // Synchronize error across ranks
  error = Utilities::MPI::max(error, MPI_COMM_WORLD);

  // Restore original state
  solution_owned = backup_solution;
  solution = solution_owned;
  deltat = backup_deltat;
  lhs_matrix.copy_from(backup_lhs);
  rhs_matrix.copy_from(backup_rhs);

  return error;
}

bool Heat::adapt_time_step(const double &current_time, 
                           const TrilinosWrappers::MPI::Vector &solution_at_tn,
                           double &next_deltat)
{
  // Estimate error of the current step that was just computed
  double time_error = estimate_time_error(current_time - deltat, solution_at_tn, deltat);
  time_error = Utilities::MPI::max(time_error, MPI_COMM_WORLD);
  
  pcout << "    [Time adaptivity] Estimated error: " << time_error;
  
  // If error is too large, reject the step
  if (time_error > time_error_upper_bound)
  {
    double new_deltat = deltat / 2.0;
    if (new_deltat < min_deltat)
    {
      pcout << " > upper bound, but deltat already at minimum. Accepting step." << std::endl;
      // Can't reduce further, accept the step but don't increase deltat
      next_deltat = deltat;
      return true; 
    }
    else
    {
      pcout << " > upper bound. REJECTING step, reducing deltat: " 
            << deltat << " -> " << new_deltat << std::endl;
      next_deltat = new_deltat;
      return false; // Reject
    }
  }
  
  // Step is acceptable, decide on next deltat
  if (time_error < time_error_lower_bound)
  {
    // Error is small, try to increase deltat for next step
    double new_deltat = std::min(deltat * 2.0, max_deltat);
    if (new_deltat > deltat)
    {
      pcout << " < lower bound. Accepting step, increasing deltat: " 
            << deltat << " -> " << new_deltat << std::endl;
      next_deltat = new_deltat;
    }
    else
    {
      pcout << " < lower bound, but deltat at maximum. Accepting step." << std::endl;
      next_deltat = deltat;
    }
  }
  else
  {
    // Error is in acceptable range
    pcout << " in acceptable range. Accepting step, keeping deltat = " << deltat << std::endl;
    next_deltat = deltat;
  }
  
  return true;
}

void
Heat::solve()
{
  // Start wall-clock timer
  auto t_total_start = std::chrono::high_resolution_clock::now();

  // Reset performance metrics
  n_time_steps   = 0;
  unsigned int num_assemblies = 0;
  unsigned int n_refinements  = 0;

  auto t0 = std::chrono::high_resolution_clock::now();
  assemble_matrices();
  auto t1 = std::chrono::high_resolution_clock::now();
  time_assemble_matrices += t1 - t0;
  ++num_assemblies; 

  pcout << "===============================================" << std::endl;

  {
    pcout << "Applying the initial condition" << std::endl;
    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    constraints.distribute(solution_owned);
    solution = solution_owned;
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }
  
  unsigned int time_step = 0;
  double time = 0;
  double next_deltat = deltat; // For adaptive time stepping

  while (time < T)
  {
    // Apply space adaptivity before taking the step
    if (enable_space_adaptivity && time_step > 0 && time_step % refinement_interval == 0)
    {
      auto tr0 = std::chrono::high_resolution_clock::now();
      refine_grid();
      auto tr1 = std::chrono::high_resolution_clock::now();
      time_refine += tr1 - tr0;
      n_refinements++;
      ++num_assemblies;
    }
    
    // Adjust deltat to not overshoot final time
    if (time + deltat > T)
    {
      deltat = T - time;
      // Reassemble matrices with adjusted deltat
      lhs_matrix.copy_from(mass_matrix);
      lhs_matrix *= (1.0 / deltat);
      lhs_matrix.add(theta, stiffness_matrix);
      rhs_matrix.copy_from(mass_matrix);
      rhs_matrix *= (1.0 / deltat);
      rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
    }

    ++time_step;
    ++n_time_steps;
    
    // Save solution before taking the step (needed for time adaptivity)
    TrilinosWrappers::MPI::Vector solution_before_step = solution_owned;
    double time_before_step = time;
    
    pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5) << std::setprecision(4) << time 
          << " -> " << time + deltat << ", deltat = " << deltat << ":" << std::flush;

    // Take the time step
    time += deltat;
    
    t0 = std::chrono::high_resolution_clock::now();
    assemble_rhs(time);
    t1 = std::chrono::high_resolution_clock::now();
    time_assemble_rhs += t1 - t0;

    t0 = std::chrono::high_resolution_clock::now();
    solve_time_step();
    t1 = std::chrono::high_resolution_clock::now();
    time_solve_step += t1 - t0;
    
    pcout << std::endl;

    // Apply time adaptivity after computing the solution
    bool step_accepted = true;
    if (enable_time_adaptivity && time_step % time_adapt_interval == 0)
    {
      step_accepted = adapt_time_step(time, solution_before_step, next_deltat);
      
      if (!step_accepted)
      {
        // Reject the step
        pcout << "    Step REJECTED - redoing with smaller deltat" << std::endl;
        solution_owned = solution_before_step;
        solution = solution_owned;
        time = time_before_step;
        --time_step; // Don't count rejected step
        --n_time_steps;
        
        deltat = next_deltat;
        // Reassemble matrices with new deltat
        lhs_matrix.copy_from(mass_matrix);
        lhs_matrix *= (1.0 / deltat);
        lhs_matrix.add(theta, stiffness_matrix);
        rhs_matrix.copy_from(mass_matrix);
        rhs_matrix *= (1.0 / deltat);
        rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
      }
      else if (next_deltat != deltat)
      {
        // Step accepted, but deltat changed for next step
        deltat = next_deltat;
        // Reassemble matrices with new deltat
        lhs_matrix.copy_from(mass_matrix);
        lhs_matrix *= (1.0 / deltat);
        lhs_matrix.add(theta, stiffness_matrix);
        rhs_matrix.copy_from(mass_matrix);
        rhs_matrix *= (1.0 / deltat);
        rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
      }
    }

  }
  output(9999);
  auto t_total_end = std::chrono::high_resolution_clock::now();
  time_total = t_total_end - t_total_start;

  // Compute and print performance metrics
  compute_and_print_metrics();
}

void
Heat::compute_and_print_metrics() const
{

  const double total_time_local = time_total.count();
  const double n_dofs = dof_handler.n_dofs();
  const double h_min_local = GridTools::minimal_cell_diameter(mesh);
  const double total_time = Utilities::MPI::max(total_time_local, MPI_COMM_WORLD);
  const double h_min      = Utilities::MPI::min(h_min_local, MPI_COMM_WORLD);

  // Metric calculations
  const double r_res       = h_min / n_dofs;
  const double r_t_per_dof = total_time / n_dofs;

  // Print performance metrics
  pcout << "\n===============================================" << std::endl;
  pcout << "=== Performance Metrics Summary ===" << std::endl;
  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Total Wall-clock time (t):              " << total_time << " s" << std::endl;
  pcout << "Final Degrees of Freedom (n_Omega):     " << n_dofs << std::endl;
  pcout << "Minimum cell diameter (h):              " << h_min << std::endl;
  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Resolution & Resource Metrics:" << std::endl;
  pcout << "  - r_res (h / n_Omega):                " << r_res << std::endl;
  pcout << "  - r_t-per-DOF (t / n_Omega):          " << r_t_per_dof << " s/DOF" << std::endl;
  pcout << "===============================================" << std::endl;
}
