# Compliant Actuator: robotic elastic actuator-inspired Interaction and PD-based acceleration Controller

In this project, the interaction between the user-controlled points (e.g., hand tracks from video) and the deformable object is modeled as a **Compliant Actuator** using a soft-coupling PD (Proportional-Derivative) mechanism.

---

### 1. Motivation: Why Compliant Actuators? (Motivation)

In the context of real-to-sim System Identification, directly applying **Kinematic Constraints** (snapping particles to target trajectories) often leads to severe numerical issues. Our debugging experience across Cloth, Rope, and Softbody materials highlights the necessity of **Compliant Actuators**:

1.  **Avoiding Numerical Shock & Particle Tearing**: Rigidly forcing a group of particles to move at a fixed velocity creates an infinite "impulse" at the contact boundary. In materials like thin Cloth or sparse Rope, this sudden impact causes **localized stress concentrations** that exceed the stability limits of the MPM grid, resulting in "particle tearing" (non-physical separation) or catastrophic simulation explosions.
2.  **Smoothing the Cost Landscape for Sampling**: Hard constraints introduce sharp, discontinuous jumps in the objective function whenever a constraint is violated. For **sampling-based (zero-order) optimizers**, these "numerical cliffs" make exploration extremely difficult. A compliant actuator uses soft springs to "blur" these boundaries, creating a smooth, continuous **Cost Landscape** that allows the optimizer to perceive gradual improvements in parameter selection.
3.  **Physical Low-Pass Filtering**: Real-world hand tracking data from videos is inevitably noisy. Directly imposing this noise as a velocity constraint injects high-frequency energy into the simulator, which is the primary driver of numerical instability. The compliant actuator acts as a **natural physical filter**, absorbing tracking jitter through its spring-damper mechanism and preventing it from "poisoning" the simulation.
4.  **Implicit Survival Rate Management**: By decoupling hand motion from the material's internal integrator, the compliant actuator allows the material to "slip" or "resist" when a sampled set of physical parameters is physically inconsistent. This ensures the simulation **survives the entire sequence** even with suboptimal parameters, providing meaningful feedback to the optimizer rather than a simple "NaN" failure.

---

### 2. Methodology: What & How (What & How)

#### The Coupling Mechanism (What)
The actuator establishes a virtual "spring-damper" connection between a controller point $\mathbf{x}_c$ and its neighboring object particles $\mathbf{x}_p$. The force (expressed as a velocity change $\Delta \mathbf{v}$) applied to the particle is:

$$
\Delta \mathbf{v} = \left[ k_p (\mathbf{x}_{target} - \mathbf{x}_p - \mathbf{o}_{init}) + k_d (\mathbf{v}_c - \mathbf{v}_p) \right] \cdot \Delta t
$$

Where:
*   $k_p$: **Controller Stiffness**, the strength of the "grip."
*   $k_d$: **Controller Damping**, suppresses oscillations and noise.
*   $\mathbf{o}_{init}$: Initial offset to preserve the local geometry at the moment of contact.

#### Refined Implementation (How)

To ensure stability and accuracy across diverse materials (Cloth, Rope, Softbody), the following refinements were implemented:

1.  **Precision Radius Grabbing**: Instead of grabbing the whole object, the `controller_radius` is tuned to match the actual contact area (e.g., 0.10 for corners). This prevents "ghost forces" from lifting the center of the object when only the corners are touched.
2.  **Soft Force Normalization**: To prevent a single particle from "overheating" when connected to hundreds of controller points, we apply a square-root normalization:
    $$\mathbf{f}_{final} = \frac{\sum \mathbf{f}_i}{\sqrt{N_{connections}}}$$
    This keeps the gradient signal strong while preventing localized numerical explosions.
3.  **Velocity Clamping (Headroom Management)**: We introduce `controller_clamp_dv` to limit the maximum velocity change per step. 
    *   *Low Clamp (0.5)*: High stability but causes "lag" in high-speed movements.
    *   *High Clamp (2.0)*: Allows high-speed tracking but requires smaller time steps or higher damping.
4.  **Staged Warm-up**: The `controller_warmup_frames` gradually ramps up the stiffness $k_p$ during the first few frames, preventing impulsive "shocks" that could break the simulation.

---

### 3. Advantages (Advantages)

*   **Robust Numerical Stability**: By decoupling the controller's motion from the simulator's internal integrator, the system can handle extremely aggressive hand movements without the particles "teleporting" through the grid.
*   **Physical Consistency**: It enforces physical consistency by coupling interaction success to material realism: if the object is too soft to transmit the applied force, it slips or fails to lift—exposing non-physical parameter estimates through increased tracking and shape loss.
*   **Tunable Precision**: The combination of `radius` and `max_neighbors` allows the actuator to switch between "Point-like" interaction (for rope) and "Area-like" interaction (for palms/fingers on softbodies).
*   **Optimization-Friendly Learnable Parameters**: Every parameter of the actuator ($k_p, k_d$) is **learnable** and highly suitable for zero-order optimization (sampling). Their continuous nature ensures that small changes in $k_p$ or $k_d$ lead to predictable, non-catastrophic changes in the simulation outcome, enabling the sampling-based optimizer to efficiently discover the optimal "interaction style" for different materials.

---

### 4. Summary of Debugging Experience

| Problem | Root Cause | Solution |
| :--- | :--- | :--- |
| **Grip Failure (Lag)** | Low `clamp_dv` or low `stiffness` | Increase `clamp_dv` to 2.0 and $k_p$ to 150k. |
| **Explosions** | High `grid_res` or lack of damping | Reduce `grid_res`, increase `damping` to 30+, and lower `clamp_dv` to 1.0. |
| **Middle Arching** | `radius` too large | Shrink `radius` to 0.10 to focus force on contact points. |
| **Detachment** | Over-softened material ($E$) | Set higher `init_raw_E` (-1.0) to ensure force propagation. |
