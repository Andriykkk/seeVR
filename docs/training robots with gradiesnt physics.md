Gradients Through Physics
It means: backpropagation through the physics simulation, like through a neural network.


Neural Network:           Physics Simulation:

input → layer → layer → output    state → step → step → result
          ↓       ↓                         ↓       ↓
       gradients flow                  gradients flow
          ↓       ↓                         ↓       ↓
        update weights               update controller
Example: Robot Throwing Ball
Goal: Learn initial throw velocity to hit target.


# Forward pass (physics simulation)
velocity = initial_velocity  # parameter we want to optimize
for t in range(100):
    position += velocity * dt
    velocity += gravity * dt
    
final_position = position

# Loss: how far from target?
loss = distance(final_position, target)

# Backward pass (gradients through physics!)
loss.backward()

# Now we have: d(loss) / d(initial_velocity)
# "If I throw faster, ball lands further"

# Update
initial_velocity -= learning_rate * initial_velocity.grad
What's Being Learned?
NOT learning physics itself. Physics is fixed equations.

Learning: Parameters that interact with physics.

What's Learned	Example
Control policy	Robot joint torques
Initial conditions	Throw velocity, starting pose
Physical parameters	Friction, mass (system ID)
Trajectory	Path for robot arm
How Gradients Flow Through Physics
Each physics step is differentiable:


# Forward step (normal physics)
velocity_new = velocity + force/mass * dt
position_new = position + velocity_new * dt

# Backward (chain rule, automatic)
d_position/d_force = d_position/d_velocity * d_velocity/d_force
                   = dt * (dt/mass)
                   = dt²/mass
Collision is tricky:


# Non-differentiable (hard switch)
if penetration > 0:
    apply_impulse()  # gradient = 0 or undefined

# Genesis uses soft/smoothed contacts for gradients
Visual: Training Robot With Differentiable Physics

                    ┌─────────────────────────────────┐
                    │     DIFFERENTIABLE PHYSICS      │
                    │                                 │
action ──┬──────────▶  state₀ → state₁ → ... → stateₙ ──▶ loss
         │          │     ↑        ↑             ↑    │     │
         │          │     │        │             │    │     │
         │          └─────│────────│─────────────│────┘     │
         │                │        │             │          │
         │          ┌─────┴────────┴─────────────┴──────────┘
         │          │              gradients
         │          ▼
         │    ┌───────────┐
         └────│  Policy   │◀─── update weights
              │  Network  │
              └───────────┘
Without differentiable physics (RL):

Run simulation, get reward
Guess which action was good (high variance)
Needs millions of samples
With differentiable physics:

Run simulation, get loss
Exact gradient: "this action → this result"
Needs thousands of samples (1000x more efficient)
Simple Example

import torch

# Physics parameters
mass = 1.0
gravity = -9.81
dt = 0.01

# Learnable: initial velocity to reach target
initial_vel = torch.tensor([10.0], requires_grad=True)
target_pos = torch.tensor([50.0])

# Forward: simulate projectile
pos = torch.tensor([0.0])
vel = initial_vel.clone()

for _ in range(100):
    vel = vel + gravity * dt      # differentiable
    pos = pos + vel * dt          # differentiable

# Loss
loss = (pos - target_pos) ** 2

# Backward: gradient through all 100 physics steps
loss.backward()

print(initial_vel.grad)  # How to adjust throw to hit target
Why Newton > PGS for Gradients

PGS (50 iterations):
state → iter₁ → iter₂ → ... → iter₅₀ → result
           ↓       ↓              ↓
        gradient noise accumulates
        
Newton (5 iterations):
state → iter₁ → iter₂ → ... → iter₅ → result
           ↓       ↓            ↓
        cleaner gradient path
Fewer iterations = less numerical noise in gradients.

Summary
Question	Answer
Does model learn physics?	No, physics is fixed
What's learned?	Controllers, trajectories, parameters
How?	Backprop through simulation (chain rule)
Why useful?	1000x more sample efficient than RL
Why Genesis uses it?	Training robot policies faster
