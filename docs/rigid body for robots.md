Regular Objects (boxes, spheres)

Each body is INDEPENDENT

Box A: position, velocity, mass
Box B: position, velocity, mass
Box C: position, velocity, mass

They only interact through COLLISIONS (temporary contacts)
Simple: Just apply forces, integrate, resolve collisions.

Robots (articulated bodies)

Bodies are PERMANENTLY CONNECTED by joints

      Base
        │ ← joint 1 (always connected)
      Link1
        │ ← joint 2 (always connected)
      Link2
        │ ← joint 3 (always connected)
      Gripper

Moving Link1 FORCES Link2 to move
Gravity on Link2 PULLS on Link1
Problem: You can't just simulate each link independently. They're mechanically coupled forever.

The Math Difference
Independent bodies:


# Simple - each body separate
for body in bodies:
    body.acc = body.force / body.mass
    body.vel += body.acc * dt
Articulated chain:


# Complex - all links coupled
# Moving joint 1 affects ALL downstream links
# Force on link 5 propagates to ALL upstream links

# Need to solve system of equations:
# M(q) * q̈ = forces - C(q, q̇)  
#
# Where M is a matrix coupling ALL joints together
Visual Example

INDEPENDENT:              ARTICULATED:
                         
  □    □    □              □──□──□──□
                               │
Each box ignores others    All links connected
until they collide         Always affect each other
Why Featherstone?
Naive approach for robot: O(n³) - solve full matrix
Featherstone ABA: O(n) - exploits tree structure


7-joint robot arm:
  Naive:       343 operations per step
  Featherstone: 7 operations per step (roughly)
For You
No joints? → Don't need Featherstone


Just boxes falling and colliding
→ Simple integration + PGS collision solver
→ This is what you want now
Want robot arm later? → Add Featherstone then


Robot with joints
→ Featherstone for joint dynamics
→ Still use collision solver for contacts with world