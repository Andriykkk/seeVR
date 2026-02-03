# Physics Algorithms Comparison

Genesis actually uses **multiple algorithms** - each optimized for different material types. MPM is one of them, but not for everything.

---

## Genesis Multi-Solver Architecture

| Solver | Algorithm | Used For | Realism | Speed |
|--------|-----------|----------|---------|-------|
| **RigidSolver** | Constraint-based (Featherstone + PGS) | Robots, hard objects | Good | Very Fast |
| **MPMSolver** | Material Point Method | Sand, snow, mud, dough | Excellent | Slow |
| **SPHSolver** | Smoothed Particle Hydrodynamics | Water, fluids | Good | Medium |
| **FEMSolver** | Finite Element Method | Elastic solids, soft bodies | Excellent | Slow |
| **PBDSolver** | Position Based Dynamics | Cloth, hair, ropes | Approximate | Fast |

---

## Why Not MPM For Everything?

### MPM Pros:
- Handles **topology changes** (fracture, melting, mixing)
- **Unified** - same algorithm for solid, liquid, gas
- Very realistic for deformables
- Naturally handles **large deformations**

### MPM Cons:
- **10-100x slower** than rigid body solver
- Needs **many particles** for accuracy (millions)
- **Rigid bodies feel "soft"** - hard to make things perfectly stiff
- Requires fine grid resolution

```
Speed comparison (rough):
Rigid solver:    1,000,000 bodies possible
MPM:                10,000 particles typical
SPH:               100,000 particles typical
```

---

## Algorithm Comparison

### For Rigid Bodies (robots, boxes, hard objects):

| Method | Speed | Stiffness | Best For |
|--------|-------|-----------|----------|
| **Constraint-based** (Genesis rigid) | ⚡⚡⚡ | Perfect | Robots, mechanisms |
| **MPM** | ⚡ | Soft | Not ideal |
| **FEM** | ⚡⚡ | Good | Soft robots |

**Genesis choice:** Constraint-based (speed + perfect rigidity)

### For Deformables (sand, snow, dough):

| Method | Speed | Realism | Best For |
|--------|-------|---------|----------|
| **MPM** | ⚡ | ⭐⭐⭐ | Sand, snow, mud |
| **FEM** | ⚡⚡ | ⭐⭐⭐ | Rubber, tissue |
| **PBD** | ⚡⚡⚡ | ⭐⭐ | Games, real-time |

**Genesis choice:** MPM for granular, FEM for elastic

### For Fluids:

| Method | Speed | Realism | Best For |
|--------|-------|---------|----------|
| **SPH** | ⚡⚡ | ⭐⭐ | Splashy water |
| **MPM** | ⚡ | ⭐⭐⭐ | Viscous fluids, mixing |
| **Eulerian (grid)** | ⚡⚡ | ⭐⭐ | Smoke, large volumes |
| **FLIP/PIC** | ⚡⚡ | ⭐⭐⭐ | Ocean, waves |

**Genesis choice:** SPH for general fluids

---

## Would Pure MPM Be More Realistic?

**Yes, for some things:**
- Mixing materials (water + sand = mud)
- Fracture and breaking
- Phase transitions (melting, freezing)
- Extreme deformations

**No, for rigid bodies:**
- MPM can't make things perfectly rigid
- A robot simulated in MPM would feel "jelly-like"
- You'd need insane particle counts

---

## The Real Tradeoff

```
                    REALISTIC
                        ↑
                        |
            MPM/FEM ●   |
                        |
     Genesis Rigid ●    |   ● PBD (games)
                        |
                        +------------------→ FAST
```

**Genesis strategy:** Use the right tool for each material
- Rigid bodies → Constraint solver (fast + stiff)
- Deformables → MPM/FEM (slower but realistic)
- Fluids → SPH (balanced)
- Cloth → PBD (fast, good enough)

---

## If You Want Maximum Realism

**Option 1: MPM for everything** (research/offline)
- Taichi has great MPM examples
- Expect ~1000x slower than Genesis rigid
- Used in movies (Disney Frozen snow)

**Option 2: Genesis approach** (real-time/training)
- Rigid solver for hard things
- MPM only for granular/deformable
- Couple them together

**Option 3: Hybrid**
```
Rigid bodies: Constraint solver
    ↓ (when they break)
Fragments: Switch to MPM
```

---

## For Your Engine

Given you want something Genesis-like:

```
Phase 1: Rigid constraint solver (what we discussed before)
         - Fast, good for robots

Phase 2: Add MPM later (optional)
         - Only for sand/snow/soft materials
         - Keep it separate, couple when needed
```

**Don't start with MPM** if you want robots/rigid bodies. You'll fight the algorithm trying to make things stiff.

---

## Quick MPM Overview

```python
# MPM basic loop
def mpm_step():
    # 1. Particles → Grid (scatter)
    for p in particles:
        grid_cell = get_cell(p.pos)
        grid[cell].mass += p.mass * weight
        grid[cell].momentum += p.mass * p.vel * weight

    # 2. Grid physics
    for cell in grid:
        cell.vel = cell.momentum / cell.mass
        cell.vel += gravity * dt
        # Apply boundary conditions

    # 3. Grid → Particles (gather)
    for p in particles:
        cell = get_cell(p.pos)
        p.vel = interpolate(grid, p.pos)
        p.pos += p.vel * dt

        # Update deformation gradient (stress/strain)
        p.F = update_deformation(p.F, grid_velocity_gradient)
```

MPM's magic is the **deformation gradient F** - it tracks how material has stretched/rotated, enabling realistic stress response.

---

## Is MPM The Most Realistic?

**No, MPM is not universally "most realistic"** - it depends on what you're simulating.

### Realism By Material Type

| Material | Most Realistic Algorithm | Why |
|----------|-------------------------|-----|
| **Rigid bodies** | Constraint-based (Newton-Euler) | Exact rigid motion, perfect stiffness |
| **Elastic solids** | FEM | Exact stress-strain relationships |
| **Granular (sand, snow)** | MPM ✓ | Handles fracture + flow |
| **Water (splashy)** | FLIP/PIC or SPH | Better surface tension, incompressibility |
| **Viscous fluids** | MPM ✓ | Good at honey, lava, mud |
| **Cloth** | FEM or Mass-Spring | Better for thin sheets |
| **Smoke/gas** | Eulerian grid | Incompressible flow equations |
| **Hair** | Discrete Elastic Rods | Specialized for strands |

### MPM Strengths (where it IS most realistic)

- **Material mixing** (water + sand = mud)
- **Phase changes** (melting, freezing)
- **Fracture** (breaking apart)
- **Large deformations** (squishing, stretching)
- **Granular flow** (sand, snow, soil)

### MPM Weaknesses (where others are better)

| Problem | Why MPM Struggles | Better Algorithm |
|---------|-------------------|------------------|
| Rigid bodies | Can't be perfectly stiff | Constraint solver |
| Thin shells | Needs too many particles | FEM shell elements |
| Incompressible fluid | Requires very small dt | FLIP/PIC |
| Cloth | Overkill, slow | PBD or FEM |
| Sharp corners | Particle smearing | Mesh-based FEM |

---

## The "Most Realistic" Hierarchy

```
For GENERAL PURPOSE (one algorithm does everything):
  MPM > SPH > PBD

For SPECIFIC MATERIALS:
  Rigid:     Constraint-based >>> MPM
  Elastic:   FEM ≈ MPM
  Fluid:     FLIP/PIC > SPH > MPM
  Granular:  MPM >>> everything else
  Cloth:     FEM > PBD >> MPM
```

---

## Speed vs Realism

```
             Realistic
                 ↑
        FEM ●    |    ● MPM
                 |
                 |    ● SPH
    Constraint ● |
       solver    |    ● PBD
                 |
                 +----------------→ Fast

Constraint solver: 10M+ bodies
PBD:               1M particles
SPH:               100K particles
FEM:               50K elements
MPM:               10K-100K particles (for real-time)
```

---

## Bottom Line

**MPM is "jack of all trades, master of few":**
- Great for things that **change** (melt, break, mix)
- Overkill/slow for things that **don't change** (rigid robot, simple fluid)

**Genesis approach is smarter:**
- Use specialized solver for each material
- Get best realism AND best speed for each type

**If you only implement ONE algorithm:**
- For **robots/games**: Constraint-based rigid + PBD cloth
- For **VFX/movies**: MPM (speed doesn't matter)
- For **research**: Whatever your paper needs








All Genesis Collision Types

# From Genesis source
GEOM_TYPE:
    PLANE   = 0   # Infinite ground plane
    SPHERE  = 1   # Sphere primitive
    CAPSULE = 2   # Capsule (cylinder + spheres)
    BOX     = 3   # Box primitive
    MESH    = 4   # Convex hull
    SDF     = 5   # Signed distance field
    TERRAIN = 6   # Heightmap
Why Primitives?
Specialized formulas are much faster than general GJK:

Pair	Method	Speed
Sphere-Sphere	Direct formula	⚡⚡⚡⚡
Sphere-Plane	Direct formula	⚡⚡⚡⚡
Sphere-Capsule	Direct formula	⚡⚡⚡⚡
Box-Box	SAT (15 axes)	⚡⚡⚡
Box-Plane	SAT	⚡⚡⚡
Capsule-Capsule	Closest line segments	⚡⚡⚡
Convex-Convex	GJK + EPA	⚡⚡
Convex-SDF	Vertex queries	⚡⚡

# Sphere-sphere: 5 operations
def sphere_sphere(a, b):
    d = length(b.pos - a.pos)
    return d < a.radius + b.radius

# GJK: 50-200 operations + iterations
def gjk(hull_a, hull_b):
    # ... much more complex
Each Type Explained
1. PLANE (infinite ground)

Infinite flat surface

    ↑ normal
    │
────┼────────────── plane
    │

# Super simple collision
def point_vs_plane(point, plane):
    distance = dot(point - plane.origin, plane.normal)
    return distance < 0
2. SPHERE

    ╭───╮
   ╱     ╲
  │   ●   │  ← center + radius
   ╲     ╱
    ╰───╯

# Just distance check
def sphere_vs_sphere(a, b):
    return length(a.pos - b.pos) < a.r + b.r
3. CAPSULE (very useful!)

Cylinder with spherical caps - great for limbs

      ╭───╮
     ╱     ╲
    │       │
    │       │  ← line segment + radius
    │       │
     ╲     ╱
      ╰───╯

# Closest point on line segment
def capsule_vs_capsule(a, b):
    closest_a, closest_b = closest_points_on_segments(a.line, b.line)
    return length(closest_a - closest_b) < a.r + b.r
Why capsules are popular:

Robot arms/legs → capsules
Fingers → capsules
Fast collision
Smooth rolling (no edges)
4. BOX

    ┌─────────┐
   ╱         ╱│
  ┌─────────┐ │
  │         │ │  ← center + half-extents + orientation
  │         │╱
  └─────────┘

# SAT: 15 axes test
def box_vs_box(a, b):
    axes = [
        a.x, a.y, a.z,           # 3 face normals A
        b.x, b.y, b.z,           # 3 face normals B  
        cross(a.x, b.x), ...     # 9 edge combinations
    ]
    for axis in axes:
        if separating(a, b, axis):
            return False
    return True
5. MESH (convex hull)

Arbitrary convex shape

      ╱╲
     ╱  ╲
    ╱    ╲
   ╱      ╲
  ╱────────╲

# GJK + EPA
def convex_vs_convex(a, b):
    if gjk_intersect(a, b):
        return epa_contact(a, b)
6. SDF (signed distance field)

Precomputed 3D grid

┌─┬─┬─┐
│+│+│+│  + outside
├─┼─┼─┤  - inside
│+│-│+│
└─┴─┴─┘

# O(1) query
def point_vs_sdf(point, sdf):
    return sdf.sample(point) < 0
7. TERRAIN (heightmap)

2D grid of heights (for ground)

     ╱╲    ╱╲
    ╱  ╲  ╱  ╲
   ╱    ╲╱    ╲
  ╱            ╲

# Fast height lookup
def point_vs_terrain(point, terrain):
    ground_y = terrain.height(point.x, point.z)
    return point.y < ground_y
What Genesis Uses For What
Object Type	Geometry
Robot links	Capsule, Box, Convex
Gripper fingers	Capsule, Box
Simple objects	Sphere, Box, Capsule
Complex objects	Convex (decomposed)
Ground	Plane or Terrain
Complex static	SDF
Walls	Box or Plane
Collision Matrix
Genesis has specialized handlers for each pair:


         │ Plane │Sphere│Capsule│ Box │Convex│ SDF │Terrain
─────────┼───────┼──────┼───────┼─────┼──────┼─────┼───────
Plane    │   -   │  ✓   │   ✓   │  ✓  │  ✓   │  -  │   -
Sphere   │   ✓   │  ✓   │   ✓   │  ✓  │  ✓   │  ✓  │   ✓
Capsule  │   ✓   │  ✓   │   ✓   │  ✓  │  ✓   │  ✓  │   ✓
Box      │   ✓   │  ✓   │   ✓   │  ✓  │  ✓   │  ✓  │   ✓
Convex   │   ✓   │  ✓   │   ✓   │  ✓  │  ✓   │  ✓  │   ✓
SDF      │   -   │  ✓   │   ✓   │  ✓  │  ✓   │  -  │   -
Terrain  │   -   │  ✓   │   ✓   │  ✓  │  ✓   │  -  │   -
Note: SDF vs SDF not supported (both must have convex)

For Your Engine
Minimum (start here):


✓ Sphere
✓ Plane
✓ Box
✓ Convex hull
Add later:


+ Capsule (great for characters/robots)
+ Convex decomposition
+ Terrain/heightmap (if needed)
Maybe never:


? SDF (only for complex static)

# Dispatch table
def collide(geom_a, geom_b):
    pair = (geom_a.type, geom_b.type)
    
    handlers = {
        (SPHERE, SPHERE): sphere_sphere,
        (SPHERE, PLANE): sphere_plane,
        (SPHERE, BOX): sphere_box,
        (BOX, BOX): box_box,
        (BOX, PLANE): box_plane,
        (CONVEX, CONVEX): gjk_epa,
        (CONVEX, PLANE): convex_plane,
        # ... etc
    }
    
    return handlers[pair](geom_a, geom_b)





Yes, for rigid bodies this is the most realistic practical setup.

Realism Breakdown
Component	What Genesis Uses	Realism
Shape representation	Convex + SDF	⭐⭐⭐ Exact (no approximation)
Collision detection	GJK + EPA	⭐⭐⭐ Mathematically exact
Constraint solver	Newton	⭐⭐⭐ Converges to exact
Integration	Implicit Euler	⭐⭐⭐ Stable + accurate
Combined: Very realistic rigid body physics.

What Affects Realism
1. Shape Accuracy

Raw mesh:              100% accurate shape
Convex decomposition:  ~95-99% accurate (small gaps)
Single convex hull:    ~70-90% accurate (loses concavity)

For most cases: Convex decomposition is "exact enough"
2. Solver Accuracy

Dantzig:     100% exact solution
Newton:      ~99.99% (converges to exact)
PGS:         ~95-99% (approximate)

Newton is effectively exact for practical purposes
3. Time Step

dt = 0.001:  Very accurate, slow
dt = 0.01:   Good balance (Genesis default)
dt = 0.1:    Fast but misses fast collisions

Smaller dt = more realistic but slower
What's NOT Realistic (limitations)
Even with perfect setup, rigid body simulation ignores:

Real Physics	Simulation
Micro-deformation	Objects are perfectly rigid
Sound/vibration	Not simulated
Heat from friction	Not simulated
Surface micro-texture	Smooth surfaces
Air resistance	Usually ignored
Material fatigue	Objects never break
But for robotics/games: These don't matter. Rigid body assumption is fine.

Realism Hierarchy

MORE REALISTIC (slower):
─────────────────────────
│ FEM (deformable solids)      │ ← Objects can bend/squish
│ MPM (everything deforms)     │ ← Even "rigid" things flex
─────────────────────────
│ Newton + Convex/SDF          │ ← Genesis level ✓
│ (perfect rigid bodies)       │
─────────────────────────
│ PGS + Convex                 │ ← Game engine level
│ (approximate rigid bodies)   │
─────────────────────────
LESS REALISTIC (faster)
When Genesis Setup Is "Wrong"

Scenario: Drop rubber ball

Real life:    Ball deforms on impact, bounces back
Genesis:      Ball stays perfectly round, bounces

Solution:     Use FEM/MPM for deformable objects
              (but much slower)


Scenario: Stack 100 boxes perfectly

Real life:    Tiny imperfections cause collapse
Genesis:      Perfect stack forever

Solution:     Add small random perturbations
              (or accept "too perfect" physics)
Summary
Question	Answer
Is Newton + Convex/SDF realistic?	Yes, for rigid bodies
More realistic than PGS?	Yes, Newton converges better
Most realistic possible?	No, FEM/MPM are more realistic
Good enough for robotics?	Yes, industry standard
Good enough for games?	Overkill, PGS is fine
Your setup (Newton + Convex + BVH): Same realism level as Genesis, MuJoCo, Drake - the standard for robotics simulation.