# Genesis: Performance & Rendering Explained

Why Genesis is fast, whether it uses AI, and how rendering works.

---

## Why Genesis is So Fast

### GPU-First Architecture (Not Traditional CPU Physics)

Genesis uses **GSTaichi** (a Taichi fork) - all physics kernels compile directly to GPU code:

```python
@ti.kernel  # This compiles to CUDA/Vulkan/Metal
def solve_physics():
    # Runs on thousands of GPU threads in parallel
```

**Key difference from MuJoCo/PyBullet/etc.:** Those run physics on CPU, then optionally visualize on GPU. Genesis runs *everything* on GPU.

### Massive Batching

```python
scene.build(n_envs=4096)  # 4096 parallel simulations on ONE GPU
```

This is why they claim **43 million FPS** - it's running thousands of robot simulations simultaneously, not one fast simulation.

### Zero-Copy Memory with PyTorch

Direct memory sharing between physics and ML via DLPack - no CPU↔GPU copies:

```
Taichi field ←→ PyTorch tensor (same GPU memory)
```

### Other Optimizations

- **Contact islands** - Groups colliding bodies, solves independently in parallel
- **Shared memory** - GPU shared memory for constraint solving
- **Sparse solving** - Only computes relevant DOFs
- **Warp reduction** - Fast matrix factorization using GPU warp-level operations
- **Memory layout optimization** - Cache-friendly data structures

---

## Does It Use AI?

**No, the physics itself is not AI-based.** It's traditional numerical physics:
- Rigid body dynamics (Articulated Body Algorithm)
- Collision detection (GJK, EPA algorithms)
- Constraint solving (Newton's method, Conjugate Gradient)

However, Genesis is **designed for AI/ML workflows**:
- Differentiable simulation (gradients flow through physics)
- Zero-copy PyTorch integration
- Batch environments for reinforcement learning
- Direct tensor access for neural network training

The speed comes from GPU parallelization, not neural networks.

---

## Rendering: Both Triangles AND Ray Tracing

Genesis has **three rendering backends**:

| Backend | Technology | Method | Use Case |
|---------|------------|--------|----------|
| **Rasterizer** | PyOpenGL + GLSL shaders | Triangle rasterization | Fast training, real-time |
| **Ray Tracer** | LuisaRender (C++) | Monte Carlo path tracing | Photo-realistic output |
| **Batch Renderer** | Madrona | GPU batch rasterization | Parallel env rendering |

### Rasterization (Default)

Traditional triangle rendering pipeline:
- Vertex processing → Triangle assembly → Rasterization → Fragment shading
- Shadow mapping
- Plane reflections
- Normal maps
- Depth buffer

**Files:** `Genesis/genesis/vis/rasterizer.py`, `rasterizer_context.py`

### Ray Tracing (Optional)

Photo-realistic rendering using LuisaRender:
- Monte Carlo path tracing
- Global illumination
- Soft shadows
- Built-in denoising
- Environment mapping (HDR)

**File:** `Genesis/genesis/vis/raytracer.py`

### Usage Example

```python
import genesis as gs

# Rasterization (fast, good for training)
scene = gs.Scene(
    viewer_options=gs.ViewerOptions(
        renderer='rasterizer',
        res=(1280, 720)
    )
)

# Ray tracing (slow, photo-realistic)
scene = gs.Scene(
    viewer_options=gs.ViewerOptions(
        renderer='raytracer',
        res=(1920, 1080),
        max_FPS=30
    )
)
```

### Batch Rendering

For parallel environments, Madrona batch renderer:

```python
scene.build(n_envs=1024)
# All 1024 environments rendered in parallel on GPU
```

---

## Performance Comparison

### Traditional Physics Engines (MuJoCo, PyBullet, etc.)

```
CPU Physics → CPU Memory → GPU Memory → GPU Render
     ↑                ↑
  Bottleneck     Bottleneck
```

- Physics runs on CPU (single-threaded or limited parallelism)
- Data must be copied to GPU for rendering
- Each environment runs sequentially

### Genesis Architecture

```
GPU Physics ←→ GPU Memory ←→ GPU Render
         (zero-copy)    (zero-copy)
```

- Physics compiles to GPU kernels (CUDA/Vulkan/Metal)
- All data stays on GPU
- Thousands of environments run in parallel

---

## The "43 Million FPS" Claim Explained

This number is **aggregate FPS across batched environments**, not single-environment speed:

```
43,000,000 FPS = ~4,000 environments × ~10,000 steps/second
```

A single environment runs at ~10K Hz (still very fast), but the headline number comes from massive parallelization.

**Why this matters for robotics/ML:**
- Reinforcement learning needs millions of environment steps
- Training a policy might need 10^9 steps
- At 43M FPS, that's ~23 seconds of simulation time
- Traditional engines at 1000 FPS would take ~12 days

---

## Technology Stack Summary

| Component | Technology | Location |
|-----------|------------|----------|
| Compute Backend | GSTaichi (Taichi fork) | `genesis/__init__.py` |
| Physics Kernels | `@ti.kernel` decorators | `genesis/engine/solvers/` |
| Collision Detection | GJK + EPA on GPU | `solvers/rigid/collider/` |
| Constraint Solving | Island-based MLCP | `solvers/rigid/constraint/` |
| Rasterization | PyOpenGL + PyRender | `genesis/vis/rasterizer.py` |
| Ray Tracing | LuisaRender (C++) | `genesis/vis/raytracer.py` |
| Batch Rendering | Madrona | `genesis/vis/batch_renderer.py` |
| ML Integration | DLPack (zero-copy) | Throughout codebase |

---

## Quick Reference

| Question | Answer |
|----------|--------|
| Why fast? | GPU-compiled physics + batch parallelization |
| Uses AI? | No - traditional physics, designed for ML integration |
| Rendering? | Both: rasterization (triangles) + ray tracing (optional) |
| 43M FPS? | Aggregate across ~4000 parallel environments |
| Backends? | CUDA, Vulkan, Metal, CPU |
