Genesis Scene Construction
Both approaches: You create scenes with code, but can load geometry/robots from files.

Basic Pattern

import genesis as gs

# 1. Initialize backend
gs.init(backend=gs.gpu)

# 2. Create scene with options
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)

# 3. Add entities (code + file loading)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(file="robot.urdf"))
box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 1)))

# 4. Build (required before simulation)
scene.build()

# 5. Simulate
for i in range(1000):
    scene.step()
What You Can Load From Files
Format	Morph Class	Use For
.urdf	gs.morphs.URDF()	Robots (ROS standard)
.xml	gs.morphs.MJCF()	Robots (MuJoCo format)
.obj, .stl, .glb	gs.morphs.Mesh()	3D objects
.usd	gs.morphs.USD()	Full scenes
What You Create In Code
Primitives:


gs.morphs.Plane()
gs.morphs.Box(size=(0.1, 0.1, 0.1))
gs.morphs.Sphere(radius=0.05)
gs.morphs.Cylinder(radius=0.05, height=0.2)
With materials/surfaces:


scene.add_entity(
    morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)),
    material=gs.materials.Rigid(rho=500),  # density
    surface=gs.surfaces.Rough(color=(1, 0, 0, 1)),  # red
)
Save/Load Scene State

# Save physics state (pickle)
scene.save_checkpoint("checkpoint.pkl")

# Load physics state
scene.load_checkpoint("checkpoint.pkl")

# Get/set state programmatically
state = scene.get_state()
scene.reset(state=state)
Batched Environments

scene.build(n_envs=4096)  # 4096 parallel copies

# Set per-environment values
robot.set_qpos(positions)  # shape: (4096, n_dofs)