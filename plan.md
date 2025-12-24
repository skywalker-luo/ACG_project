my_physics_engine/
│
├── main.py                         # 入口，创建世界、添加物体、开始 simulation
│
├── config/
│   └── settings.py                 # 全局参数（时间步、迭代次数等）
│
├── geometry/
│   ├── mesh.py                     # 加载 mesh (OBJ/PLY)
│   ├── aabb.py                     # AABB 包围盒
│   ├── bvh.py                      # BVH 加速结构（可选）
│   └── sdf.py                      # Signed Distance Field (可选)
│
├── collision/
│   ├── broadphase.py               # AABB overlap 检测
│   ├── narrowphase.py              # triangle-triangle 或 SDF narrowphase
│   └── contact.py                  # contact point 数据结构
│
├── dynamics/
│   ├── rigid_body.py               # 刚体积分器（半隐式/显式 Euler）
│   ├── constraint_solver.py        # impulse-based 碰撞响应
│   └── world.py                    # 整个物理世界（Bullet 风格）
│
├── io/
│   └── visualizer.py               # open3D 或 pyglet 可视化
│
└── utils/
    ├── timer.py
    └── logger.py

🧩 math/（基础数学工具）
不要自己手写！直接用numpy、scipy！

🧩 geometry/（几何与网格）
mesh.py

vertices（Nx3 numpy 数组）

faces（Mx3 numpy 数组）

compute_aabb()

转成 BVH 或 SDF

aabb.py

min、max、center、half_extent

overlap(other)

bvh.py（可选）

构造从 mesh 的 AABB 树

用于快速检测可能碰撞的三角形对

sdf.py（可选）

voxel grid 存储 signed distance

sample(point)

gradient(point)

🧩 collision/（碰撞检测）
broadphase.py

AABB vs AABB

返回“可能碰撞的物体对”

narrowphase.py

triangle-triangle intersection（可用 CGAL 或自己写 SAT）

或 使用 SDF：若 φ(p) < 0 → 发生 penetration

返回 Contact(contact_point, normal, penetration)

contact.py

数据结构：

class Contact:
    point: Vec3
    normal: Vec3
    penetration: float
    bodyA
    bodyB

🧩 dynamics/（物理系统）
rigid_body.py

包含：

mass

inertia

velocity

angular_velocity

force, torque accumulator

integrate(dt)

constraint_solver.py

刚体碰撞法向冲量公式

$$
J = \frac{-(1 + e) \cdot v_n}
{\frac{1}{m_1} + \frac{1}{m_2} + \hat{n} \cdot 
((I_1^{-1} (\vec{r_1} \times \hat{n}) \times \vec{r_1}) +
(I_2^{-1} (\vec{r_2} \times \hat{n}) \times \vec{r_2}))}
$$

变量说明：

| 符号 | 含义 |
|------|------|
| $J$ | 法向冲量大小 |
| $e$ | 恢复系数（弹性系数） |
| $v_n$ | 法向相对速度，$v_n = (\vec{v_1} - \vec{v_2}) \cdot \hat{n}$ |
| $m_1, m_2$ | 两个刚体的质量 |
| $\vec{r_1}, \vec{r_2}$ | 从质心到接触点的向量 |
| $\hat{n}$ | 接触法线（单位向量） |
| $I_1, I_2$ | 刚体在世界坐标系下的惯性张量 |
| $I_1^{-1}, I_2^{-1}$ | 惯性张量的逆 |

world.py（最核心部分）

仿 Bullet 的 btDiscreteDynamicsWorld

class World:
    def __init__(self):
        self.bodies = []
        self.broadphase = Broadphase()
        self.solver = ConstraintSolver()

    def step(self, dt):
        self.integrate_forces(dt)
        pairs = self.broadphase.compute_pairs(self.bodies)
        contacts = narrowphase(pairs)
        self.solver.solve(contacts)
        self.integrate_velocities(dt)

🧩 io/（可视化与 mesh I/O）
obj_loader.py

tinyobjloader-python

返回 Mesh(vertices, faces)

visualizer.py

用 Open3D 或 pyglet 显示 rigid body transform

🟩 main.py（程序主入口）

示例结构：

from dynamics.world import World
from dynamics.rigid_body import RigidBody
from io.obj_loader import load_mesh

world = World()

mesh1 = load_mesh("bunny.obj")
mesh2 = load_mesh("cube.obj")

body1 = RigidBody(mesh1, mass=1.0)
body2 = RigidBody(mesh2, mass=1.0)

world.add_body(body1)
world.add_body(body2)

while True:
    world.step(1/60.0)
    visualize(world)