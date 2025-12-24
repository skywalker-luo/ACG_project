"""
刚体动力学模块
实现刚体的物理属性、状态更新和积分器
支持线性和角运动的显式/半隐式欧拉积分
"""

import numpy as np
from typing import Optional, Tuple
import warnings
from .mesh_property import (
    MeshPropertyCalculator,
    validate_mesh_for_physics
)

class RigidBody:
    """
    刚体类，包含质量、惯性、位置、速度等物理属性
    使用半隐式欧拉积分进行状态更新
    """
    
    # 不需要给定 mass, inertia
    def __init__(self, 
                 mesh=None,
                 density: float = None,
                 position: np.ndarray = None,
                 orientation: np.ndarray = None, # 初始方向（四元数 [w,x,y,z]）
                 velocity: np.ndarray = None,
                 angular_velocity: np.ndarray = None,
                 restitution: float = 0.5,
                 friction: float = 0.3,
                 is_static: bool = False):
        """
        初始化刚体
        """
        self.mesh = mesh
        self.density = density
        self.is_static = is_static
        
        if is_static:
            self.mass = float('inf')
            self.inv_mass = 0.0
        else:
            calculator = MeshPropertyCalculator(mesh, density=self.density)
            volume, mass, center, inertia = calculator.compute_mesh_properties()
            self.mass = float(mass)
            self.inv_mass = 1.0 / self.mass
            self.local_inertia = inertia
            self.inv_inertia = np.linalg.inv(self.local_inertia)
            self.volume = volume
            
            # 将网格顶点平移,使质心位于局部坐标系原点
            # 这样world_position就直接对应质心位置
            if center is not None and np.linalg.norm(center) > 1e-10:
                mesh.vertices -= center
                # 质心现在在原点
                self.center = np.array([0.0, 0.0, 0.0])
            else:
                self.center = center
        
        # 材质属性
        self.restitution = restitution
        self.friction = friction
        
        # 位置和姿态
        self.position = np.array([0.0, 0.0, 0.0]) if position is None else np.array(position)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0]) if orientation is None else np.array(orientation)  # [w,x,y,z]
        
        # 线性运动
        self.velocity = np.array([0.0, 0.0, 0.0]) if velocity is None else np.array(velocity)
        self.force = np.array([0.0, 0.0, 0.0])  # 力累积器
        
        # 角运动
        self.angular_velocity = np.array([0.0, 0.0, 0.0]) if angular_velocity is None else np.array(angular_velocity)
        self.torque = np.array([0.0, 0.0, 0.0])  # 力矩累积器
        
        # 变换矩阵缓存
        self._transform_matrix = None
        self._world_inertia = None
        self._inv_world_inertia = None
        self._dirty_transform = True
        
        # 碰撞状态跟踪
        self.is_colliding = False
        self.colliding_with = set()  # 存储当前正在碰撞的刚体ID
    
    def get_mesh_properties(self) -> np.ndarray:
        """
        获取mesh性质
        """
        return self.volume, self.mass, self.center, self.local_inertia
    
    def get_transform_matrix(self) -> np.ndarray:
        """
        获取4x4变换矩阵
        """
        if self._dirty_transform or self._transform_matrix is None:
            self._update_transform_matrix()
            self._dirty_transform = False
        return self._transform_matrix
    
    def _update_transform_matrix(self):
        """
        更新变换矩阵
        由于质心已经在局部原点,变换简化为: world_v = R @ v + position
        """
        # 四元数转旋转矩阵
        rotation_matrix = self._quaternion_to_matrix(self.orientation)
        
        # 构建4x4变换矩阵
        self._transform_matrix = np.eye(4)
        self._transform_matrix[:3, :3] = rotation_matrix
        self._transform_matrix[:3, 3] = self.position
    
    def get_world_inertia(self) -> np.ndarray:
        """
        获取世界坐标系下的惯性张量
        I_world = R * I_local * R^T
        """
        if self._dirty_transform or self._world_inertia is None:
            rotation_matrix = self._quaternion_to_matrix(self.orientation)
            self._world_inertia = rotation_matrix @ self.local_inertia @ rotation_matrix.T
            
            if not self.is_static:
                self._inv_world_inertia = rotation_matrix @ self.inv_inertia @ rotation_matrix.T
            else:
                self._inv_world_inertia = np.zeros((3, 3))
                
        return self._world_inertia
    
    def get_inv_world_inertia(self) -> np.ndarray:
        """
        获取世界坐标系下的惯性张量逆矩阵
        """
        if self._dirty_transform or self._inv_world_inertia is None:
            self.get_world_inertia()  # 这会计算两个矩阵
        return self._inv_world_inertia
    
    @staticmethod
    def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        """
        四元数转旋转矩阵
        q = [w, x, y, z]
        """
        w, x, y, z = q
        
        # 归一化
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-10:
            return np.eye(3)
        
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        四元数乘法
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def apply_force(self, force: np.ndarray, point: np.ndarray = None):
        """
        施加力
        
        Args:
            force: 力向量 (3,)
            point: 作用点（世界坐标），如果为None则作用在质心
        """
        if self.is_static:
            return
            
        self.force += force
        
        if point is not None:
            # 计算力矩 τ = r × F
            r = point - self.position
            torque = np.cross(r, force)
            self.torque += torque
    
    def apply_torque(self, torque: np.ndarray):
        """
        施加力矩
        """
        if self.is_static:
            return
        self.torque += torque
    
    def apply_impulse(self, impulse: np.ndarray, point: np.ndarray = None):
        """
        施加冲量
        
        Args:
            impulse: 冲量向量 (3,)
            point: 作用点（世界坐标），如果为None则作用在质心
        """
        if self.is_static:
            return
            
        # 线性冲量
        self.velocity += impulse * self.inv_mass
        
        if point is not None:
            # 角冲量
            r = point - self.position
            angular_impulse = np.cross(r, impulse)
            self.angular_velocity += self.get_inv_world_inertia() @ angular_impulse
    
    def get_velocity_at_point(self, point: np.ndarray) -> np.ndarray:
        """
        获取刚体上某点的速度
        v_point = v_cm + ω × r
        
        Args:
            point: 世界坐标中的点
            
        Returns:
            该点的速度向量
        """
        r = point - self.position
        return self.velocity + np.cross(self.angular_velocity, r)
    
    def clear_forces(self):
        """
        清除累积的力和力矩
        """
        self.force.fill(0.0)
        self.torque.fill(0.0)
    
    def integrate_forces(self, dt: float):
        """
        第一阶段积分：将力积分到速度
        这是World两阶段积分的第一阶段，只更新速度不更新位置
        """
        if self.is_static:
            return
            
        # 线性运动：F = ma -> a = F/m -> v += a*dt
        acceleration = self.force * self.inv_mass
        self.velocity += acceleration * dt
        
        # 角运动：τ = Iα -> α = I^(-1)τ -> ω += α*dt
        inv_inertia = self.get_inv_world_inertia()
        angular_acceleration = inv_inertia @ self.torque
        self.angular_velocity += angular_acceleration * dt
    
    def integrate_velocities(self, dt: float):
        """
        第二阶段积分：将速度积分到位置
        这是World两阶段积分的第二阶段，使用更新后的速度来更新位置
        """
        if self.is_static:
            return
            
        # 应用阻尼
        linear_damping = 0.995  # 线性速度保留99.5%
        angular_damping = 0.98  # 角速度保留98%
        self.velocity *= linear_damping
        self.angular_velocity *= angular_damping
        
        # 线性运动：x += v*dt
        self.position += self.velocity * dt
        
        # 角运动：更新四元数方向
        self._integrate_orientation(dt)
        
        # 标记变换矩阵需要更新
        self._dirty_transform = True
    
    def integrate(self, dt: float, integration_method: str = 'semi_implicit'):
        """
        积分更新刚体状态
        
        Args:
            dt: 时间步长
            integration_method: 积分方法 ('explicit', 'semi_implicit')
        """
        if self.is_static:
            return
        
        if integration_method == 'semi_implicit':
            self._semi_implicit_euler(dt)
        else:
            self._explicit_euler(dt)
        
        # 标记变换矩阵需要更新
        self._dirty_transform = True
    
    def _explicit_euler(self, dt: float):
        """
        显式欧拉积分
        v(t+dt) = v(t) + a(t) * dt
        x(t+dt) = x(t) + v(t) * dt
        """
        # 线性运动
        acceleration = self.force * self.inv_mass
        self.position += self.velocity * dt
        self.velocity += acceleration * dt
        
        # 角运动
        inv_inertia = self.get_inv_world_inertia()
        angular_acceleration = inv_inertia @ self.torque
        
        # 更新角速度
        self.angular_velocity += angular_acceleration * dt
        
        # 更新四元数方向
        self._integrate_orientation(dt)
    
    def _semi_implicit_euler(self, dt: float):
        """
        半隐式欧拉积分（辛欧拉）
        v(t+dt) = v(t) + a(t) * dt
        x(t+dt) = x(t) + v(t+dt) * dt
        """
        # 线性运动
        acceleration = self.force * self.inv_mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # 角运动
        inv_inertia = self.get_inv_world_inertia()
        angular_acceleration = inv_inertia @ self.torque
        
        # 更新角速度
        self.angular_velocity += angular_acceleration * dt
        
        # 更新四元数方向
        self._integrate_orientation(dt)
    
    def _integrate_orientation(self, dt: float):
        """
        积分四元数方向
        使用角速度更新四元数
        """
        if np.linalg.norm(self.angular_velocity) < 1e-10:
            return
        
        # 调试：打印旋转前后的四元数
        old_orientation = self.orientation.copy()
        
        # 角速度转四元数微分
        # dq/dt = 0.5 * ω_quat * q
        omega_quat = np.array([0.0, self.angular_velocity[0], 
                              self.angular_velocity[1], self.angular_velocity[2]])
        
        # 四元数微分
        q_dot = 0.5 * self._quaternion_multiply(omega_quat, self.orientation)
        
        # 更新四元数
        self.orientation += q_dot * dt
        
        # 归一化四元数
        norm = np.linalg.norm(self.orientation)
        if norm > 1e-10:
            self.orientation /= norm
        else:
            self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
    
    def set_position(self, position: np.ndarray):
        """
        设置位置
        """
        self.position = np.array(position)
        self._dirty_transform = True
    
    def set_orientation(self, orientation: np.ndarray):
        """
        设置方向（四元数）
        """
        self.orientation = np.array(orientation)
        # 归一化
        norm = np.linalg.norm(self.orientation)
        if norm > 1e-10:
            self.orientation /= norm
        self._dirty_transform = True
    
    def set_velocity(self, velocity: np.ndarray):
        """
        设置线速度
        """
        if not self.is_static:
            self.velocity = np.array(velocity)
    
    def set_angular_velocity(self, angular_velocity: np.ndarray):
        """
        设置角速度
        """
        if not self.is_static:
            self.angular_velocity = np.array(angular_velocity)
    
    def get_kinetic_energy(self) -> float:
        """
        计算动能 = 0.5 * m * v² + 0.5 * ω^T * I * ω
        """
        if self.is_static:
            return 0.0
        
        linear_ke = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        angular_ke = 0.5 * np.dot(self.angular_velocity, 
                                  self.get_world_inertia() @ self.angular_velocity)
        return linear_ke + angular_ke
    
    def get_aabb(self):
        """
        获取轴对齐包围盒（如果有mesh的话）
        """
        transform = self.get_transform_matrix()
        return self.mesh.get_aabb(transform)
    
    def __repr__(self) -> str:
        """
        字符串表示
        """
        static_str = " (STATIC)" if self.is_static else ""
        return (f"RigidBody{static_str}:\n"
                f"  Position: {self.position}\n"
                f"  Velocity: {self.velocity}\n"
                f"  Mass: {self.mass}\n"
                f"  KE: {self.get_kinetic_energy():.3f}")

def create_mesh_body(mesh, density: float = 1.0, **kwargs) -> RigidBody:
    """
    从网格创建刚体的便捷函数
    自动计算质量和惯性张量
    """
    # 验证网格
    is_valid, error_msg = validate_mesh_for_physics(mesh)
    if not is_valid:
        print(f"警告: {error_msg}")
    
    # 直接在构造时传入 density，RigidBody 会使用 MeshPropertyCalculator 计算质量与惯性
    body = RigidBody(mesh=mesh, density=density, **kwargs)
    return body
