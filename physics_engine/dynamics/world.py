"""
物理世界（Physics World）
仿 Bullet 的 btDiscreteDynamicsWorld 实现
负责管理刚体、执行物理仿真步骤、协调碰撞检测与响应
"""

import numpy as np
from typing import List, Optional, Callable
import time
from ..collision.broadphase import Broadphase, SpatialHashBroadphase
from ..collision.narrowphase import Narrowphase
from ..collision.contact import ContactInfo
from .constraint_solver import ConstraintSolver, ImpulseConstraintSolver, create_constraint_solver
from .rigid_body import RigidBody
from ..utils.timer import Timer
from ..utils.logger import Logger


class World:
    """
    离散动力学世界
    
    实现标准的物理仿真循环：
    1. 集成外力 (integrate forces)
    2. 宽相碰撞检测 (broadphase collision detection)
    3. 窄相碰撞检测 (narrowphase collision detection)  
    4. 约束求解 (constraint solving)
    5. 集成速度 (integrate velocities)
    """
    
    def __init__(self, 
                 gravity: np.ndarray = None,
                 broadphase_type: str = 'simple',  # 'simple' or 'spatial_hash'
                 solver_type: str = 'impulse',     # 'basic' or 'impulse'
                 enable_warmstarting: bool = True,
                 max_substeps: int = 10,
                 fixed_timestep: float = 1.0/240.0,  # 240 Hz 内部时间步
                 debug: bool = False):
        """
        初始化物理世界
        
        Args:
            gravity: 重力向量，默认为 [0, -9.81, 0]
            broadphase_type: 宽相算法类型
            solver_type: 约束求解器类型
            enable_warmstarting: 是否启用热启动
            max_substeps: 最大子步数
            fixed_timestep: 固定时间步长
            debug: 调试模式
        """
        # 物理参数
        self.gravity = np.array([0.0, -9.81, 0.0]) if gravity is None else np.array(gravity)
        self.max_substeps = max_substeps
        self.fixed_timestep = fixed_timestep
        self.debug = debug
        
        # 刚体列表
        self.bodies: List[RigidBody] = []
        
        # 碰撞检测组件
        if broadphase_type == 'spatial_hash':
            self.broadphase = SpatialHashBroadphase(cell_size=2.0)
        else:
            self.broadphase = Broadphase()
            
        self.narrowphase = Narrowphase()
        
        # 约束求解器
        if solver_type == 'impulse':
            self.solver = create_constraint_solver(solver_type)
            # ImpulseConstraintSolver默认启用热启动
        else:
            self.solver = create_constraint_solver(solver_type)
        
        # 统计和调试信息
        self.step_count = 0
        self.total_time = 0.0
        self.contacts_count = 0
        self.broadphase_pairs_count = 0
        
        # 性能计时器
        if self.debug:
            self.timer = Timer()
            self.logger = Logger("World")
            
        # 回调函数
        self.pre_step_callback: Optional[Callable] = None
        self.post_step_callback: Optional[Callable] = None
        self.contact_callback: Optional[Callable] = None
    
    def add_body(self, body: RigidBody):
        """添加刚体到世界"""
        if body not in self.bodies:
            self.bodies.append(body)
            if self.debug:
                self.logger.info(f"Added body {id(body)} to world. Total bodies: {len(self.bodies)}")
    
    def remove_body(self, body: RigidBody):
        """从世界移除刚体"""
        if body in self.bodies:
            self.bodies.remove(body)
            if self.debug:
                self.logger.info(f"Removed body {id(body)} from world. Total bodies: {len(self.bodies)}")
    
    def set_gravity(self, gravity: np.ndarray):
        """设置重力"""
        self.gravity = np.array(gravity)
        
    def step(self, dt: float) -> int:
        """
        执行物理仿真步骤（变步长）
        
        Args:
            dt: 时间步长
            
        Returns:
            实际执行的子步数
        """
        if dt <= 0:
            return 0
            
        if self.debug:
            self.timer.start("world_step")
            
        # 调用前置回调
        if self.pre_step_callback:
            self.pre_step_callback(self, dt)
        
        # 清除速度约束标记，确保每个step只施加一次速度约束
        if hasattr(self.solver, 'velocity_constraints_applied'):
            self.solver.velocity_constraints_applied.clear()
        
        # 清除所有刚体的碰撞状态，准备本帧的碰撞检测
        for body in self.bodies:
            body.is_colliding = False
            body.colliding_with.clear()
        
        # 计算需要的子步数
        substeps = min(int(np.ceil(dt / self.fixed_timestep)), self.max_substeps)
        substep_dt = dt / substeps if substeps > 0 else dt
        
        # 执行子步
        # warm-start 只在第一个子步使用，避免过度响应
        for i in range(substeps):
            is_first_substep = (i == 0)
            self._internal_step(substep_dt, warmstart=is_first_substep)
        
        # 更新统计信息
        self.step_count += 1
        self.total_time += dt
        
        # 调试：如果有接触，打印冲量施加情况
        if self.contacts_count > 0 and hasattr(self.solver, '_impulse_log'):
            # 打印标题和基本信息（即使没有施加冲量也打印，用于调试）
            print(f"\n=== Step {self.step_count} (substeps={substeps}, contacts={self.contacts_count}) ===")
            print(f"solve_contacts被调用 {getattr(self.solver, '_solve_call_count', 0)} 次")
            print(f"constraint_data数量: {getattr(self.solver, '_constraint_data_count', 0)}")
            
            # 只在有速度约束冲量或位置约束冲量时才显示详细信息
            has_velocity = self.solver._impulse_log
            velocity_impulses = [log for log in self.solver._impulse_log if log['type'] == 'velocity']
            if velocity_impulses:
                print(f"施加了 {len(velocity_impulses)} 次速度约束冲量")
                for i, log in enumerate(velocity_impulses, 1):
                    print(f"  #{i}: 冲量={log['impulse']:.2f} N·s, vn={log['vn']:.3f} m/s")
                    print(f"        vel_before={log['body1_vel_before']}")
                    if 'vel_just_before' in log:
                        print(f"        vel_just_before_apply={log['vel_just_before']}")
                    if 'vel_just_after' in log:
                        print(f"        vel_just_after_apply={log['vel_just_after']}")
            else:
                print(f"施加了 0 次速度约束冲量")
            
            # 显示位置约束冲量
            if hasattr(self.solver, '_position_impulse_log') and self.solver._position_impulse_log:
                print(f"施加了 {len(self.solver._position_impulse_log)} 次位置约束冲量")
                for i, log in enumerate(self.solver._position_impulse_log):
                    print(f"  #{i+1}: 冲量={log['impulse']:.2f} N·s, 穿透={log['penetration']:.6f}m")
                    print(f"        vel_before={log['vel_before']}")
                    print(f"        target_vel={log['vel_change_target']:.3f}, eff_mass={log['effective_mass']:.2f}")
                    print(f"        lambda_delta={log['lambda_delta']:.2f}, lambda_apply={log['lambda_apply']:.2f}")
            else:
                print(f"施加了 0 次位置约束冲量")
            
            # 重置计数器
            self.solver._solve_call_count = 0
        
        # 调用后置回调
        if self.post_step_callback:
            self.post_step_callback(self, dt)
            
        if self.debug:
            elapsed = self.timer.end("world_step")
            if self.step_count % 60 == 0:  # 每60步输出一次统计
                self.logger.info(f"Step {self.step_count}: {substeps} substeps, "
                               f"{self.broadphase_pairs_count} broadphase pairs, "
                               f"{self.contacts_count} contacts, "
                               f"{elapsed*1000:.2f}ms")
        
        # （已移除）快速修复残余穿透的暴力方法；位置修正应由求解器负责。

        return substeps
    
    def step_fixed(self, num_substeps: int = 1):
        """
        执行固定步长的物理仿真
        
        Args:
            num_substeps: 子步数
        """
        if self.debug:
            self.timer.start("world_step_fixed")
            
        # 执行子步
        # warm-start 只在第一个子步使用
        for i in range(num_substeps):
            is_first_substep = (i == 0)
            self._internal_step(self.fixed_timestep, warmstart=is_first_substep)
        
        self.step_count += num_substeps
        self.total_time += num_substeps * self.fixed_timestep
        
        if self.debug:
            elapsed = self.timer.end("world_step_fixed")
            self.logger.info(f"Fixed step: {num_substeps} substeps, {elapsed*1000:.2f}ms")
    
    def _internal_step(self, dt: float, warmstart: bool = True):
        """
        内部仿真步骤（固定步长）
        实现标准的物理仿真管道
        
        Args:
            dt: 时间步长
            warmstart: 是否启用热启动（避免多子步中过度使用）
        """
        if len(self.bodies) == 0:
            return
        
        # 0. 重置所有刚体的碰撞状态标记（准备本帧的碰撞检测）
        for body in self.bodies:
            body.colliding_with.clear()
            
        # 1. 集成外力（应用重力等）
        if self.debug:
            self.timer.start("integrate_forces")
        self._integrate_forces(dt)
        if self.debug:
            self.timer.end("integrate_forces")
        
        # 2. 宽相碰撞检测
        if self.debug:
            self.timer.start("broadphase")
        broadphase_pairs = self.broadphase.compute_pairs(self.bodies)
        self.broadphase_pairs_count = len(broadphase_pairs)
        if self.debug:
            self.timer.end("broadphase")
        
        # 3. 窄相碰撞检测
        if self.debug:
            self.timer.start("narrowphase")
        contacts = self.narrowphase.detect_contacts(broadphase_pairs)
        self.contacts = contacts  # 保存contacts列表供调试使用
        self.contacts_count = len(contacts)
        
        # 标记碰撞状态和识别新碰撞
        # 使用临时集合追踪本子步中检测到的碰撞对
        current_substep_collisions = set()
        
        for contact in contacts:
            body1_id = id(contact.body1)
            body2_id = id(contact.body2)
            collision_pair = tuple(sorted([body1_id, body2_id]))
            current_substep_collisions.add(collision_pair)
            
            # 检查是否是新碰撞（本step中第一次检测到）
            is_new = (body2_id not in contact.body1.colliding_with) and \
                     (body1_id not in contact.body2.colliding_with)
            contact.is_new_collision = is_new
            
            # 更新碰撞状态
            contact.body1.colliding_with.add(body2_id)
            contact.body2.colliding_with.add(body1_id)
            contact.body1.is_colliding = True
            contact.body2.is_colliding = True
        
        # 更新没有碰撞的物体状态
        for body in self.bodies:
            if len(body.colliding_with) == 0:
                body.is_colliding = False
        
        if self.debug:
            self.timer.end("narrowphase")
        
        # 4. 约束求解（碰撞响应）
        if len(contacts) > 0:
            if self.debug:
                self.timer.start("constraint_solving")
            
            # 调用接触回调
            if self.contact_callback:
                for contact in contacts:
                    self.contact_callback(contact)
            
            # 求解约束（传递warmstart标志）
            self.solver.solve_contacts(contacts, dt, warmstart=warmstart)
            
            if self.debug:
                self.timer.end("constraint_solving")
        
        # 5. 集成速度（更新位置和旋转）
        if self.debug:
            self.timer.start("integrate_velocities")
        self._integrate_velocities(dt)
        if self.debug:
            self.timer.end("integrate_velocities")
    
    def _integrate_forces(self, dt: float):
        """
        第一阶段积分：应用外力（如重力）到速度
        """
        for body in self.bodies:
            if not body.is_static:
                # 首先清除上一步的累积力
                body.clear_forces()
                
                # 应用重力
                if body.mass > 0:
                    body.apply_force(self.gravity * body.mass)
                
                # 集成力到速度（不更新位置）
                body.integrate_forces(dt)
    
    def _integrate_velocities(self, dt: float):
        """
        第二阶段积分：使用（可能被约束修改的）速度更新位置
        """
        for body in self.bodies:
            if not body.is_static:
                # 集成速度到位置
                body.integrate_velocities(dt)
    
    def clear_forces(self):
        """清除所有刚体的累积力"""
        for body in self.bodies:
            body.clear_forces()
    
    def get_statistics(self) -> dict:
        """获取仿真统计信息"""
        return {
            'step_count': self.step_count,
            'total_time': self.total_time,
            'bodies_count': len(self.bodies),
            'active_bodies': sum(1 for body in self.bodies if not body.is_static),
            'contacts_count': self.contacts_count,
            'broadphase_pairs_count': self.broadphase_pairs_count,
            'solver_type': type(self.solver).__name__,
            'broadphase_type': type(self.broadphase).__name__,
        }
    
    def reset(self):
        """重置世界状态"""
        self.bodies.clear()
        self.step_count = 0
        self.total_time = 0.0
        self.contacts_count = 0
        self.broadphase_pairs_count = 0
        
        if self.debug:
            self.logger.info("World reset")

    def _get_body_aabb(self, body):
        """返回body的AABB (min, max) 在世界坐标（忽略旋转，基于局部顶点范围）"""
        # 如果没有网格，使用一个小盒子基于位置
        # 此方法原为 quick_fix_penetrations 提供支持，已删除该暴力修复方法。
        # 保留为将来可能的 AABB 需求的占位符（如需请重新实现）。
        raise NotImplementedError("_get_body_aabb was removed; do not use quick_fix_penetrations")

    def _check_aabb_collision(self, body1, body2) -> bool:
        """已移除：不要使用 AABB 暴力修复接口"""
        raise NotImplementedError("_check_aabb_collision was removed; do not use quick_fix_penetrations")

    def quick_fix_penetrations(self):
        """已移除：不要使用 quick_fix_penetrations。位置修正应由求解器在位置约束阶段处理。"""
        raise NotImplementedError("quick_fix_penetrations removed; use solver position correction instead")
    
    def set_contact_callback(self, callback: Callable[[ContactInfo], None]):
        """设置接触回调函数"""
        self.contact_callback = callback
    
    def set_pre_step_callback(self, callback: Callable[['World', float], None]):
        """设置步骤前回调函数"""
        self.pre_step_callback = callback
    
    def set_post_step_callback(self, callback: Callable[['World', float], None]):
        """设置步骤后回调函数"""
        self.post_step_callback = callback
    
    def raycast(self, start: np.ndarray, direction: np.ndarray, max_distance: float = 1000.0):
        """
        射线检测（基础实现）
        
        Args:
            start: 起点
            direction: 方向（应当是单位向量）
            max_distance: 最大检测距离
            
        Returns:
            碰撞信息或None
        """
        # 这里是一个简化的实现，实际应该使用更高效的算法
        direction = direction / np.linalg.norm(direction)  # 确保是单位向量
        
        closest_hit = None
        closest_distance = max_distance
        
        for body in self.bodies:
            # 简单的AABB射线检测
            aabb = body.get_aabb()
            hit_distance = self._raycast_aabb(start, direction, aabb)
            
            if hit_distance is not None and hit_distance < closest_distance:
                closest_distance = hit_distance
                hit_point = start + direction * hit_distance
                closest_hit = {
                    'body': body,
                    'point': hit_point,
                    'distance': hit_distance,
                    'normal': None  # 需要更精确的计算
                }
        
        return closest_hit
    
    def _raycast_aabb(self, start: np.ndarray, direction: np.ndarray, aabb) -> Optional[float]:
        """射线与AABB的相交检测"""
        # 简化的射线-AABB相交算法
        try:
            inv_dir = 1.0 / direction
        except ZeroDivisionError:
            return None
        
        t_min = (aabb.min_point - start) * inv_dir
        t_max = (aabb.max_point - start) * inv_dir
        
        # 确保t_min <= t_max
        t_min, t_max = np.minimum(t_min, t_max), np.maximum(t_min, t_max)
        
        t_near = np.max(t_min)
        t_far = np.min(t_max)
        
        if t_near > t_far or t_far < 0:
            return None
        
        return t_near if t_near >= 0 else t_far
    
    def __repr__(self):
        return (f"World(bodies={len(self.bodies)}, "
                f"step_count={self.step_count}, "
                f"total_time={self.total_time:.3f}s)")


def create_world(gravity: np.ndarray = None, **kwargs) -> World:
    """
    创建物理世界的便捷函数
    
    Args:
        gravity: 重力向量
        **kwargs: 其他参数传递给World构造函数
    
    Returns:
        World实例
    """
    return World(gravity=gravity, **kwargs)
