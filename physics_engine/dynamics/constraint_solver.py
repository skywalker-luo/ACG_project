"""
约束求解器（Constraint Solver）
实现基于冲量的碰撞响应，处理法向冲量计算
基于 Sequential Impulse 方法，分离速度约束和位置约束
"""

import numpy as np
from typing import List, Optional
from ..collision.contact import ContactInfo


class ConstraintSolver:
    """
    基于冲量的约束求解器
    分离处理速度约束（恢复系数）和位置约束（Baumgarte稳定化）
    """
    
    def __init__(self, 
                 restitution_threshold: float = 0.1,
                 position_correction: bool = True,
                 baumgarte_factor: float = 0.3,
                 max_iterations: int = 10):
        """
        初始化约束求解器
        
        Args:
            restitution_threshold: 恢复系数生效的最小相对速度阈值
            position_correction: 是否启用位置校正（Baumgarte稳定化）
            baumgarte_factor: 位置校正因子 (0.0-1.0)
            max_iterations: 最大迭代次数
        """
        self.restitution_threshold = restitution_threshold
        self.position_correction = position_correction
        self.baumgarte_factor = baumgarte_factor
        self.max_iterations = max_iterations
        
        # 求解统计
        self.iterations_used = 0
        self.contacts_processed = 0
    
    def solve_contacts(self, contacts: List[ContactInfo], dt: float):
        """
        求解所有接触约束（分离速度和位置约束）
        
        Args:
            contacts: 接触信息列表
            dt: 时间步长
        """
        if not contacts:
            return
        
        # 调试：清空冲量日志并记录调用次数
        if not hasattr(self, '_solve_call_count'):
            self._solve_call_count = 0
        self._solve_call_count += 1
        self._impulse_log = []
        self._position_impulse_log = []  # 清空位置冲量日志
        self._constraint_data_count = 0  # 记录constraint_data数量
        
        self.contacts_processed = len(contacts)
        
        # 预处理：计算每个接触的约束参数
        constraint_data = []
        for contact in contacts:
            if self._should_solve_contact(contact):
                # 只对新碰撞施加速度约束（恢复系数），对所有碰撞施加位置约束
                is_new = getattr(contact, 'is_new_collision', True)
                velocity_data = self._prepare_velocity_constraint(contact, dt) if is_new else None
                position_data = self._prepare_position_constraint(contact, dt)
                if velocity_data is not None or position_data is not None:
                    constraint_data.append((contact, velocity_data, position_data))
        
        if not constraint_data:
            return
        
        # 记录constraint_data数量用于调试
        self._constraint_data_count = len(constraint_data)
        # 启用位置修正的暖启动（将上一帧的累积冲量用于本帧位置修正的初值）
        if hasattr(self, 'previous_position_impulses'):
            for contact, _, position_data in constraint_data:
                if position_data is not None and 'contact_id' in position_data:
                    cid = position_data['contact_id']
                    if cid in self.previous_position_impulses:
                        position_data['accumulated_impulse'] = self.previous_position_impulses[cid]
        
        # 第一步：求解速度约束（恢复系数/弹性碰撞）
        self._solve_velocity_constraints(constraint_data)
        
        # 第二步：求解位置约束（Baumgarte 稳定化）
        if self.position_correction:
            self._solve_position_constraints(constraint_data)
    
    def _should_solve_contact(self, contact: ContactInfo) -> bool:
        """
        判断是否需要求解此接触约束
        
        Args:
            contact: 接触信息
            
        Returns:
            是否需要求解
        """
        # 跳过两个静态刚体
        if contact.body1.is_static and contact.body2.is_static:
            return False
        
        # 跳过穿透深度太小的接触（降低阈值以处理浅接触）
        if contact.penetration < 1e-9:
            return False
        
        return True
    
    def _prepare_velocity_constraint(self, contact: ContactInfo, dt: float) -> Optional[dict]:
        """
        预处理速度约束数据（只处理恢复系数/弹性碰撞）
        使用经典公式: j = -(1 + e) * vn * m_eff
        
        Args:
            contact: 接触信息
            dt: 时间步长
            
        Returns:
            速度约束数据字典，如果无效则返回None
        """
        # 更新接触运动学信息
        contact.update_kinematics()
        
        # 计算有效质量
        effective_mass = contact.get_effective_mass()
        if effective_mass < 1e-10:
            return None
        
        # 计算初始法线速度（只在第一次计算，不在迭代中更新）
        initial_normal_velocity = contact.get_normal_velocity()
        
        # 应用恢复系数的条件：
        # 1. 物体正在靠近（normal_velocity < 0，因为法线从body2指向body1）
        # 2. 速度大于阈值（避免微小振荡）
        # 3. 该接触点之前没有施加过速度约束（防止重复施加）
        # 注：法线从body2指向body1（推开方向），normal_velocity = (v1-v2)·n
        #     当 normal_velocity < 0 时，表示body1正在靠近body2
        
        # 检查是否已经对此接触施加过速度约束
        contact_id = self._get_contact_id(contact) if hasattr(self, '_get_contact_id') else str(id(contact))
        
        # 如果是ImpulseConstraintSolver，检查是否已处理过
        if hasattr(self, 'velocity_constraints_applied'):
            if contact_id in self.velocity_constraints_applied:
                # 已经施加过速度约束，跳过
                return None
        
        if initial_normal_velocity < -self.restitution_threshold:
            restitution = self._get_effective_restitution(contact)
            
            # 调试：记录冲量计算
            if not hasattr(self, '_impulse_log'):
                self._impulse_log = []
            self._impulse_log.append({
                'type': 'velocity',
                'vn': initial_normal_velocity,
                'restitution': restitution,
                'impulse': -(1.0 + restitution) * initial_normal_velocity * effective_mass,
                'body1_vel_before': contact.body1.velocity.copy()
            })
            
            # 冲量公式: j = -(1 + e) * vn * m_eff
            # 当vn < 0（靠近）时，j > 0（正冲量）
            # 正冲量沿法线方向（从body2指向body1）推开body1
            total_impulse_needed = -(1.0 + restitution) * initial_normal_velocity * effective_mass
            
            # 临时调试：减小冲量以防止弹飞
            # total_impulse_needed *= 0.1
            
            # 标记此接触已处理速度约束
            if hasattr(self, 'velocity_constraints_applied'):
                self.velocity_constraints_applied.add(contact_id)
            
            return {
                'effective_mass': effective_mass,
                'initial_normal_velocity': initial_normal_velocity,
                'total_impulse_needed': total_impulse_needed,
                'accumulated_impulse': 0.0,
                'contact_id': contact_id
            }
        
        return None  # 不需要速度约束
    
    def _prepare_position_constraint(self, contact: ContactInfo, dt: float) -> Optional[dict]:
        """
        预处理位置约束数据（Baumgarte稳定化）
        """
        # 对于弹性碰撞（有速度约束的碰撞），不应该再施加位置约束
        # 因为位置约束会抵消速度约束产生的弹跳效果
        # 位置约束应该只用于静态接触（无restitution）或大穿透的修复
        if not self.position_correction:
            return None
        
        # 如果穿透深度很小（<0.0001m = 0.1mm），忽略位置约束
        # 微小的穿透（如1微米）不应该产生冲量
        penetration_threshold = 0.0001  # 0.1mm
        if contact.penetration <= penetration_threshold:
            return None
        
        # 计算有效质量
        effective_mass = contact.get_effective_mass()
        if effective_mass < 1e-10:
            return None
        
        # Baumgarte 位置校正
        correction_velocity = -self.baumgarte_factor * contact.penetration / dt  # 更激进的修正
        
        # 限制修正速度的最大值，防止位置约束产生过大的冲量
        # 位置修正应该温和，不应该产生大的速度变化
        max_correction_velocity = 5.0  # m/s，限制最大修正速度
        correction_velocity = max(correction_velocity, -max_correction_velocity)
        
        # 从热启动的累积冲量开始（对于 ImpulseConstraintSolver）
        initial_impulse = 0.0
        if hasattr(self, 'current_impulses'):
            contact_id = self._get_contact_id(contact) if hasattr(self, '_get_contact_id') else str(id(contact))
            # 位置约束使用不同的累积冲量（可以是速度约束的一部分）
            initial_impulse = 0.0  # 位置约束从 0 开始
        
        return {
            'effective_mass': effective_mass,
            'target_velocity': correction_velocity,
            'accumulated_impulse': initial_impulse,
            'contact_id': contact_id if 'contact_id' in locals() else str(id(contact))
        }
    
    def _solve_velocity_constraints(self, constraint_data: List):
        """
        求解速度约束（恢复系数）
        对于restitution，只需要一次性应用，不需要迭代
        
        Args:
            constraint_data: 约束数据列表
        """
        # Restitution是一次性冲量，不需要迭代
        # 直接应用预计算的总冲量
        for contact, velocity_data, position_data in constraint_data:
            if velocity_data is None:
                continue
            
            # 记录施加冲量前的实际速度
            vel_just_before_impulse = contact.body1.velocity.copy() if contact.body1 else None
            
            # 求解速度约束（只执行一次）
            impulse_delta = self._solve_velocity_impulse(contact, velocity_data)
            
            if abs(impulse_delta) > 1e-12:
                self._apply_impulse_to_bodies(contact, impulse_delta)
                
                # 记录施加冲量后的实际速度
                vel_just_after_impulse = contact.body1.velocity.copy() if contact.body1 else None
                
                # 更新日志中的信息
                if hasattr(self, '_impulse_log') and self._impulse_log:
                    self._impulse_log[-1]['vel_just_before'] = vel_just_before_impulse
                    self._impulse_log[-1]['vel_just_after'] = vel_just_after_impulse
        
        self.iterations_used = 1  # 只需要一次应用
    
    def _solve_position_constraints(self, constraint_data: List):
        """
        求解位置约束（Baumgarte稳定化）
        多次迭代提升多物体穿透修正效果
        
        Args:
            constraint_data: 约束数据列表
        """
        num_iterations = self.max_iterations if hasattr(self, 'max_iterations') else 10
        # 用于暖启动，记录本帧的累积冲量
        if not hasattr(self, 'previous_position_impulses'):
            self.previous_position_impulses = {}
        for _ in range(num_iterations):
            for contact, velocity_data, position_data in constraint_data:
                if position_data is None:
                    continue
                contact.update_kinematics()
                impulse_delta = self._solve_position_impulse(contact, position_data)
                if abs(impulse_delta) > 1e-12:
                    self._apply_impulse_to_bodies(contact, impulse_delta, apply_friction=False)
                # 记录本帧的累积冲量用于下次暖启动
                if 'contact_id' in position_data:
                    self.previous_position_impulses[position_data['contact_id']] = position_data['accumulated_impulse']

    def _solve_velocity_impulse(self, contact: ContactInfo, data: dict) -> float:
        """
        求解速度约束的冲量（一次性应用restitution冲量）
        使用经典公式，不依赖迭代更新
        
        Args:
            contact: 接触信息
            data: 速度约束数据
            
        Returns:
            本次迭代应用的冲量大小
        """
        # 经典restitution: 使用预计算的总冲量，一次性应用
        # 不重新计算current_normal_velocity (避免迭代抵消)
        total_impulse = data['total_impulse_needed']
        accumulated = data['accumulated_impulse']
        
        # 本次应用的冲量 = 总需求 - 已累积
        lambda_apply = total_impulse - accumulated
        
        # 速度约束（restitution）不需要钳位，应该完全施加
        # 只有位置约束才需要钳位防止拉力
        
        # 更新累积冲量
        data['accumulated_impulse'] = total_impulse
        
        # 如果是 ImpulseConstraintSolver，保存累积冲量用于热启动
        if hasattr(self, 'current_impulses') and 'contact_id' in data:
            self.current_impulses[data['contact_id']] = total_impulse
        
        return lambda_apply
    
    def _solve_position_impulse(self, contact: ContactInfo, data: dict) -> float:
        """
        求解位置约束的冲量（Baumgarte稳定化）
        
        Args:
            contact: 接触信息
            data: 位置约束数据
            
        Returns:
            本次迭代应用的冲量大小
        """
        # 保存body1的速度用于日志
        body1_vel_before_pos = contact.body1.velocity.copy() if contact.body1 else None
        
        # 计算当前法线速度
        current_normal_velocity = contact.get_normal_velocity()
        
        # 计算所需的速度变化
        velocity_change = data['target_velocity'] - current_normal_velocity
        
        # 计算原始冲量变化量
        lambda_delta = data['effective_mass'] * velocity_change
        
        # Sequential Impulse: 累积冲量，必须添加钳位防止拉力
        lambda_old = data['accumulated_impulse']
        lambda_new = lambda_old + lambda_delta
        
        # 钳位：位置约束只能施加负冲量（推开），不能施加正冲量（拉近）
        # 因为Baumgarte目标速度是负的（减少穿透），所以冲量应该是负的
        lambda_new = min(0.0, lambda_new)
        
        # 本次实际应用的冲量
        lambda_apply = lambda_new - lambda_old
        
        # 更新累积冲量
        data['accumulated_impulse'] = lambda_new
        
        # 记录位置约束冲量到日志
        if body1_vel_before_pos is not None and abs(lambda_apply) > 1e-6:
            if not hasattr(self, '_position_impulse_log'):
                self._position_impulse_log = []
            self._position_impulse_log.append({
                'impulse': abs(lambda_apply),
                'penetration': contact.penetration,
                'vel_before': body1_vel_before_pos,
                'vel_change_target': data['target_velocity'],
                'effective_mass': data['effective_mass'],
                'lambda_delta': lambda_delta,
                'lambda_apply': lambda_apply
            })
        
        # 位置约束的冲量不用于热启动，因为它们是非物理的校正
        # 只有速度约束的冲量会被保存用于下一帧的热启动
        
        return lambda_apply
    
    def _get_effective_restitution(self, contact: ContactInfo) -> float:
        """
        计算有效恢复系数
        
        Args:
            contact: 接触信息
            
        Returns:
            有效恢复系数
        """
        # 使用两个刚体恢复系数的几何平均
        e1 = getattr(contact.body1, 'restitution', 0.5)
        e2 = getattr(contact.body2, 'restitution', 0.5)
        return np.sqrt(e1 * e2)
    
    def _get_effective_friction(self, contact: ContactInfo) -> float:
        """
        计算有效摩擦系数
        
        Args:
            contact: 接触信息
            
        Returns:
            有效摩擦系数
        """
        # 使用两个刚体摩擦系数的几何平均
        mu1 = getattr(contact.body1, 'friction', 0.3)
        mu2 = getattr(contact.body2, 'friction', 0.3)
        return np.sqrt(mu1 * mu2)
    
    def _compute_friction_impulse(self, contact: ContactInfo, normal_impulse: float) -> np.ndarray:
        """
        计算摩擦冲量（库仑摩擦模型）
        
        Args:
            contact: 接触信息
            normal_impulse: 已应用的法向冲量大小
            
        Returns:
            切向摩擦冲量向量（3D向量）
        """
        EPSILON = 1e-10
        
        # 第一步：计算接触点的相对速度
        # v_contact1 = v1 + omega1 × r1
        v_contact1 = contact.body1.velocity.copy()
        if not contact.body1.is_static:
            v_contact1 += np.cross(contact.body1.angular_velocity, contact.r1)
        
        # v_contact2 = v2 + omega2 × r2
        v_contact2 = contact.body2.velocity.copy()
        if not contact.body2.is_static:
            v_contact2 += np.cross(contact.body2.angular_velocity, contact.r2)
        
        # 相对速度：物体1相对于物体2
        v_rel = v_contact1 - v_contact2
        
        # 第二步：分离出切向速度分量
        n = contact.normal
        v_n = np.dot(v_rel, n) * n  # 法向速度分量
        v_t = v_rel - v_n            # 切向速度分量
        
        # 计算切向速度大小
        v_t_magnitude = np.linalg.norm(v_t)
        
        # 如果切向速度太小，不施加摩擦力
        if v_t_magnitude < EPSILON:
            return np.zeros(3)
        
        # 单位切向向量
        t_hat = v_t / v_t_magnitude
        
        # 第三步：计算切向方向的有效质量 K_t
        # term1 = 1/m1 + 1/m2
        term1 = 0.0
        if not contact.body1.is_static:
            term1 += contact.body1.inv_mass
        if not contact.body2.is_static:
            term1 += contact.body2.inv_mass
        
        # term2 = t · ((I1^-1 (r1 × t)) × r1)
        term2 = 0.0
        if not contact.body1.is_static:
            r_cross_t = np.cross(contact.r1, t_hat)
            inv_I_times_r_cross_t = contact.body1.get_inv_world_inertia() @ r_cross_t
            term2 = np.dot(t_hat, np.cross(inv_I_times_r_cross_t, contact.r1))
        
        # term3 = t · ((I2^-1 (r2 × t)) × r2)
        term3 = 0.0
        if not contact.body2.is_static:
            r_cross_t = np.cross(contact.r2, t_hat)
            inv_I_times_r_cross_t = contact.body2.get_inv_world_inertia() @ r_cross_t
            term3 = np.dot(t_hat, np.cross(inv_I_times_r_cross_t, contact.r2))
        
        # 计算逆有效质量
        inv_K_t = term1 + term2 + term3
        
        if inv_K_t < EPSILON:
            return np.zeros(3)
        
        # 有效质量（与法向冲量公式保持一致）
        K_t = 1.0 / inv_K_t
        
        # 第四步：计算理想摩擦冲量（完全消除切向速度）
        # j_t_desired = -v_t * m_eff （与法向冲量公式一致）
        j_t_desired = -v_t_magnitude * K_t
        
        # 第五步：应用库仑摩擦定律限制
        mu = self._get_effective_friction(contact)
        max_friction_impulse = mu * abs(normal_impulse)
        
        # 限制摩擦冲量大小
        if abs(j_t_desired) <= max_friction_impulse:
            # 静摩擦：可以完全消除滑动
            j_t = j_t_desired  # j_t_desired已经是负的，方向正确
        else:
            # 动摩擦：只能施加最大允许值，方向与速度相反
            # 因为v_t_magnitude > 0，而摩擦力与运动方向相反
            j_t = -max_friction_impulse
        
        # 第六步：验证摩擦冲量不会导致切向速度反向
        # 计算施加冲量后的切向速度（考虑完整的角速度效应）
        friction_impulse_vec = j_t * t_hat
        
        # 模拟施加冲量后的速度变化
        v_contact1_after = v_contact1.copy()
        v_contact2_after = v_contact2.copy()
        
        if not contact.body1.is_static:
            # 线性速度变化
            dv1 = friction_impulse_vec * contact.body1.inv_mass
            v1_after = contact.body1.velocity + dv1
            # 角速度变化
            angular_impulse1 = np.cross(contact.r1, friction_impulse_vec)
            domega1 = contact.body1.get_inv_world_inertia() @ angular_impulse1
            omega1_after = contact.body1.angular_velocity + domega1
            # 接触点速度
            v_contact1_after = v1_after + np.cross(omega1_after, contact.r1)
        
        if not contact.body2.is_static:
            # 线性速度变化
            dv2 = -friction_impulse_vec * contact.body2.inv_mass
            v2_after = contact.body2.velocity + dv2
            # 角速度变化
            angular_impulse2 = np.cross(contact.r2, -friction_impulse_vec)
            domega2 = contact.body2.get_inv_world_inertia() @ angular_impulse2
            omega2_after = contact.body2.angular_velocity + domega2
            # 接触点速度
            v_contact2_after = v2_after + np.cross(omega2_after, contact.r2)
        
        # 计算新的相对速度
        v_rel_after = v_contact2_after - v_contact1_after
        v_t_after = v_rel_after - np.dot(v_rel_after, n) * n
        
        # 检查：如果施加冲量后切向速度方向反转，则缩减冲量
        # v_t 和 v_t_after 的点积应该 >= 0（方向相同或为零）
        if np.linalg.norm(v_t_after) > EPSILON:
            dot_product = np.dot(v_t, v_t_after)
            if dot_product < 0:
                # 速度反转了，使用一个更小的冲量（比如原来的80%）
                # 或者直接使用 j_t_desired（它保证恰好消除速度）
                return j_t_desired * t_hat
        
        # 返回摩擦冲量向量
        return friction_impulse_vec
    
    def _apply_impulse_to_bodies(self, contact: ContactInfo, impulse_magnitude: float, apply_friction: bool = True):
        """
        将法向冲量和摩擦冲量一起应用到两个刚体
        
        注意：这个方法先计算法向和摩擦冲量，然后一次性施加，
        避免了顺序依赖导致的能量误差。
        
        Args:
            contact: 接触信息
            impulse_magnitude: 法向冲量大小（j_n）
            apply_friction: 是否施加摩擦冲量（默认True）
        """
        if impulse_magnitude <= 1e-10:
            return
        
        # 第一步：计算法向冲量向量（基于初始状态）
        normal_impulse_vec = impulse_magnitude * contact.normal
        
        # 第二步：基于未施加冲量前的速度计算摩擦冲量（仅在需要时）
        # 这确保了摩擦力计算不受法向冲量的影响
        if apply_friction:
            friction_impulse_vec = self._compute_friction_impulse(contact, impulse_magnitude)
        else:
            friction_impulse_vec = np.zeros(3)
        
        # 第三步：组合总冲量 = 法向冲量 + 摩擦冲量
        total_impulse = normal_impulse_vec + friction_impulse_vec
        
        # 应用到第一个刚体
        if not contact.body1.is_static:
            # 线性冲量
            contact.body1.velocity += total_impulse * contact.body1.inv_mass
            
            # 角冲量
            angular_impulse = np.cross(contact.r1, total_impulse)
            angular_vel_change = contact.body1.get_inv_world_inertia() @ angular_impulse
            contact.body1.angular_velocity += angular_vel_change
        
        # 应用到第二个刚体（反向）
        if not contact.body2.is_static:
            # 线性冲量
            contact.body2.velocity -= total_impulse * contact.body2.inv_mass
            
            # 角冲量
            angular_impulse = np.cross(contact.r2, -total_impulse)
            contact.body2.angular_velocity += contact.body2.get_inv_world_inertia() @ angular_impulse
    
    def warm_start(self, contacts: List[ContactInfo], previous_impulses: dict = None):
        """
        热启动：使用上一帧的冲量作为初始猜测
        
        Args:
            contacts: 接触信息列表
            previous_impulses: 上一帧的累积冲量字典 {contact_id: impulse}
        """
        if previous_impulses is None:
            return
        
        # 为每个接触应用上一帧的冲量（简化实现）
        for contact in contacts:
            contact_id = id(contact)  # 使用接触对象ID作为键
            if contact_id in previous_impulses:
                # 应用上一帧的冲量作为热启动
                warm_impulse = previous_impulses[contact_id] * 0.9  # 衰减因子
                if warm_impulse > 1e-6:
                    self._apply_impulse_to_bodies(contact, warm_impulse)
    
    def get_solver_info(self) -> dict:
        """
        获取求解器统计信息
        
        Returns:
            包含求解统计的字典
        """
        return {
            'iterations_used': self.iterations_used,
            'contacts_processed': self.contacts_processed,
            'max_iterations': self.max_iterations
        }
    
    def reset_statistics(self):
        """重置求解器统计信息"""
        self.iterations_used = 0
        self.contacts_processed = 0


class ImpulseConstraintSolver(ConstraintSolver):
    """
    专用的冲量约束求解器
    继承自基础ConstraintSolver，添加更多特性
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 冲量累积（用于调试和分析）
        self.total_impulse_applied = 0.0
        self.max_impulse_magnitude = 0.0
        
        # 热启动支持
        self.previous_impulses = {}  # 存储上一帧的累积冲量
        self.current_impulses = {}   # 当前帧的累积冲量
        
        # 跟踪已施加速度约束的接触，防止重复施加
        self.velocity_constraints_applied = set()
    
    def solve_contacts(self, contacts: List[ContactInfo], dt: float, warmstart: bool = True):
        """重写求解方法，添加统计信息和热启动
        
        Args:
            contacts: 接触点列表
            dt: 时间步长
            warmstart: 是否启用热启动（多子步时只在第一个子步启用）
        """
        self.total_impulse_applied = 0.0
        self.max_impulse_magnitude = 0.0
        self.current_impulses.clear()
        
        # 每次solve_contacts调用时清除速度约束标记
        # 注意：不应该在这里清除！应该在World的每一步开始时清除
        # 因为一步可能有多个子步，我们希望在整个步骤中只施加一次速度约束
        # self.velocity_constraints_applied.clear()
        
        # 热启动：只在允许时才应用（避免多子步中过度使用）
        if warmstart:
            self._apply_warm_start(contacts)
        
        super().solve_contacts(contacts, dt)
        
        # 保存当前帧的累积冲量用于下一帧热启动
        self.previous_impulses = self.current_impulses.copy()
    
    def _apply_impulse_to_bodies(self, contact: ContactInfo, impulse_magnitude: float, apply_friction: bool = True):
        """重写冲量应用，添加统计"""
        super()._apply_impulse_to_bodies(contact, impulse_magnitude, apply_friction)
        
        # 更新统计信息
        self.total_impulse_applied += abs(impulse_magnitude)
        self.max_impulse_magnitude = max(self.max_impulse_magnitude, abs(impulse_magnitude))
    
    def get_impulse_statistics(self) -> dict:
        """
        获取冲量统计信息
        
        Returns:
            冲量相关统计
        """
        base_info = self.get_solver_info()
        base_info.update({
            'total_impulse_applied': self.total_impulse_applied,
            'max_impulse_magnitude': self.max_impulse_magnitude,
            'average_impulse_per_contact': (
                self.total_impulse_applied / max(1, self.contacts_processed)
            )
        })
        return base_info
    
    def _apply_warm_start(self, contacts: List[ContactInfo]):
        """
        应用热启动：在约束数据中设置初始累积冲量
        
        Args:
            contacts: 接触信息列表
        """
        for contact in contacts:
            contact_id = self._get_contact_id(contact)
            if contact_id in self.previous_impulses:
                warm_impulse = self.previous_impulses[contact_id] * 0.8  # 衰减因子
                # 初始化当前帧的累积冲量
                self.current_impulses[contact_id] = warm_impulse
                
                # 立即应用热启动冲量
                if abs(warm_impulse) > 1e-6:
                    self._apply_impulse_to_bodies(contact, warm_impulse)
            else:
                self.current_impulses[contact_id] = 0.0
    
    def _get_contact_id(self, contact) -> str:
        """
        生成接触的稳定 ID（基于两个刚体和三角形 ID）
        
        Args:
            contact: 接触信息
            
        Returns:
            接触的稳定 ID
        """
        # 尝试使用稳定的ID，否则回退到内存ID
        if hasattr(contact.body1, 'stable_id'):
            body1_id = contact.body1.stable_id
        else:
            body1_id = id(contact.body1)
            
        if hasattr(contact.body2, 'stable_id'):
            body2_id = contact.body2.stable_id
        else:
            body2_id = id(contact.body2)
        
        tri1_id = contact.triangle1_id
        tri2_id = contact.triangle2_id
        
        # 确保 ID 的排列顺序一致
        if body1_id < body2_id:
            return f"{body1_id}_{body2_id}_{tri1_id}_{tri2_id}"
        else:
            return f"{body2_id}_{body1_id}_{tri2_id}_{tri1_id}"


def create_constraint_solver(solver_type: str = 'impulse', **kwargs) -> ConstraintSolver:
    """
    创建约束求解器的工厂函数
    
    Args:
        solver_type: 求解器类型 ('basic' 或 'impulse')
        **kwargs: 传递给求解器构造函数的参数
        
    Returns:
        约束求解器实例
    """
    if solver_type == 'impulse':
        return ImpulseConstraintSolver(**kwargs)
    elif solver_type == 'basic':
        return ConstraintSolver(**kwargs)
    else:
        raise ValueError(f"未知的求解器类型: {solver_type}")