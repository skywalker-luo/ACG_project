"""
碰撞接触点数据结构
存储两个刚体之间的接触信息，用于约束求解
"""

import numpy as np
from typing import Optional, List


class ContactInfo:
    """
    接触点信息，包含碰撞的详细数据
    """
    
    def __init__(self,
                 body1,
                 body2, 
                 contact_point: np.ndarray, # 世界坐标
                 normal: np.ndarray, # 从 body1 指向 body2
                 penetration: float, # 正值表示穿透
                 triangle1_id: int = -1, # -1表示无效
                 triangle2_id: int = -1): # -1表示无效
        """
        初始化接触点信息
        """
        self.body1 = body1
        self.body2 = body2
        self.contact_point = np.array(contact_point, dtype=np.float64)
        self.normal = np.array(normal, dtype=np.float64)
        self.penetration = float(penetration)
        self.triangle1_id = triangle1_id
        self.triangle2_id = triangle2_id
        
        # 确保法线是单位向量
        normal_length = np.linalg.norm(self.normal)
        if normal_length > 1e-10:
            self.normal /= normal_length
        else:
            self.normal = np.array([0.0, 1.0, 0.0])  # 默认向上法线
        
        # 计算相对位置向量（从质心到接触点）
        # 质心在世界坐标系中的位置 = position + rotation_matrix @ center
        center1_world = self._get_world_center_of_mass(self.body1)
        center2_world = self._get_world_center_of_mass(self.body2)
        
        self.r1 = self.contact_point - center1_world  # body1 质心到接触点
        self.r2 = self.contact_point - center2_world  # body2 质心到接触点
        
        # 接触速度相关
        self._relative_velocity = None
        self._normal_velocity = None
        
        # 碰撞状态标志：是否是新碰撞（用于决定是否施加速度约束）
        # 默认为True，假设是新碰撞，World会在检测时更新此标志
        self.is_new_collision = True
    
    def get_relative_velocity(self) -> np.ndarray:
        """
        计算接触点的相对速度
        v_rel = (v1 + ω1 × r1) - (v2 + ω2 × r2)
        """
        if self._relative_velocity is None:
            v1_contact = self.body1.velocity + np.cross(self.body1.angular_velocity, self.r1)
            v2_contact = self.body2.velocity + np.cross(self.body2.angular_velocity, self.r2)
            self._relative_velocity = v1_contact - v2_contact
        return self._relative_velocity
    
    def get_normal_velocity(self) -> float:
        """
        获取法线方向的相对速度
        v_n = v_rel · n
        """
        if self._normal_velocity is None:
            self._normal_velocity = np.dot(self.get_relative_velocity(), self.normal)
        return self._normal_velocity
    
    def get_tangential_velocity(self) -> np.ndarray:
        """
        获取切线方向的相对速度
        v_t = v_rel - (v_rel · n) * n
        """
        v_rel = self.get_relative_velocity()
        v_n = self.get_normal_velocity()
        return v_rel - v_n * self.normal
    
    def update_kinematics(self):
        """
        更新运动学相关缓存（当刚体状态改变时调用）
        """
        center1_world = self._get_world_center_of_mass(self.body1)
        center2_world = self._get_world_center_of_mass(self.body2)
        self.r1 = self.contact_point - center1_world
        self.r2 = self.contact_point - center2_world
        self._relative_velocity = None
        self._normal_velocity = None
    
    def is_separating(self) -> bool:
        """
        判断两个刚体是否正在分离
        法线从body1指向body2，所以(v1-v2)·n > 0表示靠近，< 0表示分离
        """
        return self.get_normal_velocity() < 0.0
    
    def is_resting(self, threshold: float = 1e-3) -> bool:
        """
        判断是否为静止接触
        
        Args:
            threshold: 速度阈值
        """
        return abs(self.get_normal_velocity()) < threshold
    
    def get_effective_mass(self) -> float:
        """
        计算接触点在法线方向的有效质量
        用于冲量计算：1/m_eff = 1/m1 + 1/m2 + (r1×n)^T * I1^-1 * (r1×n) + (r2×n)^T * I2^-1 * (r2×n)
        """
        # 线性部分
        linear_term = self.body1.inv_mass + self.body2.inv_mass
        
        # 角动量部分 - 使用正确的公式：(r×n)^T * I^-1 * (r×n)
        if not self.body1.is_static:
            r1_cross_n = np.cross(self.r1, self.normal)
            temp1 = self.body1.get_inv_world_inertia() @ r1_cross_n
            angular_term1 = np.dot(r1_cross_n, temp1)  # 这保证是正值
        else:
            angular_term1 = 0.0
            
        if not self.body2.is_static:
            r2_cross_n = np.cross(self.r2, self.normal)  
            temp2 = self.body2.get_inv_world_inertia() @ r2_cross_n
            angular_term2 = np.dot(r2_cross_n, temp2)  # 这保证是正值
        else:
            angular_term2 = 0.0
        
        inv_mass_eff = linear_term + angular_term1 + angular_term2
        
        # 避免除零和负值
        if inv_mass_eff > 1e-10:
            return 1.0 / inv_mass_eff
        else:
            return 0.0
    
    def _get_world_center_of_mass(self, body) -> np.ndarray:
        """
        获取刚体质心在世界坐标系中的位置
        质心世界位置 = position + rotation_matrix @ center
        """
        if hasattr(body, 'center') and body.center is not None:
            rotation_matrix = self._quaternion_to_matrix(body.orientation)
            return body.position + rotation_matrix @ body.center
        else:
            # 如果没有 center 属性，假设质心就在局部原点
            return body.position
    
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
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"ContactInfo(\n"
                f"  body1={id(self.body1)}, body2={id(self.body2)}\n"
                f"  point={self.contact_point}\n"  
                f"  normal={self.normal}\n"
                f"  penetration={self.penetration:.4f}\n"
                f"  triangles=({self.triangle1_id}, {self.triangle2_id})\n"
                f")")


class ContactManifold:
    """
    接触流形，管理同一对刚体之间的多个接触点
    """
    
    def __init__(self, body1, body2, max_contacts: int = 4):
        self.body1 = body1
        self.body2 = body2
        self.contacts: List[ContactInfo] = []
        self.max_contacts = max_contacts
    
    def add_contact(self, contact: ContactInfo):
        """
        添加接触点，如果超过最大数量则移除最旧的
        """
        self.contacts.append(contact)
        
        # 如果超过最大接触点数，移除最旧的
        if len(self.contacts) > self.max_contacts:
            self.contacts.pop(0)
    
    def update_contacts(self):
        """
        更新所有接触点的运动学信息
        """
        for contact in self.contacts:
            contact.update_kinematics()
    
    def remove_separated_contacts(self, separation_threshold: float = 0.1):
        """
        移除已经分离的接触点
        
        Args:
            separation_threshold: 分离阈值
        """
        self.contacts = [c for c in self.contacts 
                        if c.penetration >= -separation_threshold]
    
    def get_deepest_contact(self) -> Optional[ContactInfo]:
        """
        获取穿透最深的接触点
        """
        if not self.contacts:
            return None
        return max(self.contacts, key=lambda c: c.penetration)
    
    def get_contact_count(self) -> int:
        """获取接触点数量"""
        return len(self.contacts)
    
    def clear(self):
        """清除所有接触点"""
        self.contacts.clear()
    
    def __repr__(self) -> str:
        return (f"ContactManifold(body1={id(self.body1)}, body2={id(self.body2)}, "
                f"contacts={len(self.contacts)})")


def create_contact(body1, body2, 
                  contact_point: np.ndarray,
                  normal: np.ndarray, 
                  penetration: float,
                  triangle1_id: int = -1,
                  triangle2_id: int = -1) -> ContactInfo:
    """
    创建接触点的便捷函数
    
    Args:
        body1, body2: 碰撞的两个刚体
        contact_point: 接触点位置
        normal: 接触法线
        penetration: 穿透深度
        triangle1_id, triangle2_id: 三角形ID
    
    Returns:
        ContactInfo 实例
    """
    return ContactInfo(body1, body2, contact_point, normal, 
                      penetration, triangle1_id, triangle2_id)
