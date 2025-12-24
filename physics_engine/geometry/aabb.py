"""
轴对齐包围盒（AABB）模块
用于碰撞检测的广义相位，提供快速的重叠测试
支持包围盒的基本操作和变换
"""

import numpy as np
from typing import Optional, List, Tuple, Union


class AABB:
    """
    轴对齐包围盒类
    用于快速碰撞检测和空间查询
    """
    
    def __init__(self, min_point: np.ndarray = None, max_point: np.ndarray = None):
        """
        初始化AABB
        
        Args:
            min_point: 最小点坐标 (3,)
            max_point: 最大点坐标 (3,)
        """
        if min_point is None:
            min_point = np.array([0.0, 0.0, 0.0])
        if max_point is None:
            max_point = np.array([0.0, 0.0, 0.0])
            
        self.min_point = np.array(min_point, dtype=np.float32)
        self.max_point = np.array(max_point, dtype=np.float32)
        
        # 确保min <= max
        self._fix_bounds()
    
    def _fix_bounds(self):
        """
        确保最小点的每个坐标都小于等于最大点
        """
        for i in range(3):
            if self.min_point[i] > self.max_point[i]:
                self.min_point[i], self.max_point[i] = self.max_point[i], self.min_point[i]
    
    @classmethod
    def from_points(cls, points: np.ndarray) -> 'AABB':
        """
        从点集创建AABB
        """
        if len(points) == 0:
            return cls()
        
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        return cls(min_point, max_point)
    
    @classmethod
    def from_center_size(cls, center: np.ndarray, size: np.ndarray) -> 'AABB':
        """
        从中心点和尺寸创建AABB
        """
        center = np.array(center)
        size = np.array(size)
        half_size = size * 0.5
        
        return cls(center - half_size, center + half_size)
    
    def center(self) -> np.ndarray:
        """
        获取AABB中心点
        """
        return (self.min_point + self.max_point) * 0.5
    
    def size(self) -> np.ndarray:
        """
        获取AABB尺寸
        """
        return self.max_point - self.min_point
    
    def half_extent(self) -> np.ndarray:
        """
        获取AABB半尺寸
        """
        return self.size() * 0.5
    
    def volume(self) -> float:
        """
        计算AABB体积
        """
        size = self.size()
        return size[0] * size[1] * size[2]
    
    def surface_area(self) -> float:
        """
        计算AABB表面积
        """
        size = self.size()
        return 2.0 * (size[0] * size[1] + size[1] * size[2] + size[0] * size[2])
    
    def diagonal_length(self) -> float:
        """
        计算AABB对角线长度
        """
        return np.linalg.norm(self.size())
    
    def contains_point(self, point: np.ndarray) -> bool:
        """
        检查点是否在AABB内
        """
        point = np.array(point)
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
    
    def contains_aabb(self, other: 'AABB') -> bool:
        """
        检查是否完全包含另一个AABB
        """
        return (np.all(other.min_point >= self.min_point) and 
                np.all(other.max_point <= self.max_point))
    
    def overlaps(self, other: 'AABB') -> bool:
        """
        检查与另一个AABB是否重叠
        """
        # 分离轴定理：如果在任何轴上分离，则不重叠
        return not (
            (self.max_point[0] < other.min_point[0] or self.min_point[0] > other.max_point[0]) or
            (self.max_point[1] < other.min_point[1] or self.min_point[1] > other.max_point[1]) or
            (self.max_point[2] < other.min_point[2] or self.min_point[2] > other.max_point[2])
        )

    def union(self, other: 'AABB') -> 'AABB':
        """
        计算与另一个AABB的并集
        """
        min_point = np.minimum(self.min_point, other.min_point)
        max_point = np.maximum(self.max_point, other.max_point)
        
        return AABB(min_point, max_point)
    
    def expand(self, delta: Union[float, np.ndarray]) -> 'AABB':
        """
        扩展AABB
        
        Args:
            delta: 扩展量，可以是标量或向量
        """
        if isinstance(delta, (int, float)):
            delta_vec = np.array([delta, delta, delta])
        else:
            delta_vec = np.array(delta)
        
        return AABB(self.min_point - delta_vec, self.max_point + delta_vec)
    
    def translate(self, offset: np.ndarray) -> 'AABB':
        """
        平移AABB
        """
        offset = np.array(offset)
        return AABB(self.min_point + offset, self.max_point + offset)
    
    def scale(self, factor: Union[float, np.ndarray], center: np.ndarray = None) -> 'AABB':
        """
        缩放AABB
        """
        if center is None:
            center = self.center()
        else:
            center = np.array(center)
        
        if isinstance(factor, (int, float)):
            factor_vec = np.array([factor, factor, factor])
        else:
            factor_vec = np.array(factor)
        
        # 相对于中心缩放
        min_rel = (self.min_point - center) * factor_vec
        max_rel = (self.max_point - center) * factor_vec
        
        new_min = center + min_rel
        new_max = center + max_rel
        
        return AABB(new_min, new_max)
    
    def transform(self, transform_matrix: np.ndarray) -> 'AABB':
        """
        应用4x4变换矩阵
        对于AABB，需要变换所有8个角点然后重新计算包围盒
        """
        if transform_matrix.shape != (4, 4):
            raise ValueError("变换矩阵必须是4x4")
        
        # 获取AABB的8个角点
        corners = self.get_corners()
        
        # 转换为齐次坐标
        corners_homogeneous = np.ones((8, 4))
        corners_homogeneous[:, :3] = corners
        
        # 应用变换
        transformed_corners = (transform_matrix @ corners_homogeneous.T).T
        
        # 转回3D坐标
        transformed_points = transformed_corners[:, :3]
        
        # 计算新的包围盒
        return AABB.from_points(transformed_points)
    
    def get_corners(self) -> np.ndarray:
        """
        获取AABB的8个角点
        """
        min_x, min_y, min_z = self.min_point
        max_x, max_y, max_z = self.max_point
        
        corners = np.array([
            [min_x, min_y, min_z],  # 0: min corner
            [max_x, min_y, min_z],  # 1
            [max_x, max_y, min_z],  # 2
            [min_x, max_y, min_z],  # 3
            [min_x, min_y, max_z],  # 4
            [max_x, min_y, max_z],  # 5
            [max_x, max_y, max_z],  # 6: max corner
            [min_x, max_y, max_z],  # 7
        ])
        
        return corners
    
    def squared_distance_to_point(self, point: np.ndarray) -> float:
        """
        计算AABB到点的最短距离的平方（避免开方运算）
        
        Args:
            point: 查询点 (3,)
            
        Returns:
            最短距离的平方
        """
        point = np.array(point)
        
        # 计算每个轴上的距离
        dx = max(0, max(self.min_point[0] - point[0], point[0] - self.max_point[0]))
        dy = max(0, max(self.min_point[1] - point[1], point[1] - self.max_point[1]))
        dz = max(0, max(self.min_point[2] - point[2], point[2] - self.max_point[2]))
        
        return dx*dx + dy*dy + dz*dz
    
    def is_valid(self) -> bool:
        """
        检查AABB是否有效
        
        Returns:
            是否有效
        """
        return (np.all(np.isfinite(self.min_point)) and 
                np.all(np.isfinite(self.max_point)) and
                np.all(self.min_point <= self.max_point))
    
    def is_empty(self) -> bool:
        """
        检查AABB是否为空（体积为0）
        """
        return np.any(self.min_point >= self.max_point)
    
    def longest_axis(self) -> int:
        """
        获取最长轴的索引
        
        Returns:
            轴索引 (0=X, 1=Y, 2=Z)
        """
        size = self.size()
        return np.argmax(size)
    
    def split(self, axis: int, position: float) -> Tuple['AABB', 'AABB']:
        """
        沿指定轴在指定位置分割AABB
        
        Args:
            axis: 分割轴 (0=X, 1=Y, 2=Z)
            position: 分割位置
            
        Returns:
            两个分割后的AABB
        """
        # 确保分割位置在有效范围内
        position = np.clip(position, self.min_point[axis], self.max_point[axis])
        
        # 创建两个新的AABB
        left_max = self.max_point.copy()
        left_max[axis] = position
        left = AABB(self.min_point, left_max)
        
        right_min = self.min_point.copy()
        right_min[axis] = position
        right = AABB(right_min, self.max_point)
        
        return left, right
    
    def copy(self) -> 'AABB':
        """
        创建AABB的深拷贝
        
        Returns:
            新的AABB实例
        """
        return AABB(self.min_point.copy(), self.max_point.copy())
    
    def __eq__(self, other: 'AABB') -> bool:
        """
        相等性比较
        """
        if not isinstance(other, AABB):
            return False
        return (np.allclose(self.min_point, other.min_point) and 
                np.allclose(self.max_point, other.max_point))
    
    def __repr__(self) -> str:
        """
        字符串表示
        """
        return f"AABB(min={self.min_point}, max={self.max_point}, size={self.size()})"


class AABBTree:
    """
    AABB树节点，用于构建空间分割数据结构
    """
    
    def __init__(self, aabb: AABB, data=None):
        """
        初始化AABB树节点
        
        Args:
            aabb: 节点的包围盒
            data: 节点存储的数据
        """
        self.aabb = aabb
        self.data = data
        self.left = None
        self.right = None
        self.is_leaf = True
    
    def insert(self, new_aabb: AABB, new_data=None) -> 'AABBTree':
        """
        插入新的AABB到树中
        
        Args:
            new_aabb: 新的包围盒
            new_data: 新的数据
            
        Returns:
            更新后的树根
        """
        if self.is_leaf:
            # 叶子节点，创建新的内部节点
            old_node = AABBTree(self.aabb, self.data)
            new_node = AABBTree(new_aabb, new_data)
            
            self.left = old_node
            self.right = new_node
            self.aabb = self.aabb.union(new_aabb)
            self.data = None
            self.is_leaf = False
            
            return self
        else:
            # 内部节点，选择最佳子树插入
            left_cost = self.left.aabb.union(new_aabb).volume() - self.left.aabb.volume()
            right_cost = self.right.aabb.union(new_aabb).volume() - self.right.aabb.volume()
            
            if left_cost < right_cost:
                self.left = self.left.insert(new_aabb, new_data)
            else:
                self.right = self.right.insert(new_aabb, new_data)
            
            # 更新当前节点的包围盒
            self.aabb = self.left.aabb.union(self.right.aabb)
            
            return self
    
    def query_overlaps(self, query_aabb: AABB) -> List:
        """
        查询与给定AABB重叠的所有数据
        
        Args:
            query_aabb: 查询包围盒
            
        Returns:
            重叠的数据列表
        """
        results = []
        
        if not self.aabb.overlaps(query_aabb):
            return results
        
        if self.is_leaf:
            if self.data is not None:
                results.append(self.data)
        else:
            if self.left:
                results.extend(self.left.query_overlaps(query_aabb))
            if self.right:
                results.extend(self.right.query_overlaps(query_aabb))
        
        return results


def merge_aabbs(aabbs: List[AABB]) -> AABB:
    """
    合并多个AABB为一个包围所有AABB的大包围盒
    
    Args:
        aabbs: AABB列表
        
    Returns:
        合并后的AABB
    """
    if not aabbs:
        return AABB()
    
    result = aabbs[0].copy()
    for aabb in aabbs[1:]:
        result = result.union(aabb)
    
    return result


def compute_aabb_from_points(points: np.ndarray) -> AABB:
    """
    从点集计算AABB的便捷函数
    
    Args:
        points: 点坐标数组 (N, 3)
        
    Returns:
        包围所有点的AABB
    """
    return AABB.from_points(points)
