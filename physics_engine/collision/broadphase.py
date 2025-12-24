"""
宽相碰撞检测（Broadphase Collision Detection）
使用 AABB 重叠检测快速筛选可能发生碰撞的刚体对
"""

import numpy as np
from typing import List, Tuple, Set
from ..geometry.aabb import AABB


class BroadphaseResult:
    """宽相检测结果，存储可能碰撞的刚体对"""
    
    def __init__(self, body_a, body_b):
        self.body_a = body_a
        self.body_b = body_b
    
    def __repr__(self):
        return f"BroadphaseResult({id(self.body_a)}, {id(self.body_b)})"


class Broadphase:
    """
    宽相碰撞检测器
    使用简单的 O(n²) AABB 重叠检测
    """
    
    def __init__(self):
        self.pairs = []
    
    def compute_pairs(self, bodies: List) -> List[BroadphaseResult]:
        """
        计算所有可能碰撞的刚体对
        """
        self.pairs.clear()
        
        # O(n²) 暴力检测
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                body_a = bodies[i]
                body_b = bodies[j]
                
                # 跳过两个都是静态的刚体
                if body_a.is_static and body_b.is_static:
                    continue
                
                # 获取 AABB 并检测重叠
                aabb_a = body_a.get_aabb()
                aabb_b = body_b.get_aabb()
                
                if aabb_a.overlaps(aabb_b):
                    self.pairs.append(BroadphaseResult(body_a, body_b))
        
        return self.pairs
    
    def get_pair_count(self) -> int:
        """获取当前检测到的碰撞对数量"""
        return len(self.pairs)


class SpatialHashBroadphase:
    """
    基于空间哈希的宽相检测器
    适用于刚体分布相对均匀的场景
    """
    
    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.hash_table = {}
        self.pairs = []
    
    def _hash_position(self, x: float, y: float, z: float) -> int:
        """计算位置的哈希值"""
        ix = int(x / self.cell_size)
        iy = int(y / self.cell_size)
        iz = int(z / self.cell_size)
        return hash((ix, iy, iz))
    
    def _get_cell_indices(self, aabb: AABB) -> Set[int]:
        """获取 AABB 覆盖的所有网格单元"""
        cells = set()
        
        # 计算 AABB 覆盖的网格范围
        min_x = int(aabb.min_point[0] / self.cell_size)
        max_x = int(aabb.max_point[0] / self.cell_size)
        min_y = int(aabb.min_point[1] / self.cell_size)
        max_y = int(aabb.max_point[1] / self.cell_size)
        min_z = int(aabb.min_point[2] / self.cell_size)
        max_z = int(aabb.max_point[2] / self.cell_size)
        
        # 添加所有覆盖的网格单元
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    cells.add(hash((x, y, z)))
        
        return cells
    
    def compute_pairs(self, bodies: List) -> List[BroadphaseResult]:
        """
        使用空间哈希计算可能碰撞的刚体对
        """
        self.pairs.clear()
        self.hash_table.clear()
        
        # 将刚体放入哈希表
        for body in bodies:
            aabb = body.get_aabb()
            cells = self._get_cell_indices(aabb)
            
            for cell_hash in cells:
                if cell_hash not in self.hash_table:
                    self.hash_table[cell_hash] = []
                self.hash_table[cell_hash].append(body)
        
        # 检测每个网格单元内的碰撞对
        checked_pairs = set()
        
        for cell_bodies in self.hash_table.values():
            if len(cell_bodies) < 2:
                continue
                
            # 检测该网格内的所有刚体对
            for i in range(len(cell_bodies)):
                for j in range(i + 1, len(cell_bodies)):
                    body_a = cell_bodies[i]
                    body_b = cell_bodies[j]
                    
                    # 避免重复检测
                    pair_key = (id(body_a), id(body_b)) if id(body_a) < id(body_b) else (id(body_b), id(body_a))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)
                    
                    # 跳过两个都是静态的刚体
                    if body_a.is_static and body_b.is_static:
                        continue
                    
                    # 精确 AABB 重叠检测
                    aabb_a = body_a.get_aabb()
                    aabb_b = body_b.get_aabb()
                    
                    if aabb_a.overlaps(aabb_b):
                        self.pairs.append(BroadphaseResult(body_a, body_b))
        
        return self.pairs
    
    def get_pair_count(self) -> int:
        """获取当前检测到的碰撞对数量"""
        return len(self.pairs)
    
    def get_cell_count(self) -> int:
        """获取当前使用的网格单元数量"""
        return len(self.hash_table)


def create_broadphase(method: str = "simple", **kwargs):
    """
    创建宽相检测器的工厂函数
    
    Args:
        method: 检测方法 ("simple" 或 "spatial_hash")
        **kwargs: 传递给检测器的额外参数
    
    Returns:
        宽相检测器实例
    """
    if method == "simple":
        return Broadphase()
    elif method == "spatial_hash":
        return SpatialHashBroadphase(**kwargs)
    else:
        raise ValueError(f"未知的宽相检测方法: {method}")
