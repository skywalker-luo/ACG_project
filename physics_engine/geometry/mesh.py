"""
网格几何模块
用于加载和处理三角网格数据
支持OBJ文件格式，提供基本的几何计算功能
"""

import numpy as np
from typing import Optional, List, Tuple, Union
import os
from pathlib import Path


class Mesh:
    """
    三角网格类，用于存储和处理网格数据
    """
    
    def __init__(self, vertices: np.ndarray = None, faces: np.ndarray = None, shape: str = None, shape_params: dict = None):
        """
        初始化网格
        
        Args:
            vertices: 顶点坐标数组，形状为 (N, 3)
            faces: 面索引数组，形状为 (M, 3)，每行包含三个顶点索引
        """
        self.vertices = vertices if vertices is not None else np.empty((0, 3))
        self.faces = faces if faces is not None else np.empty((0, 3), dtype=int)
        
        # 缓存的计算结果
        self._face_normals = None
        self._vertex_normals = None
        self._face_areas = None
        self._total_area = None
        self._aabb = None
        self._dirty = True  # 标记是否需要重新计算缓存
        # 可选的几何类型标记（例如 'sphere'）和参数（如 {'radius': ...}）
        self.shape = shape
        self.shape_params = shape_params if shape_params is not None else {}
    
    @classmethod
    def from_file(cls, filepath: str, shape: str = None, shape_params: dict = None) -> 'Mesh':
        """
        从文件加载网格
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"网格文件不存在: {filepath}")
        
        if filepath.suffix.lower() == '.obj':
            return cls._load_obj(filepath, shape=shape, shape_params=shape_params)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")
    
    @classmethod
    def _load_obj(cls, filepath: Path, shape: str = None, shape_params: dict = None) -> 'Mesh':
        """
        加载OBJ文件
        """
        vertices = []
        faces = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 跳过空行和注释
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    # 解析顶点
                    if parts[0] == 'v':
                        if len(parts) >= 4:
                            try:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                vertices.append([x, y, z])
                            except ValueError:
                                print(f"警告: 第{line_num}行顶点坐标格式错误")
                    
                    # 解析面
                    elif parts[0] == 'f':
                        if len(parts) >= 4:
                            try:
                                # 处理面索引，支持 "v", "v/vt", "v/vt/vn" 格式
                                face_vertices = []
                                for vertex_data in parts[1:]:
                                    # 取第一个数字作为顶点索引
                                    vertex_index = int(vertex_data.split('/')[0])
                                    # OBJ文件索引从1开始，转换为从0开始
                                    if vertex_index > 0:
                                        face_vertices.append(vertex_index - 1)
                                    else:
                                        # 负索引表示从末尾倒数
                                        face_vertices.append(len(vertices) + vertex_index)
                                
                                # 三角化：如果是四边形或多边形，分解为三角形
                                if len(face_vertices) >= 3:
                                    # 扇形三角化
                                    for i in range(1, len(face_vertices) - 1):
                                        faces.append([face_vertices[0], 
                                                    face_vertices[i], 
                                                    face_vertices[i + 1]])
                            except (ValueError, IndexError):
                                print(f"警告: 第{line_num}行面索引格式错误")
        
        except IOError as e:
            raise IOError(f"无法读取文件 {filepath}: {e}")
        
        if not vertices:
            raise ValueError(f"文件 {filepath} 中没有找到顶点数据")
        
        vertices_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32)
        
        print(f"成功加载网格: {len(vertices_array)} 个顶点, {len(faces_array)} 个三角形")

        # 使用调用方传入的 shape 与 shape_params（如果有）来标注网格类型，
        # 否则保持为通用 Mesh
        mesh = cls(vertices_array, faces_array, shape=shape, shape_params=shape_params if shape_params is not None else {})
        return mesh
    
    def compute_face_normals(self) -> np.ndarray:
        """
        计算面法向量
        """
        if self._face_normals is not None and not self._dirty:
            return self._face_normals
        
        if len(self.faces) == 0:
            self._face_normals = np.empty((0, 3))
            return self._face_normals
        
        # 获取三角形的三个顶点
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        # 计算两条边向量
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 叉积得到法向量
        normals = np.cross(edge1, edge2)
        
        # 归一化
        norms = np.linalg.norm(normals, axis=1)
        # 避免除零
        valid_mask = norms > 1e-10
        normals[valid_mask] = normals[valid_mask] / norms[valid_mask, np.newaxis]
        
        self._face_normals = normals
        return self._face_normals
    
    def compute_vertex_normals(self) -> np.ndarray:
        """
        计算顶点法向量（基于相邻面法向量的加权平均）
        """
        if self._vertex_normals is not None and not self._dirty:
            return self._vertex_normals
        
        vertex_normals = np.zeros((len(self.vertices), 3))
        
        if len(self.faces) == 0:
            self._vertex_normals = vertex_normals
            return self._vertex_normals
        
        face_normals = self.compute_face_normals()
        face_areas = self.compute_face_areas()
        
        # 累加每个面对其顶点法向量的贡献（按面积加权）
        for i, face in enumerate(self.faces):
            weighted_normal = face_normals[i] * face_areas[i]
            vertex_normals[face[0]] += weighted_normal
            vertex_normals[face[1]] += weighted_normal
            vertex_normals[face[2]] += weighted_normal
        
        # 归一化
        norms = np.linalg.norm(vertex_normals, axis=1)
        valid_mask = norms > 1e-10
        vertex_normals[valid_mask] = vertex_normals[valid_mask] / norms[valid_mask, np.newaxis]
        
        self._vertex_normals = vertex_normals
        return self._vertex_normals
    
    def compute_face_areas(self) -> np.ndarray:
        """
        计算每个面的面积
        """
        if self._face_areas is not None and not self._dirty:
            return self._face_areas
        
        if len(self.faces) == 0:
            self._face_areas = np.empty(0)
            return self._face_areas
        
        # 获取三角形的三个顶点
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        # 计算两条边向量
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 叉积的模长的一半就是三角形面积
        cross_products = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross_products, axis=1)
        
        self._face_areas = areas
        return self._face_areas
    
    def compute_total_area(self) -> float:
        """
        计算总表面积
        """
        if self._total_area is not None and not self._dirty:
            return self._total_area
        
        face_areas = self.compute_face_areas()
        self._total_area = np.sum(face_areas)
        return self._total_area
    
    def compute_aabb(self):
        """
        计算轴对齐包围盒
        """
        if self._aabb is not None and not self._dirty:
            return self._aabb
        
        if len(self.vertices) == 0:
            from .aabb import AABB
            self._aabb = AABB(np.zeros(3), np.zeros(3))
            return self._aabb
        
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        
        from .aabb import AABB
        self._aabb = AABB(min_coords, max_coords)
        return self._aabb
    
    def get_aabb(self, transform: np.ndarray = None):
        """
        获取包围盒，可选择应用变换矩阵
        """
        aabb = self.compute_aabb()
        
        if transform is not None:
            return aabb.transform(transform)
        else:
            return aabb
    
    def translate(self, offset: np.ndarray):
        """
        平移网格
        """
        self.vertices += np.array(offset)
        self._invalidate_cache()
    
    def scale(self, factor: Union[float, np.ndarray]):
        """
        缩放网格
        
        Args:
            factor: 缩放因子，可以是标量或3维向量
        """
        if isinstance(factor, (int, float)):
            self.vertices *= factor
        else:
            self.vertices *= np.array(factor)
        self._invalidate_cache()
    
    def rotate(self, rotation_matrix: np.ndarray):
        """
        旋转网格
        """
        self.vertices = (rotation_matrix @ self.vertices.T).T
        self._invalidate_cache()
    
    def transform(self, transform_matrix: np.ndarray):
        """
        应用4x4变换矩阵
        """
        if transform_matrix.shape != (4, 4):
            raise ValueError("变换矩阵必须是4x4")
        
        # 转换为齐次坐标
        vertices_homogeneous = np.ones((len(self.vertices), 4))
        vertices_homogeneous[:, :3] = self.vertices
        
        # 应用变换
        transformed = (transform_matrix @ vertices_homogeneous.T).T
        
        # 转回3D坐标
        self.vertices = transformed[:, :3]
        self._invalidate_cache()
    
    def _invalidate_cache(self):
        """
        标记缓存失效
        """
        self._dirty = True
        self._face_normals = None
        self._vertex_normals = None
        self._face_areas = None
        self._total_area = None
        self._aabb = None
        self._center = None
    
    def is_valid(self) -> bool:
        """
        检查网格是否有效
        """
        if len(self.vertices) == 0:
            return False
        
        if len(self.faces) > 0:
            # 检查面索引是否在有效范围内
            if np.any(self.faces < 0) or np.any(self.faces >= len(self.vertices)):
                return False
        
        # 检查是否有NaN或无穷值
        if np.any(~np.isfinite(self.vertices)):
            return False
        
        return True
    
    def get_statistics(self) -> dict:
        """
        获取网格统计信息
        """
        stats = {
            'vertex_count': len(self.vertices),
            'face_count': len(self.faces),
            'is_valid': self.is_valid(),
            'has_faces': len(self.faces) > 0,
        }
        
        if len(self.vertices) > 0:
            stats.update({
                'center': self.compute_center().tolist(),
                'total_area': self.compute_total_area() if len(self.faces) > 0 else 0.0,
            })
            
            aabb = self.compute_aabb()
            stats.update({
                'aabb_min': aabb.min_point.tolist(),
                'aabb_max': aabb.max_point.tolist(),
                'aabb_size': aabb.size().tolist(),
            })
        
        return stats
    
    def save_obj(self, filepath: str):
        """
        保存为OBJ文件
        
        Args:
            filepath: 输出文件路径
        """
        with open(filepath, 'w') as f:
            f.write("# Generated by Mesh Physics Engine\n")
            
            # 写入顶点
            for vertex in self.vertices:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # 写入面（OBJ索引从1开始）
            for face in self.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def copy(self) -> 'Mesh':
        """
        创建网格的深拷贝
        
        Returns:
            新的Mesh实例
        """
        new_mesh = Mesh(self.vertices.copy(), self.faces.copy())
        return new_mesh
    
    def __len__(self) -> int:
        """
        返回顶点数量
        """
        return len(self.vertices)
    
    def __repr__(self) -> str:
        """
        字符串表示
        """
        return f"Mesh({len(self.vertices)} vertices, {len(self.faces)} faces)"


def create_box_mesh(width: float = 1.0, 
                   height: float = 1.0, 
                   depth: float = 1.0,
                   center: np.ndarray = None) -> Mesh:
    """
    创建立方体网格
    
    Args:
        width: 宽度 (X方向)
        height: 高度 (Y方向)  
        depth: 深度 (Z方向)
        center: 中心点，默认为原点
        
    Returns:
        立方体网格
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    # 半尺寸
    hx, hy, hz = width/2, height/2, depth/2
    
    # 8个顶点
    vertices = np.array([
        [-hx, -hy, -hz],  # 0
        [ hx, -hy, -hz],  # 1
        [ hx,  hy, -hz],  # 2
        [-hx,  hy, -hz],  # 3
        [-hx, -hy,  hz],  # 4
        [ hx, -hy,  hz],  # 5
        [ hx,  hy,  hz],  # 6
        [-hx,  hy,  hz],  # 7
    ]) + center
    
    # 12个三角形面（每个立方体面2个三角形）
    faces = np.array([
        # 前面 (-Z)
        [0, 1, 2], [0, 2, 3],
        # 后面 (+Z)
        [4, 7, 6], [4, 6, 5],
        # 左面 (-X)
        [0, 3, 7], [0, 7, 4],
        # 右面 (+X)
        [1, 5, 6], [1, 6, 2],
        # 底面 (-Y)
        [0, 4, 5], [0, 5, 1],
        # 顶面 (+Y)
        [3, 2, 6], [3, 6, 7],
    ])
    
    return Mesh(vertices, faces)


def create_sphere_mesh(radius: float = 0.5,
                      subdivisions: int = 2,
                      center: np.ndarray = None) -> Mesh:
    """
    创建球体网格（基于细分二十面体）
    
    Args:
        radius: 半径
        subdivisions: 细分级别（越高越光滑）
        center: 中心点，默认为原点
        
    Returns:
        球体网格
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    # 创建二十面体的基础顶点
    phi = (1 + np.sqrt(5)) / 2  # 黄金比例
    
    vertices = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    
    # 归一化到单位球
    vertices = vertices / np.linalg.norm(vertices[0])
    
    # 二十面体的20个面
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])
    
    # 细分
    for _ in range(subdivisions):
        vertices, faces = _subdivide_mesh(vertices, faces)
        # 将新顶点投影到球面上
        norms = np.linalg.norm(vertices, axis=1)
        vertices = vertices / norms[:, np.newaxis]
    
    # 缩放到指定半径并平移到中心
    vertices = vertices * radius + center
    
    return Mesh(vertices, faces)


def _subdivide_mesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    细分网格（每个三角形分成4个）
    """
    edge_dict = {}  # 边到新顶点索引的映射
    new_vertices = vertices.tolist()
    new_faces = []
    
    def get_edge_vertex(v1: int, v2: int) -> int:
        """获取边的中点顶点索引"""
        edge = tuple(sorted([v1, v2]))
        if edge not in edge_dict:
            # 创建新的中点顶点
            mid_point = (vertices[v1] + vertices[v2]) / 2
            new_vertices.append(mid_point)
            edge_dict[edge] = len(new_vertices) - 1
        return edge_dict[edge]
    
    # 处理每个面
    for face in faces:
        v0, v1, v2 = face
        
        # 获取三条边的中点
        mid01 = get_edge_vertex(v0, v1)
        mid12 = get_edge_vertex(v1, v2)
        mid02 = get_edge_vertex(v0, v2)
        
        # 创建4个新三角形
        new_faces.extend([
            [v0, mid01, mid02],
            [v1, mid12, mid01],
            [v2, mid02, mid12],
            [mid01, mid12, mid02]
        ])
    
    return np.array(new_vertices), np.array(new_faces)
