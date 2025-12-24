"""
网格物理属性计算模块
使用四面体化方法计算网格的质量、质心、体积和惯性张量
支持封闭三角网格的精确物理属性计算
"""

import numpy as np
from typing import Tuple, Optional

class MeshPropertyCalculator:
    """
    网格物理属性计算器
    使用四面体化方法计算精确的质量分布和惯性张量
    """
    
    def __init__(self, mesh, density: float = 1.0):
        """
        初始化网格属性计算器
        """
        self.mesh = mesh
        self.density = density
        
        # 缓存计算结果
        self._volume = None
        self._mass = None
        self._center = None
        self._inertia = None
        self._dirty = True
    
    def set_density(self, density: float):
        """
        设置材料密度
        """
        if density != self.density:
            if self._mass is not None:
                self._mass *= (density / self.density)
            if self._inertia is not None:
                self._inertia *= (density / self.density)
            self.density = density
    
    def invalidate_cache(self):
        """
        标记缓存失效，强制重新计算
        """
        self._dirty = True
        self._volume = None
        self._mass = None
        self._center = None
        self._inertia = None
    
    @staticmethod
    def properties_from_mesh(vertices: np.ndarray, faces: np.ndarray, density: float = 1.0):
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)

        # helper functions (as in Mirtich)
        # f1, f2, f3 for a coordinate array a = [a0,a1,a2]
        def f1(a):
            return a[0] + a[1] + a[2]

        def f2(a):
            return (a[0]**2 + a[1]**2 + a[2]**2 +
                    a[0]*a[1] + a[1]*a[2] + a[2]*a[0])

        def f3(a):
            return (a[0]**3 + a[1]**3 + a[2]**3 +
                    a[0]**2*(a[1] + a[2]) +
                    a[1]**2*(a[2] + a[0]) +
                    a[2]**2*(a[0] + a[1]) + 
                    a[0]*a[1]*a[2])

        # g functions used for mixed terms
        def g0(a):
            return f2(a) + a[0]*(f1(a) + a[0])

        def g1(a):
            return f2(a) + a[1]*(f1(a) + a[1])

        def g2(a):
            return f2(a) + a[2]*(f1(a) + a[2])
        
        mult = [1/6, 1/24, 1/24, 1/24, 1/60, 1/60, 1/60, 1/120, 1/120, 1/120]
        intg = [0]*10 # order: 1, x, y, z, xˆ2, yˆ2, zˆ2, xy, yz, zx

        # Loop over faces
        for fi in range(faces.shape[0]):
            # get vertices of triangle
            i0, i1, i2 = faces[fi]
            p0 = vertices[i0]
            p1 = vertices[i1]
            p2 = vertices[i2]

            x0, y0, z0 = p0
            x1, y1, z1 = p1
            x2, y2, z2 = p2

            # get edges and cross product of edges
            a1 = x1 - x0
            b1 = y1 - y0
            c1 = z1 - z0
            a2 = x2 - x0
            b2 = y2 - y0
            c2 = z2 - z0
            d0 = b1 * c2 - c1 * b2
            d1 = c1 * a2 - a1 * c2
            d2 = a1 * b2 - b1 * a2

            # compute helper variables
            x = np.array([x0, x1, x2])
            y = np.array([y0, y1, y2])
            z = np.array([z0, z1, z2])

            f1x = f1(x)
            f2x = f2(x)
            f3x = f3(x)
            f1y = f1(y)
            f2y = f2(y)
            f3y = f3(y)
            f1z = f1(z)
            f2z = f2(z)
            f3z = f3(z)

            g0x = g0(x)
            g1x = g1(x)
            g2x = g2(x)
            g0y = g0(y)
            g1y = g1(y)
            g2y = g2(y)
            g0z = g0(z)
            g1z = g1(z)
            g2z = g2(z)
            
            # update integrals
            intg[0] += d0 * f1x

            intg[1] += d0 * f2x
            intg[2] += d1 * f2y
            intg[3] += d2 * f2z

            intg[4] += d0 * f3x
            intg[5] += d1 * f3y
            intg[6] += d2 * f3z

            intg[7] += d0 * (y0 * g0x + y1 * g1x + y2 * g2x)
            intg[8] += d1 * (z0 * g0y + z1 * g1y + z2 * g2y)
            intg[9] += d2 * (x0 * g0z + x1 * g1z + x2 * g2z)

        for i in range(10):
            intg[i] *= mult[i]

        volume = intg[0]

        mass = volume * density if density > 0 else 0.0

        center = np.array([intg[1], intg[2], intg[3]]) / intg[0] if intg[0] != 0 else np.zeros(3)

        # 惯性张量计算（Mirtich公式）
        # intg[4], intg[5], intg[6]是体积积分，需要乘以density
        # 平行轴定理修正也需要使用mass
        Ixx = density * (intg[5] + intg[6]) - mass * (center[1]**2 + center[2]**2)
        Iyy = density * (intg[4] + intg[6]) - mass * (center[0]**2 + center[2]**2)
        Izz = density * (intg[4] + intg[5]) - mass * (center[0]**2 + center[1]**2)
        Ixy = -density * intg[7] + mass * center[0] * center[1]
        Iyz = -density * intg[8] + mass * center[1] * center[2]
        Izx = -density * intg[9] + mass * center[2] * center[0]

        inertia = np.array([[Ixx, Ixy, Izx],
                            [Ixy, Iyy, Iyz],
                            [Izx, Iyz, Izz]])

        return volume, mass, center, inertia
    
    def compute_mesh_properties(self):
        """
        计算网格的物理属性：体积、质量、质心和惯性张量
        """
        if not self._dirty:
            return self._volume, self._mass, self._center, self._inertia
        
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        volume, mass, center, inertia = MeshPropertyCalculator.properties_from_mesh(
            vertices, faces, density=self.density
        )
        
        self._volume = volume
        self._mass = mass
        self._center = center
        self._inertia = inertia
        self._dirty = False
        
        return volume, mass, center, inertia
    
    def get_volume(self) -> float:
        """
        获取网格体积
        """
        if self._volume is None:
            self.compute_mesh_properties()
        return self._volume
    
    def get_mass(self) -> float:
        """
        获取网格质量
        """
        if self._mass is None:
            self.compute_mesh_properties()
        return self._mass
    
    def get_center(self) -> np.ndarray:
        """
        获取网格质心
        """
        if self._center is None:
            self.compute_mesh_properties()
        return self._center.copy()
    
    def get_inertia(self) -> np.ndarray:
        """
        获取相对于质心的惯性张量
        """
        if self._inertia is None:
            self.compute_mesh_properties()
        return self._inertia.copy()
    
    def get_all_properties(self) -> dict:
        """
        获取所有物理属性
        """
        volume, mass, center, inertia = self.compute_mesh_properties()
        
        return {
            'volume': volume,
            'mass': mass,
            'density': self.density,
            'center': center.copy(),
            'inertia': inertia.copy(),
            'is_valid': volume > 1e-12
        }

def validate_mesh_for_physics(mesh) -> Tuple[bool, str]:
    """
    验证网格是否适用于物理计算
    """
    if not hasattr(mesh, 'vertices'):
        return False, "网格缺少顶点数据"
    
    if not hasattr(mesh, 'faces'):
        return False, "网格缺少面数据"
    
    if len(mesh.vertices) == 0:
        return False, "网格没有顶点"
    
    if len(mesh.faces) == 0:
        return False, "网格没有面"
    
    # 检查面索引是否有效
    if np.any(mesh.faces < 0) or np.any(mesh.faces >= len(mesh.vertices)):
        return False, "网格面索引超出范围"
    
    # 检查是否有NaN或无穷值
    if np.any(~np.isfinite(mesh.vertices)):
        return False, "网格顶点包含无效值"
    
    # 计算体积检查
    try:
        calculator = MeshPropertyCalculator(mesh)
        volume = calculator.get_volume()
        if volume <= 1e-12:
            return False, "网格体积为零或负数，可能不是封闭网格"
    except Exception as e:
        return False, f"网格物理属性计算失败: {str(e)}"
    
    return True, "网格有效"
