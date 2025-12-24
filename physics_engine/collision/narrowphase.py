"""
窄相碰撞检测（Narrowphase Collision Detection）
实现三角形-三角形精确相交检测
基于 Möller-Trumbore 和 SAT (Separating Axis Theorem) 算法
"""

import numpy as np
from typing import List, Optional, Tuple
from .contact import ContactInfo, create_contact
from .broadphase import BroadphaseResult


class TriangleIntersection:
    """三角形相交结果"""
    
    def __init__(self, is_intersecting: bool = False,
                 contact_point: np.ndarray = None,
                 normal: np.ndarray = None,
                 penetration: float = 0.0):
        self.is_intersecting = is_intersecting
        self.contact_point = contact_point if contact_point is not None else np.zeros(3)
        self.normal = normal if normal is not None else np.zeros(3)
        self.penetration = penetration


class Narrowphase:
    """
    窄相碰撞检测器
    对宽相筛选出的刚体对进行精确的三角形级别碰撞检测
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.contacts = []
    
    def detect_contacts(self, broadphase_pairs: List[BroadphaseResult]) -> List[ContactInfo]:
        """
        对宽相结果进行精确碰撞检测
        """
        self.contacts.clear()
        
        for pair in broadphase_pairs:
            body_contacts = self._detect_mesh_collision(pair.body_a, pair.body_b)
            self.contacts.extend(body_contacts)
        
        return self.contacts
    
    def _detect_mesh_collision(self, body1, body2) -> List[ContactInfo]:
        """
        检测两个网格刚体之间的碰撞
        添加了AABB剪枝优化
        """
        # 确保body1是动态物体，body2是静态物体（如果有一个是静态的话）
        # 这样冲量计算时，动态物体会受到正确的作用力
        if body1.is_static and not body2.is_static:
            body1, body2 = body2, body1
        
        contacts = []
        
        # 检查是否都有网格
        if not (hasattr(body1, 'mesh') and hasattr(body2, 'mesh') and
                body1.mesh is not None and body2.mesh is not None):
            return contacts
        
        # 获取变换后的顶点
        vertices1 = self._transform_vertices(body1)
        vertices2 = self._transform_vertices(body2)
        
        faces1 = body1.mesh.faces
        faces2 = body2.mesh.faces
        
        # 收集所有可能的接触点
        all_contacts = []
        
        # 早期退出：如果AABB不重叠，直接返回
        # 计算整体AABB
        aabb1_min = vertices1.min(axis=0)
        aabb1_max = vertices1.max(axis=0)
        aabb2_min = vertices2.min(axis=0)
        aabb2_max = vertices2.max(axis=0)
        
        if not self._aabb_overlap(aabb1_min, aabb1_max, aabb2_min, aabb2_max):
            return contacts
        
        # 三角形-三角形检测（带AABB早期剪枝）
        for i, face1 in enumerate(faces1):
            tri1 = vertices1[face1]
            
            # 计算tri1的AABB
            tri1_min = tri1.min(axis=0)
            tri1_max = tri1.max(axis=0)
            
            for j, face2 in enumerate(faces2):
                tri2 = vertices2[face2]
                
                # 计算tri2的AABB
                tri2_min = tri2.min(axis=0)
                tri2_max = tri2.max(axis=0)
                
                # AABB重叠测试（快速剪枝）
                if not self._aabb_overlap(tri1_min, tri1_max, tri2_min, tri2_max):
                    continue
                
                # 检测两个三角形是否相交
                intersection = self._triangle_triangle_intersection(tri1, tri2, body1, body2)
                
                if intersection.is_intersecting:
                    contact = create_contact(
                        body1, body2,
                        intersection.contact_point,
                        intersection.normal,
                        intersection.penetration,
                        triangle1_id=i,
                        triangle2_id=j
                    )
                    all_contacts.append(contact)
        
        # 关键：对于每对刚体，只返回穿透最深的一个接触点
        # 这样确保每次碰撞只施加一次冲量，不会累积
        if all_contacts:
            deepest_contact = max(all_contacts, key=lambda c: c.penetration)
            contacts.append(deepest_contact)
        
        return contacts
    
    def _aabb_overlap(self, min1, max1, min2, max2) -> bool:
        """
        快速AABB重叠测试
        
        Args:
            min1, max1: 第一个AABB的最小/最大点
            min2, max2: 第二个AABB的最小/最大点
            
        Returns:
            是否重叠
        """
        return (min1[0] <= max2[0] and max1[0] >= min2[0] and
                min1[1] <= max2[1] and max1[1] >= min2[1] and
                min1[2] <= max2[2] and max1[2] >= min2[2])
    
    def _transform_vertices(self, body) -> np.ndarray:
        """
        将网格顶点变换到世界坐标系
        
        Returns:
            变换后的顶点数组
        """
        if not hasattr(body, 'mesh') or body.mesh is None:
            return np.array([])
        
        vertices = body.mesh.vertices.copy()
        transform_matrix = body.get_transform_matrix()
        
        # 将顶点转换为齐次坐标
        ones = np.ones((vertices.shape[0], 1))
        vertices_homo = np.hstack([vertices, ones])
        
        # 应用变换
        transformed = (transform_matrix @ vertices_homo.T).T
        
        return transformed[:, :3]
    
    def _triangle_triangle_intersection(self, tri1: np.ndarray, tri2: np.ndarray, 
                                        body1, body2) -> TriangleIntersection:
        """
        检测两个三角形是否相交
        使用 Möller 1997 算法
        
        Args:
            tri1: 第一个三角形的顶点 (3x3)
            tri2: 第二个三角形的顶点 (3x3)
            body1: 第一个刚体
            body2: 第二个刚体
            
        Returns:
            TriangleIntersection 结果
        """
        V0, V1, V2 = tri1[0], tri1[1], tri1[2]
        U0, U1, U2 = tri2[0], tri2[1], tri2[2]
        
        # 使用 Möller 算法检测相交
        intersect, isect_points = Narrowphase.tri_tri_intersect_moller(
            V0, V1, V2, U0, U1, U2, self.tolerance
        )
        
        if not intersect:
            return TriangleIntersection()
        
        # 计算接触信息
        if len(isect_points) >= 1:
            contact_point = isect_points[0]
            if len(isect_points) == 2:
                # 如果有两个交点，取中点
                contact_point = (isect_points[0] + isect_points[1]) * 0.5
        else:
            # 如果是共面相交但没有具体交点，使用三角形中心
            contact_point = (np.mean(tri1, axis=0) + np.mean(tri2, axis=0)) * 0.5
        
        # 计算接触法线（应该从tri1指向tri2，即分离方向）
        normal1 = self._compute_triangle_normal(tri1)
        normal2 = self._compute_triangle_normal(tri2)

        if np.linalg.norm(normal1) > self.tolerance:
            normal = normal1 / np.linalg.norm(normal1)
        else:
            normal = np.array([1.0, 0.0, 0.0])
        
        # 确保法线指向正确方向：从body1中心指向body2中心
        # 这样冲量会将两个物体推开
        body1_center = body1.position
        body2_center = body2.position
        separation_direction = body2_center - body1_center
        
        # 如果法线和分离方向相同，翻转法线
        # (因为冲量应用约定需要法线从body2指向body1)
        if np.dot(normal, separation_direction) > 0:
            normal = -normal
        
        # 估算穿透深度
        penetration = self._estimate_penetration_depth(tri1, tri2, normal)
        
        # 如果穿透深度太小或为负（物体已分离），不报告接触
        # tolerance设为一个极小值（如1e-6m = 1微米）来避免数值误差
        # 但不要强制将微小/负穿透改为正值，否则会错误地报告已分离的物体仍在接触
        if penetration >= self.tolerance:
            pass
        elif penetration < self.tolerance and penetration > 0:
            penetration = self.tolerance
        else:
            return TriangleIntersection(None, None, None, 0.0)
        
        return TriangleIntersection(True, contact_point, normal, penetration)
    
    def _compute_triangle_normal(self, triangle: np.ndarray) -> np.ndarray:
        """
        计算三角形法线
        
        Args:
            triangle: 三角形顶点 (3x3)
            
        Returns:
            法线向量
        """
        v0, v1, v2 = triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        return np.cross(edge1, edge2)
    
    import numpy as np


    def tri_tri_intersect_moller(V0, V1, V2, U0, U1, U2, eps=1e-9):
        """
        Möller 1997 triangle-triangle intersection test.
        Returns (intersect: bool, segment: (P0, P1) or None)
        """

        def dot(a, b): return np.dot(a, b)
        def cross(a, b): return np.cross(a, b)

        # ================================================================
        # 1. Compute plane equation of triangle V
        # ================================================================
        E1 = V1 - V0
        E2 = V2 - V0
        N1 = cross(E1, E2)
        d1 = -dot(N1, V0)

        # signed distances of U0,U1,U2 to plane of V
        du0 = dot(N1, U0) + d1
        du1 = dot(N1, U1) + d1
        du2 = dot(N1, U2) + d1

        # tolerance
        du0 = 0.0 if abs(du0) < eps else du0
        du1 = 0.0 if abs(du1) < eps else du1
        du2 = 0.0 if abs(du2) < eps else du2

        # one side test
        if (du0 > 0 and du1 > 0 and du2 > 0) or (du0 < 0 and du1 < 0 and du2 < 0):
            return False, None

        # ================================================================
        # 2. Compute plane equation of triangle U
        # ================================================================
        E1 = U1 - U0
        E2 = U2 - U0
        N2 = cross(E1, E2)
        d2 = -dot(N2, U0)

        dv0 = dot(N2, V0) + d2
        dv1 = dot(N2, V1) + d2
        dv2 = dot(N2, V2) + d2

        dv0 = 0.0 if abs(dv0) < eps else dv0
        dv1 = 0.0 if abs(dv1) < eps else dv1
        dv2 = 0.0 if abs(dv2) < eps else dv2

        # one side test
        if (dv0 > 0 and dv1 > 0 and dv2 > 0) or (dv0 < 0 and dv1 < 0 and dv2 < 0):
            return False, None

        # ================================================================
        # 3. Check if triangles are coplanar
        # ================================================================
        D = cross(N1, N2)   # direction of intersection line
        if np.linalg.norm(D) < eps:
            # coplanar case not implemented here
            # you can implement 2D projection test if needed
            return False, None

        # choose the largest component of D to project interval
        absD = np.abs(D)
        index = np.argmax(absD)     # 0,1,2

        # ================================================================
        # helper: compute interval for one triangle
        # ================================================================
        def compute_intervals(p0, p1, p2, d0, d1, d2):
            """
            From Möller 1997:
            Returns two scalars isect0, isect1 defining the interval.
            """
            # reorder vertices so d0 and d1 are one side, d2 the other
            if d0 * d1 > 0.0:
                # d2 is alone
                a = p2
                b = p0
                c = p1
                da = d2
                db = d0
                dc = d1
            elif d0 * d2 > 0.0:
                # d1 is alone
                a = p1
                b = p0
                c = p2
                da = d1
                db = d0
                dc = d2
            else:
                # d0 is alone
                a = p0
                b = p1
                c = p2
                da = d0
                db = d1
                dc = d2

            # compute parametric intervals on chosen axis
            denominator = db - da
            if abs(denominator) < eps:
                # 退化情况，使用端点
                return a, b
            t = db / denominator
            isect0 = b + (a - b) * t

            denominator = dc - da
            if abs(denominator) < eps:
                # 退化情况，使用端点
                return a, c
            t = dc / denominator
            isect1 = c + (a - c) * t

            if isect0[index] < isect1[index]:
                return isect0, isect1
            else:
                return isect1, isect0

        # ================================================================
        # 4. compute intervals for both triangles
        # ================================================================
        isectV0, isectV1 = compute_intervals(V0, V1, V2, dv0, dv1, dv2)
        isectU0, isectU1 = compute_intervals(U0, U1, U2, du0, du1, du2)

        # check interval overlap
        if isectV1[index] < isectU0[index] - eps or isectU1[index] < isectV0[index] - eps:
            return False, None

        # ================================================================
        # 5. Compute true 3D intersection segment endpoints
        # ================================================================
        is0 = isectV0 if isectV0[index] > isectU0[index] else isectU0
        is1 = isectV1 if isectV1[index] < isectU1[index] else isectU1

        return True, (is0, is1)
    
    def _coplanar_tri_tri(self, N, V0, V1, V2, U0, U1, U2, eps=1e-9):
        """
        Coplanar triangle-triangle intersection based on Möller's method
        Project onto dominant axis
        """
        # choose projection axis: largest absolute normal component
        absN = np.abs(N)
        if absN[0] > absN[1] and absN[0] > absN[2]:
            i0, i1 = 1, 2  # project to YZ
        elif absN[1] > absN[2]:
            i0, i1 = 0, 2  # project to XZ
        else:
            i0, i1 = 0, 1  # project to XY
        
        # 2D points
        V = [(V0[i0], V0[i1]), (V1[i0], V1[i1]), (V2[i0], V2[i1])]
        U = [(U0[i0], U0[i1]), (U1[i0], U1[i1]), (U2[i0], U2[i1])]
        
        # Standard 2D triangle overlap test
        return self._tri_tri_overlap_2D(V, U)
    
    def _tri_tri_overlap_2D(self, V, U):
        """
        Simple 2D triangle overlap test using separating axis theorem
        V, U: list of 3 (x, y)
        """
        def edge(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        
        # check V's edges
        for i in range(3):
            p = V[i]
            q = V[(i + 1) % 3]
            if all(edge(p, q, u) < 0 for u in U) or all(edge(p, q, u) > 0 for u in U):
                return False
        
        # check U's edges
        for i in range(3):
            p = U[i]
            q = U[(i + 1) % 3]
            if all(edge(p, q, v) < 0 for v in V) or all(edge(p, q, v) > 0 for v in V):
                return False
        
        return True
    
    def _estimate_penetration_depth(self, tri1, tri2, normal):
        """
        使用投影区间重叠计算穿透深度
        """        
        # 分别计算每个三角形的投影范围
        proj1 = np.dot(tri1, normal)
        proj2 = np.dot(tri2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # 计算重叠区间
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)
        
        if overlap_max < overlap_min:
            # 没有重叠
            return 0.0
        
        # 穿透深度 = 重叠区间长度
        penetration = overlap_max - overlap_min
        
        return penetration
    
    def get_contact_count(self) -> int:
        """获取检测到的接触数量"""
        return len(self.contacts)


def create_narrowphase(tolerance: float = 1e-6) -> Narrowphase:
    """
    创建窄相检测器的工厂函数
    """
    return Narrowphase(tolerance)
