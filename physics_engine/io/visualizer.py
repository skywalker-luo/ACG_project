"""
物理仿真可视化模块
使用Open3D进行实时可视化显示刚体运动
"""

import numpy as np
from typing import List, Dict, Optional, Callable
import time
import open3d as o3d

class Visualizer:
    """
    物理世界可视化器
    实时显示刚体的位置、姿态和运动
    """
    
    def __init__(self, 
                 window_name: str = "Physics Simulation",
                 width: int = 1280,
                 height: int = 720,
                 show_axes: bool = True,
                 show_wireframe: bool = False,
                 background_color: tuple = (0.1, 0.1, 0.1)):
        """
        初始化可视化器
        
        Args:
            window_name: 窗口标题
            width: 窗口宽度
            height: 窗口高度
            show_axes: 是否显示坐标轴
            show_wireframe: 是否显示线框模式
            background_color: 背景颜色 (R, G, B)
        """
        
        self.window_name = window_name
        self.width = width
        self.height = height
        self.show_axes = show_axes
        self.show_wireframe = show_wireframe
        self.background_color = background_color
        
        # 可视化窗口
        self.vis = None
        self.view_control = None
        
        # 刚体几何体映射
        self.body_geometries: Dict[int, o3d.geometry.TriangleMesh] = {}
        self.body_colors: Dict[int, np.ndarray] = {}
        
        # 坐标轴
        self.coordinate_frame = None
        
        # 回调函数
        self.update_callback: Optional[Callable] = None
        
        # 统计信息
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        
        # 是否初始化
        self.initialized = False
    
    def initialize(self):
        """初始化可视化窗口"""
        if self.initialized:
            return
        
        # 创建可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=self.window_name,
            width=self.width,
            height=self.height
        )
        
        # 获取视图控制
        self.view_control = self.vis.get_view_control()
        
        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array(self.background_color)
        render_option.light_on = True
        render_option.show_coordinate_frame = False  # 我们自己添加
        
        if self.show_wireframe:
            render_option.mesh_show_wireframe = True
        
        # 添加坐标轴
        if self.show_axes:
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
            )
            self.vis.add_geometry(self.coordinate_frame)
        
        self.initialized = True
    
    def add_body(self, body, color: Optional[np.ndarray] = None):
        """
        添加刚体到可视化
        
        Args:
            body: RigidBody对象
            color: 颜色 (R, G, B)，如果为None则自动分配
        """
        if not self.initialized:
            self.initialize()
        
        body_id = id(body)
        
        # 如果已经存在，先移除
        if body_id in self.body_geometries:
            self.remove_body(body)
        
        # 创建网格几何体
        mesh = self._create_mesh_geometry(body)
        
        # 设置颜色
        if color is None:
            # 自动分配颜色（根据body id生成）
            np.random.seed(body_id % 10000)
            color = np.random.rand(3) * 0.7 + 0.3  # 避免太暗的颜色
        
        mesh.paint_uniform_color(color)
        self.body_colors[body_id] = color
        
        # 添加到可视化
        self.vis.add_geometry(mesh)
        self.body_geometries[body_id] = mesh
    
    def remove_body(self, body):
        """从可视化中移除刚体"""
        body_id = id(body)
        
        if body_id in self.body_geometries:
            self.vis.remove_geometry(self.body_geometries[body_id])
            del self.body_geometries[body_id]
            if body_id in self.body_colors:
                del self.body_colors[body_id]
    
    def update(self, world):
        """
        更新可视化
        
        Args:
            world: World对象
            
        Returns:
            bool: 窗口是否仍然打开
        """
        if not self.initialized:
            self.initialize()
        
        # 更新所有刚体的变换
        for body in world.bodies:
            body_id = id(body)
            
            # 如果刚体还未添加到可视化，自动添加
            if body_id not in self.body_geometries:
                # 静态物体用灰色，动态物体用彩色
                if body.is_static:
                    color = np.array([0.5, 0.5, 0.5])
                else:
                    color = None  # 自动分配
                self.add_body(body, color)
            
            # 更新刚体的变换
            mesh = self.body_geometries[body_id]
            self._update_mesh_transform(mesh, body)
        
        # 调用用户回调
        if self.update_callback:
            self.update_callback(self, world)
        
        # 更新几何体
        for mesh in self.body_geometries.values():
            self.vis.update_geometry(mesh)
        
        # 更新渲染
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # 更新FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        return True  # 继续运行
    
    def run(self, world, dt: float = 1.0/60.0, max_frames: Optional[int] = None):
        """
        运行仿真循环
        
        Args:
            world: World对象
            dt: 仿真时间步长
            max_frames: 最大帧数，None表示无限制
        """
        if not self.initialized:
            self.initialize()
        
        frame = 0
        
        try:
            while True:
                # 物理步进
                world.step(dt)
                
                # 更新可视化
                self.update(world)
                
                # 控制帧率
                time.sleep(dt)
                
                frame += 1
                if max_frames is not None and frame >= max_frames:
                    break
                
        except KeyboardInterrupt:
            print("\n可视化已中断")
        finally:
            self.close()
    
    def close(self):
        """关闭可视化窗口"""
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
            self.initialized = False
    
    def set_camera(self, 
                   front: Optional[np.ndarray] = None,
                   lookat: Optional[np.ndarray] = None,
                   up: Optional[np.ndarray] = None,
                   zoom: Optional[float] = None):
        """
        设置相机位置
        
        Args:
            front: 相机朝向
            lookat: 观察点
            up: 向上方向
            zoom: 缩放级别
        """
        if not self.initialized:
            return
        
        if front is not None:
            self.view_control.set_front(front)
        if lookat is not None:
            self.view_control.set_lookat(lookat)
        if up is not None:
            self.view_control.set_up(up)
        if zoom is not None:
            self.view_control.set_zoom(zoom)
    
    def reset_camera(self):
        """重置相机到默认位置"""
        if self.initialized:
            self.view_control.reset_camera_local_rotate()
    
    def capture_screen(self, filename: str):
        """
        截图保存
        
        Args:
            filename: 保存文件名
        """
        if self.initialized:
            self.vis.capture_screen_image(filename)
            print(f"截图已保存: {filename}")
    
    def get_fps(self) -> float:
        """获取当前FPS"""
        return self.fps
    
    def set_update_callback(self, callback: Callable):
        """
        设置更新回调函数
        
        Args:
            callback: 回调函数，签名为 callback(visualizer, world)
        """
        self.update_callback = callback
    
    def _create_mesh_geometry(self, body) -> o3d.geometry.TriangleMesh:
        """从刚体创建Open3D网格几何体"""
        # 从body.mesh获取顶点和面
        vertices = body.mesh.vertices.astype(np.float64)
        faces = body.mesh.faces.astype(np.int32)
        
        # 创建Open3D网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # 计算法线
        mesh.compute_vertex_normals()
        
        return mesh
    
    def _update_mesh_transform(self, mesh: o3d.geometry.TriangleMesh, body):
        """更新网格的变换矩阵"""
        # 获取刚体的变换矩阵
        transform = self._get_transform_matrix(body)
        
        # 重置到原始顶点
        original_vertices = body.mesh.vertices.astype(np.float64)
        mesh.vertices = o3d.utility.Vector3dVector(original_vertices)
        
        # 应用变换
        mesh.transform(transform)
        
        # 重新计算法线
        mesh.compute_vertex_normals()
    
    def _get_transform_matrix(self, body) -> np.ndarray:
        """获取刚体的4x4变换矩阵"""
        # 从四元数转换为旋转矩阵
        q = body.orientation  # [w, x, y, z]
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # 四元数转旋转矩阵
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # 构造4x4齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = body.position
        
        return T
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


class SimpleVisualizer:
    """
    简化的可视化器（不依赖Open3D）
    仅在终端打印仿真信息
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        初始化简单可视化器
        
        Args:
            update_interval: 更新间隔（秒）
        """
        self.update_interval = update_interval
        self.last_update = time.time()
        self.frame_count = 0
    
    def update(self, world):
        """更新显示"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_update >= self.update_interval:
            stats = world.get_statistics()
            
            print(f"\n{'='*60}")
            print(f"仿真步数: {stats['step_count']}")
            print(f"仿真时间: {stats['total_time']:.2f}s")
            print(f"刚体数量: {stats['bodies_count']} ({stats['active_bodies']} 个活动)")
            print(f"接触点数: {stats['contacts_count']}")
            print(f"FPS: {self.frame_count / (current_time - self.last_update):.1f}")
            print(f"{'='*60}")
            
            # 显示每个刚体的状态
            for i, body in enumerate(world.bodies):
                status = "静态" if body.is_static else "动态"
                print(f"  刚体{i} ({status}):")
                print(f"    位置: [{body.position[0]:7.2f}, {body.position[1]:7.2f}, {body.position[2]:7.2f}]")
                print(f"    速度: [{body.velocity[0]:7.2f}, {body.velocity[1]:7.2f}, {body.velocity[2]:7.2f}]")
            
            self.frame_count = 0
            self.last_update = current_time
        
        return True
    
    def run(self, world, dt: float = 1.0/60.0, max_frames: Optional[int] = None):
        """运行仿真循环"""
        frame = 0
        
        try:
            while True:
                world.step(dt)
                self.update(world)
                
                frame += 1
                if max_frames is not None and frame >= max_frames:
                    break
                
                time.sleep(dt)  # 控制帧率
                
        except KeyboardInterrupt:
            print("\n仿真已中断")
    
    def close(self):
        """关闭（占位符）"""
        pass


def create_visualizer(use_open3d: bool = True, **kwargs) -> 'Visualizer':
    """
    创建可视化器的便捷函数
    """
    if use_open3d:
        return Visualizer(**kwargs)
    else:
        return SimpleVisualizer(**kwargs)
