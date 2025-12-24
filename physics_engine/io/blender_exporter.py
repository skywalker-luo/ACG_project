"""
Blender导出器
将物理仿真结果导出为可在Blender中播放的Python脚本
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from physics_engine.geometry.mesh import Mesh


class BlenderExporter:
    """
    物理仿真到Blender的导出器
    记录仿真帧数据并生成Blender Python脚本
    """
    
    def __init__(self, world=None):
        """
        初始化导出器
        
        Args:
            world: World对象（可选）
        """
        self.world = world
        self.frames_data = []
        self.bodies_info = {}
        self.render_meshes = {}  # 存储高精度渲染网格 {body_id: mesh}
        self.fps = 24.0
        self.coordinate_system = "blender"  # "blender" 或 "opengl"
        self.export_static_bodies = True
        
    def set_export_settings(self, fps: float = 24.0, 
                           coordinate_system: str = "blender",
                           export_static_bodies: bool = True):
        """
        设置导出参数
        
        Args:
            fps: 目标帧率
            coordinate_system: 坐标系统 ("blender" 或 "opengl")
            export_static_bodies: 是否导出静态物体
        """
        self.fps = fps
        self.coordinate_system = coordinate_system
        self.export_static_bodies = export_static_bodies
    
    def set_render_mesh(self, body, render_mesh):
        """
        为刚体设置高精度渲染网格（用于导出，不影响物理计算）
        
        重要：渲染网格的质心必须与物理网格对齐到局部坐标原点
        
        Args:
            body: RigidBody对象
            render_mesh: 高精度Mesh对象（用于渲染）
        """
        from physics_engine.dynamics.mesh_property import MeshPropertyCalculator
        
        body_id = id(body)
        
        # 创建渲染网格的副本（避免修改原始网格）
        render_mesh_copy = Mesh(
            vertices=render_mesh.vertices.copy(),
            faces=render_mesh.faces.copy()
        )
        
        # 计算渲染网格的质心
        calculator = MeshPropertyCalculator(render_mesh_copy, density=1.0)
        _, _, center, _ = calculator.compute_mesh_properties()
        
        # 将渲染网格的质心平移到局部坐标原点（与物理网格对齐）
        if center is not None and np.linalg.norm(center) > 1e-10:
            render_mesh_copy.vertices -= center
        
        self.render_meshes[body_id] = render_mesh_copy
    
    def capture_frame(self):
        """捕获当前帧的所有刚体状态"""
        if self.world is None:
            raise RuntimeError("World对象未设置")
        
        frame_data = {
            'time': self.world.total_time,
            'bodies': {}
        }
        
        for body in self.world.bodies:
            body_id = id(body)
            
            # 首次遇到此刚体，记录其信息
            if body_id not in self.bodies_info:
                # 优先使用高精度渲染网格，否则使用物理网格
                mesh_to_use = self.render_meshes.get(body_id, body.mesh)
                self.bodies_info[body_id] = {
                    'name': f"Body_{body_id}",
                    'is_static': body.is_static,
                    'mesh': mesh_to_use,
                    'has_mesh': mesh_to_use is not None
                }
            
            # 记录变换数据
            frame_data['bodies'][body_id] = {
                'position': body.position.copy(),
                'orientation': body.orientation.copy(),  # 四元数 [w,x,y,z]
                'velocity': body.velocity.copy(),
                'angular_velocity': body.angular_velocity.copy()
            }
        
        self.frames_data.append(frame_data)
    
    def export_json(self, filepath: str):
        """
        导出为JSON格式
        
        Args:
            filepath: 输出文件路径
        """
        data = {
            'metadata': {
                'fps': self.fps,
                'total_frames': len(self.frames_data),
                'coordinate_system': self.coordinate_system,
                'total_time': self.frames_data[-1]['time'] if self.frames_data else 0.0
            },
            'bodies': {
                str(body_id): {
                    'name': info['name'],
                    'is_static': info['is_static'],
                    'has_mesh': info['has_mesh']
                }
                for body_id, info in self.bodies_info.items()
            },
            'frames': [
                {
                    'time': float(frame['time']),
                    'bodies': {
                        str(body_id): {
                            'position': [float(x) for x in data['position']],
                            'orientation': [float(x) for x in data['orientation']],
                            'velocity': [float(x) for x in data['velocity']],
                            'angular_velocity': [float(x) for x in data['angular_velocity']]
                        }
                        for body_id, data in frame['bodies'].items()
                    }
                }
                for frame in self.frames_data
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ JSON数据已导出: {filepath}")
    
    def export_blender_script(self, filepath: str, scene_name: str = "PhysicsSimulation"):
        """
        导出Blender Python脚本
        
        Args:
            filepath: 输出脚本路径
            scene_name: 场景名称
        """
        script = self._generate_blender_script(scene_name)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(script)
        
        print(f"✓ Blender脚本已导出: {filepath}")
    
    def _generate_blender_script(self, scene_name: str) -> str:
        """生成Blender Python脚本内容"""
        
        # 生成网格数据字典
        meshes_data = {}
        for body_id, info in self.bodies_info.items():
            if info['has_mesh']:
                mesh = info['mesh']
                meshes_data[body_id] = {
                    'vertices': mesh.vertices.tolist(),
                    'faces': mesh.faces.tolist()
                }
        
        # 生成帧数据
        frames_json = json.dumps([
            {
                'time': float(frame['time']),
                'bodies': {
                    str(body_id): {
                        'position': [float(x) for x in data['position']],
                        'orientation': [float(x) for x in data['orientation']]
                    }
                    for body_id, data in frame['bodies'].items()
                }
            }
            for frame in self.frames_data
        ], indent=4)
        
        meshes_json = json.dumps({
            str(k): v for k, v in meshes_data.items()
        }, indent=4)
        
        # 将字典转换为Python代码（使用True/False而不是true/false）
        bodies_info_py = "{\n"
        for body_id, info in self.bodies_info.items():
            bodies_info_py += f'    "{body_id}": {{\n'
            bodies_info_py += f'        "name": "{info["name"]}",\n'
            bodies_info_py += f'        "is_static": {info["is_static"]},\n'
            bodies_info_py += f'        "has_mesh": {info["has_mesh"]}\n'
            bodies_info_py += '    },\n'
        bodies_info_py += "}"
        
        script = f'''"""
{scene_name} - Blender动画脚本
由BlenderExporter自动生成

使用方法：
1. 打开Blender
2. 切换到Scripting标签页
3. 打开此脚本
4. 点击运行按钮
5. 播放动画查看物理仿真结果
"""

import bpy
import json
from mathutils import Vector, Quaternion, Matrix

# ============================================================================
# 仿真数据
# ============================================================================

FPS = {self.fps}
SCENE_NAME = "{scene_name}"

# 刚体信息
BODIES_INFO = {bodies_info_py}

# 网格数据
MESHES_DATA = {meshes_json}

# 帧数据
FRAMES_DATA = {frames_json}


# ============================================================================
# 辅助函数
# ============================================================================

def clear_scene():
    """清空场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)


def create_mesh_object(vertices, faces, name="Mesh"):
    """从顶点和面创建网格对象（Y-up转Z-up）"""
    # 坐标转换：Y-up (physics) -> Z-up (Blender)
    # (x, y, z) -> (x, -z, y)  修正z方向
    vertices_zup = [[v[0], -v[2], v[1]] for v in vertices]
    
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices_zup, [], faces)
    mesh.update()
    
    # 计算法线（使用validate确保mesh有效）
    mesh.validate()
    mesh.update(calc_edges=True)
    
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    return obj


def set_material(obj, color=(0.8, 0.3, 0.2), is_static=False):
    """设置材质"""
    mat = bpy.data.materials.new(name=f"{{obj.name}}_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    if is_static:
        bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.3
    else:
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.0
    
    bsdf.inputs['Roughness'].default_value = 0.4
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def setup_camera(location=(10, -10, 8), target=(0, 2, 0)):
    """设置相机"""
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    
    direction = Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    bpy.context.scene.camera = camera


def setup_lighting():
    """设置光照"""
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 8))
    area = bpy.context.active_object
    area.data.energy = 500
    area.data.size = 10


def apply_animation_data(objects_map, frames_data):
    """应用动画数据到对象（Y-up转Z-up）"""
    print(f"应用动画: {{len(frames_data)}} 帧")
    
    for frame_idx, frame in enumerate(frames_data):
        for body_id, transform in frame['bodies'].items():
            if body_id not in objects_map:
                continue
            
            obj = objects_map[body_id]
            
            # 坐标转换：Y-up -> Z-up
            # 位置转换: (x, y, z) -> (x, -z, y)  修正z方向
            pos = transform['position']
            obj.location = Vector([pos[0], -pos[2], pos[1]])
            
            # 四元数转换：[w,x,-z,y]
            quat = transform['orientation']
            quat_zup = Quaternion((quat[0], quat[1], -quat[3], quat[2]))
            
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = quat_zup
            
            # 插入关键帧
            obj.keyframe_insert(data_path="location", frame=frame_idx)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print(f"{{SCENE_NAME}} - 导入物理仿真")
    print("=" * 60)
    
    # 1. 清空场景
    print("\\n[1] 清空场景...")
    clear_scene()
    
    # 2. 创建网格对象
    print("\\n[2] 创建网格对象...")
    objects_map = {{}}
    colors = [
        (0.8, 0.3, 0.2),  # 红橙色
        (0.2, 0.6, 0.9),  # 蓝色
        (0.3, 0.8, 0.3),  # 绿色
        (0.9, 0.7, 0.2),  # 黄色
        (0.7, 0.3, 0.8),  # 紫色
    ]
    
    for idx, (body_id, info) in enumerate(BODIES_INFO.items()):
        if not info['has_mesh']:
            print(f"  跳过无网格物体: {{info['name']}}")
            continue
        
        if body_id not in MESHES_DATA:
            print(f"  ⚠️ 警告: 未找到网格数据 {{body_id}}")
            continue
        
        mesh_data = MESHES_DATA[body_id]
        obj = create_mesh_object(
            mesh_data['vertices'],
            mesh_data['faces'],
            info['name']
        )
        
        # 设置材质
        color = colors[idx % len(colors)]
        set_material(obj, color=color, is_static=info['is_static'])
        
        objects_map[body_id] = obj
        print(f"  ✓ {{info['name']}}: {{len(mesh_data['vertices'])}} 顶点, {{len(mesh_data['faces'])}} 面")
    
    # 3. 应用动画数据
    print("\\n[3] 应用动画数据...")
    apply_animation_data(objects_map, FRAMES_DATA)
    
    # 4. 设置场景
    print("\\n[4] 设置场景...")
    setup_camera()
    setup_lighting()
    
    # 5. 配置时间轴
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(FRAMES_DATA) - 1
    bpy.context.scene.render.fps = int(FPS)
    
    # 6. 渲染设置
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    
    print("\\n" + "=" * 60)
    print("✅ 导入完成！")
    print("=" * 60)
    print(f"  对象数: {{len(objects_map)}}")
    print(f"  总帧数: {{len(FRAMES_DATA)}}")
    print(f"  帧率: {{FPS}} FPS")
    print(f"  时长: {{len(FRAMES_DATA) / FPS:.2f}}秒")
    print("\\n提示:")
    print("  • 按空格键播放动画")
    print("  • 拖动时间轴查看不同帧")
    print("  • Ctrl+F12 渲染动画")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''
        
        return script
    
    def generate_blender_import_instructions(self, json_path: str) -> str:
        """
        生成Blender导入说明
        
        Args:
            json_path: JSON数据文件路径
            
        Returns:
            导入说明文本
        """
        instructions = f"""
Blender导入说明
{'='*60}

1. 打开Blender（推荐3.0+版本）

2. 切换到Scripting标签页

3. 点击"Open"按钮打开生成的脚本文件

4. 点击▶️运行脚本按钮

5. 播放动画：
   - 按空格键开始/暂停播放
   - 拖动时间轴查看任意帧
   - 使用鼠标中键旋转视角

6. 渲染动画（可选）：
   - 按Ctrl+F12开始渲染
   - 渲染设置：Cycles引擎，128采样，1920x1080

数据文件: {json_path}
帧率: {self.fps} FPS
总帧数: {len(self.frames_data)}
仿真时长: {self.frames_data[-1]['time'] if self.frames_data else 0.0:.2f}秒

{'='*60}
"""
        return instructions
