"""
template file
运行物理仿真并导出为Blender脚本
"""

import numpy as np
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physics_engine.geometry.mesh import Mesh
from physics_engine.dynamics.rigid_body import RigidBody
from physics_engine.dynamics.world import World
from physics_engine.io.blender_exporter import BlenderExporter
from physics_engine.config.settings import config
from physics_engine.utils.contact_debugger import ContactDebugger


def run_simulation_export(
    # ============ 必需参数 ============
    title: str,
    description: str,
    
    # ============ 网格参数 ============
    object_physics_file: List[str],
    object_render_file: List[str],
    object_scale_factors: List[float],
    
    # ============ 物理属性参数 ============
    is_static_list: List[bool],
    position_list: List[np.ndarray],
    orientation_list: List[np.ndarray],
    density_list: List[float],
    restitution_list: List[float],
    friction_list: List[float],
    velocity_list: Optional[List[np.ndarray]] = None,
    angular_velocity_list: Optional[List[np.ndarray]] = None,
    object_shape_list: Optional[List[str]] = None,
    object_shape_params_list: Optional[List[dict]] = None,
    
    # ============ 仿真参数 ============
    duration: float = 3.33,  # 仿真时长（秒）
    fps: float = 240.0,      # 导出帧率
    gravity: np.ndarray = config.physics.GRAVITY_EARTH,
    
    # ============ 输出参数 ============
    enable_warmstarting: bool = False,
    
    # ============ 其他配置 ============
    **kwargs  # 额外参数，用于未来扩展
):
    # ========================================
    # 1. 加载网格（物理用 + 渲染用）
    # ========================================
    print("\n【1】加载网格...")

    object_physics_path_list = [project_root / "mesh" / f for f in object_physics_file]
    object_render_path_list = [project_root / "mesh" / f for f in object_render_file]

    # 检查文件是否存在
    for path in object_physics_path_list:
        if not path.exists():
            raise FileNotFoundError(f"找不到网格文件: {path}")
    for path in object_render_path_list:
        if not path.exists():
            raise FileNotFoundError(f"找不到网格文件: {path}")
    
    # 加载物理网格
    object_physics_meshes = []
    for idx, (path, scale) in enumerate(zip(object_physics_path_list, object_scale_factors)):
        shape = None
        shape_params = None
        if object_shape_list is not None and idx < len(object_shape_list):
            shape = object_shape_list[idx]
        if object_shape_params_list is not None and idx < len(object_shape_params_list):
            shape_params = object_shape_params_list[idx]

        if shape is not None:
            mesh = Mesh.from_file(str(path), shape=shape, shape_params=shape_params)
        else:
            mesh = Mesh.from_file(str(path))

        mesh.scale(scale)
        object_physics_meshes.append(mesh)
        print(f"  ✓ 物理网格: {path.name} - {len(mesh.vertices)} 顶点, {len(mesh.faces)} 三角形")
    
    # 加载渲染网格
    object_render_meshes = []
    for idx, (path, scale) in enumerate(zip(object_render_path_list, object_scale_factors)):
        # 渲染网格通常不需要 shape 信息，但如果提供则传递
        shape = None
        shape_params = None
        if object_shape_list is not None and idx < len(object_shape_list):
            shape = object_shape_list[idx]
        if object_shape_params_list is not None and idx < len(object_shape_params_list):
            shape_params = object_shape_params_list[idx]

        if shape is not None:
            mesh = Mesh.from_file(str(path), shape=shape, shape_params=shape_params)
        else:
            mesh = Mesh.from_file(str(path))

        mesh.scale(scale)
        object_render_meshes.append(mesh)
        print(f"  ✓ 渲染网格: {path.name} - {len(mesh.vertices)} 顶点, {len(mesh.faces)} 三角形")
    
    # ========================================
    # 2. 创建刚体（使用物理网格）
    # ========================================
    print("\n【2】创建刚体...")

    object_vertices = [mesh.vertices for mesh in object_physics_meshes]
    object_mins = [verts.min(axis=0) for verts in object_vertices]
    object_maxs = [verts.max(axis=0) for verts in object_vertices]

    body_list = []
    
    for i in range(len(object_physics_meshes)):
        body = RigidBody(
            mesh=object_physics_meshes[i],
            density=density_list[i],
            position=position_list[i],
            orientation=orientation_list[i],
            velocity=None if is_static_list[i] else velocity_list[i],
            angular_velocity=None if is_static_list[i] else angular_velocity_list[i],
            restitution=restitution_list[i],
            friction=friction_list[i],
            is_static=is_static_list[i]
        )
        body_list.append(body)
        body_type = "静态" if is_static_list[i] else "动态"
        print(f"  ✓ 刚体 {i+1}: {body_type}, 位置={position_list[i]}")
    
    # ========================================
    # 3. 创建物理世界
    # ========================================
    print("\n【3】创建物理世界...")
    world = World(
        gravity=gravity,
        broadphase_type='simple',
        solver_type='impulse',
        enable_warmstarting=enable_warmstarting,
        max_substeps=config.simulation.MAX_SUBSTEPS,
        fixed_timestep=config.simulation.FIXED_TIMESTEP,
        debug=False
    )
    
    for body in body_list:
        world.add_body(body)
    
    print(f"  ✓ 重力: {world.gravity}")
    print(f"  ✓ 固定时间步: {world.fixed_timestep*1000:.2f}ms")
    print(f"  ✓ 刚体数: {len(world.bodies)}")
    
    # ========================================
    # 4. 创建Blender导出器和接触调试器
    # ========================================
    print("\n【4】创建Blender导出器和调试器...")
    exporter = BlenderExporter(world)
    exporter.set_export_settings(
        fps=fps,
        coordinate_system="blender",
        export_static_bodies=True
    )
    
    # 为每个刚体设置高精度渲染网格
    for body, render_mesh in zip(body_list, object_render_meshes):
        exporter.set_render_mesh(body, render_mesh)
    
    print(f"  ✓ 导出帧率: {exporter.fps} FPS")
    print(f"  ✓ 已设置高精度渲染网格")
    
    # 创建接触调试器
    debugger = ContactDebugger()
    print(f"  ✓ 接触调试器已创建")
    
    # ========================================
    # 5. 运行仿真并捕获帧
    # ========================================
    print("\n【5】运行物理仿真...")
    
    duration = duration
    dt = 1.0 / fps
    total_steps = int(duration * fps)
    capture_interval = 1  # 每帧都捕获
    
    print(f"  时长: {duration}s")
    print(f"  时间步: {dt*1000:.2f}ms")
    print(f"  总步数: {total_steps}")
    
    collision_count = 0
    first_collision = True
    last_print_time = 0
    
    for step in range(total_steps):
        step_start = __import__('time').time()
        world.step(dt)
        step_time = __import__('time').time() - step_start
        
        # 捕获帧
        if step % capture_interval == 0:
            exporter.capture_frame()
        
        # 记录接触信息用于调试
        if world.contacts_count > 0:
            debugger.record_contacts(step, world.contacts)
        
        # 检测碰撞
        if world.contacts_count > 0 and first_collision:
            print(f"  💥 首次碰撞！时间={world.total_time:.2f}s")
            first_collision = False
        
        if world.contacts_count > 0:
            collision_count += 1
        
        # 进度显示（更频繁，包含性能信息）
        if step % 10 == 0 or step_time > 0.1:  # 每10步或慢步骤
            current_time = __import__('time').time()
            if current_time - last_print_time > 1.0 or step_time > 0.1:  # 每秒或慢步骤
                print(f"  帧 {step}/{total_steps} ({step/total_steps*100:.1f}%) | "
                          f"接触数={world.contacts_count} | "
                          f"步骤耗时={step_time*1000:.1f}ms")
                for i, body in enumerate(body_list):
                    if not body.is_static:
                        print(f"  Object {i+1} Y={body.position[1]:.2f}m | "
                              f"速度={np.linalg.norm(body.velocity):.2f}m/s | "
                              f"角速度={np.linalg.norm(body.angular_velocity):.2f}rad/s | ")
                
                last_print_time = current_time
    
    print(f"\n✓ 仿真完成")
    print(f"  总步数: {step + 1}")
    print(f"  捕获帧数: {len(exporter.frames_data)}")
    print(f"  碰撞帧数: {collision_count}")
    for i, body in enumerate(body_list):
        if not body.is_static:
            print(f"\n  刚体 {i+1} 最终状态:")
            print(f"  最终位置: {body.position}")
            print(f"  最终速度: {np.linalg.norm(body.velocity):.3f} m/s")
            print(f"  最终角速度: {np.linalg.norm(body.angular_velocity):.3f} rad/s")
    
    # 显示接触统计
    if debugger.contact_history:
        debugger.print_contact_summary()
    else:
        print(f"\n  ℹ️ 未检测到接触")
    
    # ========================================
    # 6. 导出数据
    # ========================================
    print("\n【6】导出仿真数据...")

    output_dir = project_root / "output" / title
    output_dir.mkdir(exist_ok=True)
    
    # 导出JSON数据
    json_path = output_dir / f"{title}_simulation.json"
    exporter.export_json(str(json_path))
    
    # 导出Blender脚本
    blender_script_path = output_dir / f"{title}_blender.py"
    exporter.export_blender_script(str(blender_script_path), description)
    
    # 导出接触调试可视化脚本
    if debugger.contact_history:
        debug_script_path = output_dir / "contact_debug_blender.py"
        debug_code = debugger.generate_blender_visualization_code()
        with open(debug_script_path, 'w', encoding='utf-8') as f:
            f.write(debug_code)
        print(f"  ✓ 接触调试脚本: {debug_script_path}")
        print(f"     在Blender中运行此脚本可视化接触点、法线和穿透深度")
    
    # 生成导入说明
    instructions_path = output_dir / "blender_import_instructions.txt"
    instructions = exporter.generate_blender_import_instructions(str(json_path.absolute()))
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    print(f"  ✓ 导入说明: {instructions_path}")
    
    # 导出仿真统计

    # 准备物体列表（假设每个物体都有唯一名称）
    objects = []
    for i, body in enumerate(body_list):
        obj_name = f"object_{i+1}"
        obj_data = {
            "name": obj_name,
            "body": body,
            "initial_position": position_list[i],
            "physics_mesh": object_physics_meshes[i],
            "render_mesh": object_render_meshes[i],
            "mass": body.mass,
            "is_static": body.is_static
        }
        objects.append(obj_data)

    # 构建最终的stats字典
    stats = {
        "metadata": {
            "scene_name": "Bunny碰撞Icosahedron",
            "duration": float(world.total_time),
            "total_frames": len(exporter.frames_data),
            "fps": exporter.fps,
            "collision_frames": collision_count,
            "object_count": len(objects)  # 新增：物体数量
        },
        "objects": {},  # 改为字典结构
        "physics": {
            "gravity": [float(x) for x in world.gravity],
            "timestep": float(world.fixed_timestep),
            "solver": "impulse",
            "warmstart": True
        }
    }

    # 动态填充物体数据
    for obj_data in objects:
        obj_name = obj_data["name"]
        stats["objects"][obj_name] = generate_object_stats(obj_data)

    stats_path = output_dir / f"{title}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(to_serializable(stats), f, indent=2)
    print(f"  ✓ 统计数据: {stats_path}")
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 60)
    print("✅ 导出完成！")
    print("=" * 60)
    print(f"\n📁 输出文件:")
    print(f"  • Blender脚本: {blender_script_path.name}")
    print(f"  • JSON数据: {json_path.name}")
    print(f"  • 统计数据: {stats_path.name}")
    print(f"  • 导入说明: {instructions_path.name}")
    print(f"\n🎬 在Blender中打开脚本:")
    print(f"  1. 打开Blender")
    print(f"  2. 切换到Scripting标签页")
    print(f"  3. 打开: {blender_script_path.absolute()}")
    print(f"  4. 点击运行按钮")
    print(f"  5. 按空格键播放动画")
    print("\n" + "=" * 60)

# 动态生成统计数据的函数
def generate_object_stats(obj_data):
    """为单个物体生成统计数据"""
    body = obj_data["body"]
    stats = {
        "name": obj_data["name"],
        "is_static": obj_data["is_static"],
        "initial_position": [float(x) for x in obj_data["initial_position"]],
        "final_position": [float(x) for x in body.position] if not obj_data["is_static"] else obj_data["initial_position"],
        "final_velocity": [float(x) for x in body.velocity] if not obj_data["is_static"] else [0.0, 0.0, 0.0],
        "mass": float(obj_data["mass"]),
        "physics_vertices": len(obj_data["physics_mesh"].vertices),
        "physics_faces": len(obj_data["physics_mesh"].faces),
        "render_vertices": len(obj_data["render_mesh"].vertices),
        "render_faces": len(obj_data["render_mesh"].faces)
    }
    return stats

# 保存JSON文件（与原来相同）
def to_serializable(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj