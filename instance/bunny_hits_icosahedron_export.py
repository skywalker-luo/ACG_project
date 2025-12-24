"""
Bunny碰撞Icosahedron演示 - 导出Blender版本
运行物理仿真并导出为Blender脚本
"""

import numpy as np
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from physics_engine.geometry.mesh import Mesh
from physics_engine.dynamics.rigid_body import RigidBody
from physics_engine.dynamics.world import World
from physics_engine.io.blender_exporter import BlenderExporter
from physics_engine.config.settings import config
from physics_engine.utils.contact_debugger import ContactDebugger


def main():
    print("=" * 60)
    print("BUNNY碰撞ICOSAHEDRON - BLENDER导出演示")
    print("=" * 60)
    
    # ========================================
    # 1. 加载网格（物理用 + 渲染用）
    # ========================================
    print("\n【1】加载网格...")
    
    # 物理模拟用网格（低精度，快速）
    bunny_physics_path = project_root / "mesh" / "bunny_200_subdivided_1.obj"
    icosahedron_physics_path = project_root / "mesh" / "icosahedron_subdivided_2.obj"
    
    # 渲染用网格（高精度，高质量）
    bunny_render_path = project_root / "mesh" / "bunny_200_subdivided_3.obj"
    icosahedron_render_path = project_root / "mesh" / "icosahedron_subdivided_4.obj"
    
    # 检查文件是否存在
    for path in [bunny_physics_path, icosahedron_physics_path, bunny_render_path, icosahedron_render_path]:
        if not path.exists():
            raise FileNotFoundError(f"找不到网格文件: {path}")
    
    # 加载物理网格
    bunny_physics_mesh = Mesh.from_file(str(bunny_physics_path))
    icosahedron_physics_mesh = Mesh.from_file(str(icosahedron_physics_path))
    bunny_physics_mesh.scale(2.0)  # 放大Bunny
    
    # 加载渲染网格
    bunny_render_mesh = Mesh.from_file(str(bunny_render_path))
    icosahedron_render_mesh = Mesh.from_file(str(icosahedron_render_path))
    bunny_render_mesh.scale(2.0)  # 相同的缩放
    
    print(f"  ✓ Bunny (物理): {len(bunny_physics_mesh.vertices)} 顶点, {len(bunny_physics_mesh.faces)} 三角形")
    print(f"  ✓ Bunny (渲染): {len(bunny_render_mesh.vertices)} 顶点, {len(bunny_render_mesh.faces)} 三角形")
    print(f"  ✓ Icosahedron (物理): {len(icosahedron_physics_mesh.vertices)} 顶点, {len(icosahedron_physics_mesh.faces)} 三角形")
    print(f"  ✓ Icosahedron (渲染): {len(icosahedron_render_mesh.vertices)} 顶点, {len(icosahedron_render_mesh.faces)} 三角形")
    
    # ========================================
    # 2. 创建刚体（使用物理网格）
    # ========================================
    print("\n【2】创建刚体...")
    
    bunny_vertices = bunny_physics_mesh.vertices
    bunny_min = bunny_vertices.min(axis=0)
    bunny_max = bunny_vertices.max(axis=0)
    
    icosahedron_vertices = icosahedron_physics_mesh.vertices
    icosahedron_max = icosahedron_vertices.max(axis=0)
    
    # Icosahedron (静态，使用物理网格)
    icosahedron_position = np.array([0.0, 0.0, 0.0])
    icosahedron_body = RigidBody(
        mesh=icosahedron_physics_mesh,
        density=1000.0,
        position=icosahedron_position,
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        restitution=0.8,
        is_static=True
    )
    print(f"  ✓ Icosahedron: 静态，位置={icosahedron_position}")
    
    # Bunny (动态，使用物理网格)
    vertical_gap = 1.0  # 减小到1米，降低碰撞速度
    bunny_position = np.array([
        0.0,
        icosahedron_max[1] + vertical_gap - bunny_min[1],
        0.0
    ])
    
    # 使用单位四元数，让OBJ文件的原始姿态决定bunny的朝向
    # Blender脚本会自动做Y-up到Z-up的坐标转换
    bunny_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    
    bunny_body = RigidBody(
        mesh=bunny_physics_mesh,
        density=800.0,
        position=bunny_position,
        orientation=bunny_orientation,
        velocity=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        restitution=0.8,
        is_static=False
    )
    
    volume, mass, center, inertia = bunny_body.get_mesh_properties()
    print(f"  ✓ Bunny: 质量={mass:.3f}kg, 初始高度={bunny_position[1]:.2f}m")
    
    # ========================================
    # 3. 创建物理世界
    # ========================================
    print("\n【3】创建物理世界...")
    world = World(
        gravity=config.physics.GRAVITY_EARTH,
        broadphase_type='simple',
        solver_type='impulse',
        enable_warmstarting=False,  # 关闭warm-start测试
        max_substeps=config.simulation.MAX_SUBSTEPS,
        fixed_timestep=config.simulation.FIXED_TIMESTEP,
        debug=False
    )
    
    world.add_body(icosahedron_body)
    world.add_body(bunny_body)
    
    print(f"  ✓ 重力: {world.gravity}")
    print(f"  ✓ 固定时间步: {world.fixed_timestep*1000:.2f}ms")
    print(f"  ✓ 刚体数: {len(world.bodies)}")
    
    # ========================================
    # 4. 创建Blender导出器和接触调试器
    # ========================================
    print("\n【4】创建Blender导出器和调试器...")
    exporter = BlenderExporter(world)
    exporter.set_export_settings(
        fps=240.0,
        coordinate_system="blender",
        export_static_bodies=True
    )
    
    # 为每个刚体设置高精度渲染网格
    exporter.set_render_mesh(bunny_body, bunny_render_mesh)
    exporter.set_render_mesh(icosahedron_body, icosahedron_render_mesh)
    
    print(f"  ✓ 导出帧率: {exporter.fps} FPS")
    print(f"  ✓ 已设置高精度渲染网格")
    
    # 创建接触调试器
    debugger = ContactDebugger()
    print(f"  ✓ 接触调试器已创建")
    
    # ========================================
    # 5. 运行仿真并捕获帧
    # ========================================
    print("\n【5】运行物理仿真...")
    
    duration = 3.33  # 仿真时长（800帧 @ 240fps = 3.33秒）
    dt = 1.0 / 240.0  # 时间步长（240 FPS）
    total_steps = 800  # 800步 @ 240fps
    capture_interval = 1  # 每帧都捕获
    
    print(f"  时长: {duration}s")
    print(f"  时间步: {dt*1000:.2f}ms")
    print(f"  总步数: {total_steps}")
    
    collision_count = 0
    first_collision = True
    last_print_time = 0
    
    for step in range(total_steps):
        # 物理步进
        old_velocity = bunny_body.velocity.copy()
        
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
                angular_speed = np.linalg.norm(bunny_body.angular_velocity)
                print(f"  帧 {step}/{total_steps} ({step/total_steps*100:.1f}%) | "
                      f"Bunny Y={bunny_body.position[1]:.2f}m | "
                      f"速度={np.linalg.norm(bunny_body.velocity):.2f}m/s | "
                      f"角速度={angular_speed:.2f}rad/s | "
                      f"接触数={world.contacts_count} | "
                      f"步骤耗时={step_time*1000:.1f}ms")
                last_print_time = current_time
        
        # 提前结束条件：静止
        if step > 120 and np.linalg.norm(bunny_body.velocity) < 0.01:
            print(f"  ⏹️ Bunny已静止，提前结束 (t={world.total_time:.2f}s)")
            # 继续捕获几帧静止状态
            for _ in range(30):
                exporter.capture_frame()
            break
    
    print(f"\n✓ 仿真完成")
    print(f"  总步数: {step + 1}")
    print(f"  捕获帧数: {len(exporter.frames_data)}")
    print(f"  碰撞帧数: {collision_count}")
    print(f"  最终位置: {bunny_body.position}")
    print(f"  最终速度: {np.linalg.norm(bunny_body.velocity):.3f} m/s")
    
    # 显示接触统计
    if debugger.contact_history:
        debugger.print_contact_summary()
    else:
        print(f"\n  ℹ️ 未检测到接触")
    
    # ========================================
    # 6. 导出数据
    # ========================================
    print("\n【6】导出仿真数据...")
    
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 导出JSON数据
    json_path = output_dir / "bunny_icosahedron_simulation.json"
    exporter.export_json(str(json_path))
    
    # 导出Blender脚本
    blender_script_path = output_dir / "bunny_icosahedron_blender.py"
    exporter.export_blender_script(str(blender_script_path), "Bunny_Hits_Icosahedron")
    
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
    stats = {
        "metadata": {
            "scene_name": "Bunny碰撞Icosahedron",
            "duration": float(world.total_time),
            "total_frames": len(exporter.frames_data),
            "fps": exporter.fps,
            "collision_frames": collision_count
        },
        "bunny": {
            "initial_position": [float(x) for x in bunny_position],
            "final_position": [float(x) for x in bunny_body.position],
            "final_velocity": [float(x) for x in bunny_body.velocity],
            "mass": float(mass),
            "physics_vertices": len(bunny_physics_mesh.vertices),
            "physics_faces": len(bunny_physics_mesh.faces),
            "render_vertices": len(bunny_render_mesh.vertices),
            "render_faces": len(bunny_render_mesh.faces)
        },
        "icosahedron": {
            "position": [float(x) for x in icosahedron_position],
            "is_static": True,
            "physics_vertices": len(icosahedron_physics_mesh.vertices),
            "physics_faces": len(icosahedron_physics_mesh.faces),
            "render_vertices": len(icosahedron_render_mesh.vertices),
            "render_faces": len(icosahedron_render_mesh.faces)
        },
        "physics": {
            "gravity": [float(x) for x in world.gravity],
            "timestep": float(world.fixed_timestep),
            "solver": "impulse",
            "warmstart": True
        }
    }
    
    stats_path = output_dir / "bunny_icosahedron_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
