"""
接触点调试可视化工具
用于在导出的Blender脚本中可视化接触信息
"""

import numpy as np
from typing import List


class ContactDebugger:
    """接触点调试器"""
    
    def __init__(self):
        self.contact_history = []
        self.frame_contacts = {}
    
    def record_contacts(self, frame_num: int, contacts: List):
        """
        记录一帧的所有接触点
        
        Args:
            frame_num: 帧号
            contacts: ContactInfo列表
        """
        frame_data = []
        
        for contact in contacts:
            contact_data = {
                'point': contact.contact_point.tolist(),
                'normal': contact.normal.tolist(),
                'penetration': float(contact.penetration),
                'body1_id': id(contact.body1),
                'body2_id': id(contact.body2),
                'normal_velocity': float(contact.get_normal_velocity())
            }
            frame_data.append(contact_data)
        
        self.frame_contacts[frame_num] = frame_data
        self.contact_history.extend(frame_data)
    
    def generate_blender_visualization_code(self) -> str:
        """
        生成Blender可视化代码
        
        Returns:
            Python代码字符串
        """
        # 转换为JSON格式
        import json
        contacts_json = json.dumps({
            str(k): v for k, v in self.frame_contacts.items()
        }, indent=4)
        
        code = f'''
# ============================================================================
# 接触点可视化数据
# ============================================================================

CONTACT_DATA = {contacts_json}


def visualize_contacts():
    """可视化所有接触点"""
    import bpy
    from mathutils import Vector
    
    # 创建材质
    def create_contact_material(name, color):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.inputs['Base Color'].default_value = (*color, 1.0)
            bsdf.inputs['Emission'].default_value = (*color, 1.0)
            bsdf.inputs['Emission Strength'].default_value = 2.0
            
            output = nodes.new(type='ShaderNodeOutputMaterial')
            mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        return mat
    
    # 创建接触点球体
    print(f"\\n创建接触点可视化...")
    contact_count = 0
    
    for frame_str, contacts in CONTACT_DATA.items():
        frame = int(frame_str)
        
        for i, contact in enumerate(contacts):
            point = contact['point']
            normal = contact['normal']
            penetration = contact['penetration']
            normal_vel = contact['normal_velocity']
            
            # 创建球体表示接触点
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=0.05,
                location=Vector([point[0], point[2], -point[1]])  # Y->Z转换
            )
            sphere = bpy.context.active_object
            sphere.name = f"Contact_F{{frame}}_{{i}}"
            
            # 根据穿透深度设置颜色
            if penetration < 0:
                # 负穿透（错误）- 红色
                color = (1.0, 0.0, 0.0)
            elif penetration < 0.001:
                # 浅接触 - 黄色
                color = (1.0, 1.0, 0.0)
            elif penetration < 0.01:
                # 中等接触 - 橙色
                color = (1.0, 0.5, 0.0)
            else:
                # 深度接触 - 红色
                color = (1.0, 0.0, 0.0)
            
            mat = create_contact_material(f"Contact_{{color}}", color)
            if sphere.data.materials:
                sphere.data.materials[0] = mat
            else:
                sphere.data.materials.append(mat)
            
            # 创建箭头表示法线
            # 法线起点
            start = Vector([point[0], point[2], -point[1]])
            # 法线终点（Y->Z转换）
            normal_zup = Vector([normal[0], normal[2], -normal[1]])
            end = start + normal_zup * 0.3
            
            # 创建圆柱体作为箭头
            direction = end - start
            length = direction.length
            if length > 0.01:
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.01,
                    depth=length,
                    location=(start + end) / 2
                )
                arrow = bpy.context.active_object
                arrow.name = f"Normal_F{{frame}}_{{i}}"
                
                # 旋转箭头对齐法线方向
                direction.normalize()
                arrow.rotation_mode = 'QUATERNION'
                arrow.rotation_quaternion = direction.to_track_quat('Z', 'Y')
                
                # 法线颜色（青色）
                normal_mat = create_contact_material("Normal_Material", (0.0, 1.0, 1.0))
                if arrow.data.materials:
                    arrow.data.materials[0] = normal_mat
                else:
                    arrow.data.materials.append(normal_mat)
                
                # 设置可见性动画（仅在对应帧可见）
                sphere.hide_viewport = True
                sphere.hide_render = True
                sphere.keyframe_insert(data_path="hide_viewport", frame=0)
                sphere.keyframe_insert(data_path="hide_render", frame=0)
                
                arrow.hide_viewport = True
                arrow.hide_render = True
                arrow.keyframe_insert(data_path="hide_viewport", frame=0)
                arrow.keyframe_insert(data_path="hide_render", frame=0)
                
                # 在对应帧显示
                sphere.hide_viewport = False
                sphere.hide_render = False
                sphere.keyframe_insert(data_path="hide_viewport", frame=frame)
                sphere.keyframe_insert(data_path="hide_render", frame=frame)
                
                arrow.hide_viewport = False
                arrow.hide_render = False
                arrow.keyframe_insert(data_path="hide_viewport", frame=frame)
                arrow.keyframe_insert(data_path="hide_render", frame=frame)
                
                # 下一帧隐藏
                if frame < max(int(f) for f in CONTACT_DATA.keys()):
                    sphere.hide_viewport = True
                    sphere.hide_render = True
                    sphere.keyframe_insert(data_path="hide_viewport", frame=frame+1)
                    sphere.keyframe_insert(data_path="hide_render", frame=frame+1)
                    
                    arrow.hide_viewport = True
                    arrow.hide_render = True
                    arrow.keyframe_insert(data_path="hide_viewport", frame=frame+1)
                    arrow.keyframe_insert(data_path="hide_render", frame=frame+1)
            
            contact_count += 1
            
            # 添加文本显示穿透深度
            bpy.ops.object.text_add(location=start + Vector([0, 0, 0.1]))
            text_obj = bpy.context.active_object
            text_obj.name = f"PenText_F{{frame}}_{{i}}"
            text_obj.data.body = f"{{penetration:.4f}}\\nv={{normal_vel:.2f}}"
            text_obj.data.size = 0.05
            text_obj.data.align_x = 'CENTER'
            
            # 文本可见性
            text_obj.hide_viewport = True
            text_obj.hide_render = True
            text_obj.keyframe_insert(data_path="hide_viewport", frame=0)
            text_obj.keyframe_insert(data_path="hide_render", frame=0)
            
            text_obj.hide_viewport = False
            text_obj.hide_render = False
            text_obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            text_obj.keyframe_insert(data_path="hide_render", frame=frame)
            
            if frame < max(int(f) for f in CONTACT_DATA.keys()):
                text_obj.hide_viewport = True
                text_obj.hide_render = True
                text_obj.keyframe_insert(data_path="hide_viewport", frame=frame+1)
                text_obj.keyframe_insert(data_path="hide_render", frame=frame+1)
    
    print(f"✓ 创建了 {{contact_count}} 个接触点可视化")
    print("\\n接触点图例:")
    print("  🔴 红色球体: 深度穿透或负穿透（错误）")
    print("  🟠 橙色球体: 中等穿透")
    print("  🟡 黄色球体: 浅接触")
    print("  🔵 青色箭头: 接触法线方向")
    print("  📝 文本: 穿透深度和法线速度")


# 在main函数最后调用
# visualize_contacts()
'''
        return code
    
    def print_contact_summary(self):
        """打印接触点统计摘要"""
        if not self.contact_history:
            print("无接触点记录")
            return
        
        print("\n" + "="*60)
        print("接触点调试摘要")
        print("="*60)
        
        penetrations = [c['penetration'] for c in self.contact_history]
        normal_vels = [c['normal_velocity'] for c in self.contact_history]
        
        print(f"总接触数: {len(self.contact_history)}")
        print(f"帧数: {len(self.frame_contacts)}")
        print(f"\n穿透深度统计:")
        print(f"  最小: {min(penetrations):.6f} m")
        print(f"  最大: {max(penetrations):.6f} m")
        print(f"  平均: {np.mean(penetrations):.6f} m")
        print(f"  负值数量: {sum(1 for p in penetrations if p < 0)}")
        
        print(f"\n法线速度统计:")
        print(f"  最小: {min(normal_vels):.3f} m/s")
        print(f"  最大: {max(normal_vels):.3f} m/s")
        print(f"  平均: {np.mean(normal_vels):.3f} m/s")
        print(f"  负值数量（分离）: {sum(1 for v in normal_vels if v < 0)}")
        
        # 检查异常
        print(f"\n⚠️  潜在问题:")
        if any(p < 0 for p in penetrations):
            print(f"  • 发现负穿透深度！")
        if any(abs(v) > 50 for v in normal_vels):
            print(f"  • 发现异常高速度！")
        
        print("="*60)
