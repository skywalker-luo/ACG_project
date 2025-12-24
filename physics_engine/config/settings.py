"""
全局配置参数
定义物理引擎的默认参数和常量
"""

import numpy as np
from typing import Optional


# ============================================================================
# 物理常量
# ============================================================================

class PhysicsConstants:
    """物理常量"""
    
    # 重力加速度 (m/s²)
    GRAVITY_EARTH = np.array([0.0, -9.81, 0.0])  # 地球标准重力
    GRAVITY_MOON = np.array([0.0, -1.62, 0.0])   # 月球重力
    GRAVITY_MARS = np.array([0.0, -3.71, 0.0])   # 火星重力
    GRAVITY_NONE = np.array([0.0, 0.0, 0.0])     # 无重力
    
    # 默认密度
    DENSITY_MATERIAL = 1.0
    
    # 数值精度
    EPSILON = 1e-9          # 通用epsilon
    CONTACT_EPSILON = 1e-6  # 接触检测epsilon
    VELOCITY_EPSILON = 1e-8 # 速度判断epsilon


# ============================================================================
# 仿真参数
# ============================================================================

class SimulationSettings:
    """仿真设置"""
    
    # 时间步长
    DEFAULT_TIMESTEP = 1.0 / 60.0       # 默认时间步 (60 Hz)
    FIXED_TIMESTEP = 1.0 / 240.0        # 固定时间步 (240 Hz)
    MIN_TIMESTEP = 1.0 / 1000.0         # 最小时间步 (1 ms)
    MAX_TIMESTEP = 1.0 / 30.0           # 最大时间步 (33 ms)
    
    # 子步控制
    MAX_SUBSTEPS = 10                   # 最大子步数
    MIN_SUBSTEPS = 1                    # 最小子步数
    
    # 求解器迭代
    SOLVER_ITERATIONS = 10              # 约束求解器迭代次数
    POSITION_ITERATIONS = 3             # 位置校正迭代次数
    VELOCITY_ITERATIONS = 8             # 速度求解迭代次数


# ============================================================================
# 碰撞检测参数
# ============================================================================

class CollisionSettings:
    """碰撞检测设置"""
    
    # 宽相检测
    BROADPHASE_TYPE = 'simple'          # 'simple' 或 'spatial_hash'
    SPATIAL_HASH_CELL_SIZE = 2.0        # 空间哈希网格大小
    
    # 窄相检测
    NARROWPHASE_TYPE = 'moller'         # 三角形-三角形检测算法
    TRIANGLE_EPSILON = 1e-9             # 三角形相交检测容差
    
    # 接触点
    CONTACT_BREAKING_THRESHOLD = 0.02   # 接触点断开阈值 (m)
    CONTACT_PROCESSING_THRESHOLD = 0.0  # 接触点处理阈值
    MAX_CONTACTS_PER_PAIR = 4           # 每对物体的最大接触点数


# ============================================================================
# 约束求解参数
# ============================================================================

class ConstraintSettings:
    """约束求解设置"""
    
    # 求解器类型
    SOLVER_TYPE = 'impulse'             # 'basic' 或 'impulse'
    
    # Sequential Impulse 参数
    ENABLE_WARM_STARTING = True         # 启用热启动
    WARM_START_DECAY = 0.8              # 热启动衰减因子
    
    # 约束参数
    BAUMGARTE_FACTOR = 0.3              # Baumgarte稳定化因子 (0.1 - 0.3)
    RESTITUTION_THRESHOLD = 0.1         # 恢复系数阈值 (m/s) - 降低以允许低速反弹
    VELOCITY_THRESHOLD = 0.01           # 速度阈值，低于此值视为静止
    
    # 摩擦
    ENABLE_FRICTION = False             # 启用摩擦（暂未实现）
    FRICTION_ITERATIONS = 2             # 摩擦迭代次数
    
    # 数值稳定性
    IMPULSE_CLAMP = 1e6                 # 冲量钳位上限
    MIN_EFFECTIVE_MASS = 1e-10          # 最小有效质量


# ============================================================================
# 材质属性
# ============================================================================

class MaterialSettings:
    """材质默认属性"""
    
    # 恢复系数 (弹性)
    RESTITUTION_RUBBER = 0.9            # 橡胶
    RESTITUTION_WOOD = 0.4              # 木头
    RESTITUTION_METAL = 0.5             # 金属
    RESTITUTION_CONCRETE = 0.3          # 混凝土
    RESTITUTION_GLASS = 0.7             # 玻璃
    RESTITUTION_DEFAULT = 0.5           # 默认
    
    # 摩擦系数
    FRICTION_ICE = 0.05                 # 冰
    FRICTION_METAL = 0.3                # 金属
    FRICTION_WOOD = 0.4                 # 木头
    FRICTION_RUBBER = 0.8               # 橡胶
    FRICTION_DEFAULT = 0.5              # 默认


# ============================================================================
# 刚体参数
# ============================================================================

class RigidBodySettings:
    """刚体设置"""
    
    # 默认属性
    DEFAULT_DENSITY = 1.0            # 默认密度
    DEFAULT_RESTITUTION = 0.5           # 默认恢复系数
    DEFAULT_FRICTION = 0.5              # 默认摩擦系数
    
    # 数值限制
    MIN_MASS = 1e-6                     # 最小质量
    MAX_MASS = 1e10                     # 最大质量
    MIN_INERTIA = 1e-10                 # 最小惯性
    
    # 休眠阈值
    SLEEP_LINEAR_THRESHOLD = 0.8        # 线性速度休眠阈值 (m/s)
    SLEEP_ANGULAR_THRESHOLD = 1.0       # 角速度休眠阈值 (rad/s)
    SLEEP_TIME = 0.5                    # 需要保持低速的时间 (s)
    
    # 阻尼
    LINEAR_DAMPING = 0.0                # 线性阻尼系数
    ANGULAR_DAMPING = 0.0               # 角阻尼系数


# ============================================================================
# 积分器参数
# ============================================================================

class IntegratorSettings:
    """积分器设置"""
    
    # 积分方法
    METHOD = 'semi-implicit'            # 'explicit', 'semi-implicit', 'implicit'
    
    # 数值稳定性
    MAX_LINEAR_VELOCITY = 500.0         # 最大线性速度 (m/s)
    MAX_ANGULAR_VELOCITY = 50.0         # 最大角速度 (rad/s)
    MAX_LINEAR_ACCELERATION = 1000.0    # 最大线性加速度 (m/s²)
    MAX_ANGULAR_ACCELERATION = 500.0    # 最大角加速度 (rad/s²)


# ============================================================================
# 可视化参数
# ============================================================================

class VisualizationSettings:
    """可视化设置"""
    
    # 窗口
    DEFAULT_WIDTH = 1280                # 默认窗口宽度
    DEFAULT_HEIGHT = 720                # 默认窗口高度
    WINDOW_NAME = "Physics Simulation"  # 默认窗口标题
    
    # 渲染
    BACKGROUND_COLOR = (0.1, 0.1, 0.1)  # 背景颜色 (R, G, B)
    SHOW_AXES = True                    # 显示坐标轴
    SHOW_WIREFRAME = False              # 显示线框
    COORDINATE_FRAME_SIZE = 1.0         # 坐标系大小
    
    # 颜色
    STATIC_BODY_COLOR = (0.5, 0.5, 0.5) # 静态物体颜色（灰色）
    DYNAMIC_BODY_ALPHA = 1.0            # 动态物体透明度
    
    # 相机
    DEFAULT_CAMERA_FRONT = [0.5, -0.3, -0.8]
    DEFAULT_CAMERA_LOOKAT = [0, 5, 0]
    DEFAULT_CAMERA_UP = [0, 1, 0]
    DEFAULT_CAMERA_ZOOM = 0.5


# ============================================================================
# 调试和性能参数
# ============================================================================

class DebugSettings:
    """调试设置"""
    
    # 调试输出
    ENABLE_DEBUG = False                # 启用调试输出
    DEBUG_LEVEL = 1                     # 调试级别 (0-3)
    PRINT_INTERVAL = 60                 # 打印间隔（帧数）
    
    # 性能分析
    ENABLE_PROFILING = False            # 启用性能分析
    PROFILE_INTERVAL = 60               # 性能分析间隔（帧数）
    
    # 统计信息
    TRACK_STATISTICS = True             # 跟踪统计信息
    STATISTICS_WINDOW = 60              # 统计窗口大小（帧数）


# ============================================================================
# 配置类（整合所有设置）
# ============================================================================

class Config:
    """
    全局配置类
    整合所有配置参数，提供便捷的访问接口
    """
    
    def __init__(self):
        """初始化配置"""
        # 引用各个配置类
        self.physics = PhysicsConstants()
        self.simulation = SimulationSettings()
        self.collision = CollisionSettings()
        self.constraint = ConstraintSettings()
        self.material = MaterialSettings()
        self.rigid_body = RigidBodySettings()
        self.integrator = IntegratorSettings()
        self.visualization = VisualizationSettings()
        self.debug = DebugSettings()
    
    def reset_to_defaults(self):
        """重置所有配置到默认值"""
        self.__init__()
    
    def get_world_settings(self) -> dict:
        """
        获取World初始化所需的设置
        
        Returns:
            包含World参数的字典
        """
        return {
            'gravity': self.physics.GRAVITY_EARTH,
            'fixed_timestep': self.simulation.FIXED_TIMESTEP,
            'max_substeps': self.simulation.MAX_SUBSTEPS,
            'broadphase_type': self.collision.BROADPHASE_TYPE,
            'solver_type': self.constraint.SOLVER_TYPE,
            'debug': self.debug.ENABLE_DEBUG,
        }
    
    def get_solver_settings(self) -> dict:
        """
        获取求解器设置
        
        Returns:
            包含求解器参数的字典
        """
        return {
            'max_iterations': self.simulation.SOLVER_ITERATIONS,
            'restitution_threshold': self.constraint.RESTITUTION_THRESHOLD,
            'baumgarte_factor': self.constraint.BAUMGARTE_FACTOR,
        }
    
    def get_visualizer_settings(self) -> dict:
        """
        获取可视化器设置
        
        Returns:
            包含可视化参数的字典
        """
        return {
            'window_name': self.visualization.WINDOW_NAME,
            'width': self.visualization.DEFAULT_WIDTH,
            'height': self.visualization.DEFAULT_HEIGHT,
            'show_axes': self.visualization.SHOW_AXES,
            'show_wireframe': self.visualization.SHOW_WIREFRAME,
            'background_color': self.visualization.BACKGROUND_COLOR,
        }
    
    def print_summary(self):
        """打印配置摘要"""
        print("="*60)
        print("物理引擎配置摘要")
        print("="*60)
        print(f"\n【仿真参数】")
        print(f"  时间步长: {self.simulation.DEFAULT_TIMESTEP*1000:.2f}ms")
        print(f"  固定时间步: {self.simulation.FIXED_TIMESTEP*1000:.2f}ms")
        print(f"  最大子步数: {self.simulation.MAX_SUBSTEPS}")
        print(f"  求解器迭代: {self.simulation.SOLVER_ITERATIONS}")
        
        print(f"\n【物理参数】")
        print(f"  重力: {self.physics.GRAVITY_EARTH}")
        print(f"  默认密度: {self.rigid_body.DEFAULT_DENSITY} kg/m³")
        
        print(f"\n【碰撞检测】")
        print(f"  宽相类型: {self.collision.BROADPHASE_TYPE}")
        print(f"  窄相类型: {self.collision.NARROWPHASE_TYPE}")
        
        print(f"\n【约束求解】")
        print(f"  求解器类型: {self.constraint.SOLVER_TYPE}")
        print(f"  热启动: {'启用' if self.constraint.ENABLE_WARM_STARTING else '禁用'}")
        print(f"  Baumgarte因子: {self.constraint.BAUMGARTE_FACTOR}")
        
        print(f"\n【材质】")
        print(f"  默认恢复系数: {self.material.RESTITUTION_DEFAULT}")
        print(f"  默认摩擦系数: {self.material.FRICTION_DEFAULT}")
        
        print(f"\n【调试】")
        print(f"  调试模式: {'启用' if self.debug.ENABLE_DEBUG else '禁用'}")
        print(f"  性能分析: {'启用' if self.debug.ENABLE_PROFILING else '禁用'}")
        print("="*60)


# ============================================================================
# 全局配置实例
# ============================================================================

# 创建全局配置实例
config = Config()


# ============================================================================
# 便捷函数
# ============================================================================

def get_config() -> Config:
    """
    获取全局配置实例
    
    Returns:
        全局Config实例
    """
    return config


def reset_config():
    """重置全局配置到默认值"""
    global config
    config = Config()


def set_gravity(gravity: np.ndarray):
    """
    设置全局重力
    
    Args:
        gravity: 重力向量
    """
    config.physics.GRAVITY_EARTH = np.array(gravity)


def set_timestep(timestep: float):
    """
    设置全局时间步长
    
    Args:
        timestep: 时间步长（秒）
    """
    if timestep < config.simulation.MIN_TIMESTEP:
        print(f"警告: 时间步长 {timestep} 小于最小值，已调整为 {config.simulation.MIN_TIMESTEP}")
        timestep = config.simulation.MIN_TIMESTEP
    elif timestep > config.simulation.MAX_TIMESTEP:
        print(f"警告: 时间步长 {timestep} 大于最大值，已调整为 {config.simulation.MAX_TIMESTEP}")
        timestep = config.simulation.MAX_TIMESTEP
    
    config.simulation.DEFAULT_TIMESTEP = timestep


def enable_debug(enable: bool = True):
    """
    启用/禁用调试模式
    
    Args:
        enable: 是否启用
    """
    config.debug.ENABLE_DEBUG = enable


def enable_profiling(enable: bool = True):
    """
    启用/禁用性能分析
    
    Args:
        enable: 是否启用
    """
    config.debug.ENABLE_PROFILING = enable


# ============================================================================
# 预设配置
# ============================================================================

class Presets:
    """预设配置"""
    
    @staticmethod
    def high_accuracy():
        """高精度配置（慢但准确）"""
        config.simulation.FIXED_TIMESTEP = 1.0 / 480.0
        config.simulation.MAX_SUBSTEPS = 20
        config.simulation.SOLVER_ITERATIONS = 20
        config.constraint.BAUMGARTE_FACTOR = 0.3
        print("✅ 已切换到高精度配置")
    
    @staticmethod
    def performance():
        """性能优化配置（快但可能不够精确）"""
        config.simulation.FIXED_TIMESTEP = 1.0 / 120.0
        config.simulation.MAX_SUBSTEPS = 5
        config.simulation.SOLVER_ITERATIONS = 5
        config.constraint.BAUMGARTE_FACTOR = 0.3
        print("✅ 已切换到性能优化配置")
    
    @staticmethod
    def balanced():
        """平衡配置（默认）"""
        reset_config()
        print("✅ 已切换到平衡配置")
    
    @staticmethod
    def zero_gravity():
        """零重力配置"""
        config.physics.GRAVITY_EARTH = PhysicsConstants.GRAVITY_NONE
        print("✅ 已设置为零重力")
    
    @staticmethod
    def moon_gravity():
        """月球重力配置"""
        config.physics.GRAVITY_EARTH = PhysicsConstants.GRAVITY_MOON
        print("✅ 已设置为月球重力")


# ============================================================================
# 模块初始化
# ============================================================================

if __name__ == "__main__":
    # 如果直接运行此文件，打印配置摘要
    config.print_summary()
    
    print("\n测试预设配置:")
    print("\n1. 高精度模式:")
    Presets.high_accuracy()
    
    print("\n2. 性能模式:")
    Presets.performance()
    
    print("\n3. 恢复默认:")
    Presets.balanced()
