# sphere collision simulation

import numpy as np
from simulation_export import run_simulation_export
from physics_engine.config.settings import config

if __name__ == "__main__":
    r = 1.34213
    orient = np.array([1.0, 0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0, 0.0])
    init_ang_vel = np.array([0.0, 0.0, 0.0])
    file = "sphere_subdivided_4.obj"

    title = "sphere_collision"
    description = "Sphere_Collision"

    object_physics_file = [file, file, file, file, file, file, file]
    object_render_file = [file, file, file, file, file, file, file]
    object_scale_factors = [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 1.0]
    is_static_list = [False, False, False, False, False, False, True]
    position_list = [np.array([0.0, 4.0, 0.3]), np.array([0.0, 6.0, 0.2]), np.array([0.3, 2.5, -0.2]),
                     np.array([0.5, 5.0, -0.9]), np.array([1.4, 4.5, 0.0]), np.array([-0.4, 8.0, 0.0]),
                     np.array([0.0, 0.0, 0.0])]
    orientation_list = [orient, orient, orient, orient, orient, orient, orient]
    density_list = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    restitution_list = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    friction_list = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    velocity_list = [init_vel, init_vel, init_vel, init_vel, init_vel, init_vel, None]
    angular_velocity_list = [init_ang_vel, init_ang_vel, init_ang_vel, init_ang_vel, init_ang_vel, init_ang_vel, None]
    object_shape_list = ["sphere", "sphere", "sphere", "sphere", "sphere", "sphere", "sphere"]
    object_shape_params_list = [{"radius":r * 0.5}, {"radius":r * 0.5}, {"radius":r * 0.5},
                                {"radius":r * 0.3}, {"radius":r * 0.3}, {"radius":r * 0.3},
                                {"radius":r}]

    duration = 4
    fps = 120.0
    gravity=config.physics.GRAVITY_EARTH

    enable_warmstarting=False

    run_simulation_export(
        title=title,
        description=description,

        object_physics_file=object_physics_file,
        object_render_file=object_render_file,
        object_scale_factors=object_scale_factors,

        is_static_list=is_static_list,
        position_list=position_list,
        orientation_list=orientation_list,
        density_list=density_list,
        restitution_list=restitution_list,
        friction_list=friction_list,
        velocity_list=velocity_list,
        angular_velocity_list=angular_velocity_list,
        object_shape_list=object_shape_list,
        object_shape_params_list=object_shape_params_list,

        duration=duration,
        fps=fps,
        gravity=gravity,

        enable_warmstarting=enable_warmstarting,
    )