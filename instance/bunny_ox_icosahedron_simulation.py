# bunny hits icosahedron simulation

import numpy as np
from simulation_export import run_simulation_export
from physics_engine.config.settings import config

if __name__ == "__main__":
    title = "bunny_ox_icosahedron"
    description = "Bunny_Ox_Hits_Icosahedron"

    object_physics_file = ["bunny_200.obj", "cow_simplified_5000.obj", "icosahedron_subdivided_1.obj"]
    object_render_file = ["bunny_200_subdivided_3.obj", "cow.obj", "icosahedron_subdivided_4.obj"]
    object_scale_factors = [2.0, 0.04, 1.0]

    is_static_list = [False, False, True]
    position_list = [np.array([0.0, 1.234, 0.0]), np.array([0.0, 3.123, 0.0]), np.array([0.0, 0.0, 0.0])]
    orientation_list = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])]
    density_list = [800.0, 800.0, 1000.0]
    restitution_list = [0.8, 0.8, 0.8]
    friction_list = [0.3, 0.3, 0.3]
    velocity_list = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), None]
    angular_velocity_list = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), None]

    duration = 3.33
    fps = 240.0
    gravity=config.physics.GRAVITY_EARTH

    enable_warmstarting=True

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

        duration=duration,
        fps=fps,
        gravity=gravity,

        enable_warmstarting=enable_warmstarting,
    )