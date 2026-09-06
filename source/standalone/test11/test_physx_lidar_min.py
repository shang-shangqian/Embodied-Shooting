import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni
import omni.kit.app
import omni.kit.commands

from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage


def enable_lidar_extensions():
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    for ext_name in [
        "isaacsim.sensors.physx",
        "omni.isaac.range_sensor",
    ]:
        try:
            if not ext_mgr.is_extension_enabled(ext_name):
                ext_mgr.set_extension_enabled_immediate(ext_name, True)
                print(f"[ext] enabled: {ext_name}")
            else:
                print(f"[ext] already enabled: {ext_name}")
        except Exception as e:
            print(f"[ext] enable failed: {ext_name} -> {e}")


def acquire_lidar_interface():
    try:
        import isaacsim.sensors.physx._range_sensor as rs
        return rs.acquire_lidar_sensor_interface()
    except Exception:
        import omni.isaac.range_sensor._range_sensor as rs
        return rs.acquire_lidar_sensor_interface()


def main():
    enable_lidar_extensions()

    world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()

    lidar_path = "/Lidar0"

    ok, prim = omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path=lidar_path,
        parent="/World",
        min_range=0.4,
        max_range=20.0,
        draw_points=False,
        draw_lines=True,
        horizontal_fov=360.0,
        vertical_fov=30.0,
        horizontal_resolution=1.0,
        vertical_resolution=4.0,
        rotation_rate=0.0,
        high_lod=False,
        yaw_offset=0.0,
        enable_semantics=False,
    )

    print(f"[lidar create] ok={ok}, prim={prim}, lidar_path={lidar_path}")

    lidar_iface = acquire_lidar_interface()
    print("[lidar iface]", lidar_iface)

    stage_prim = stage.GetPrimAtPath("/World/Lidar0")
    print("[stage prim exists]", stage_prim.IsValid(), stage_prim.GetTypeName() if stage_prim.IsValid() else None)

    # 先启动仿真并推进几帧，让 LiDAR 插件完成注册
    world.reset()
    for i in range(5):
        world.step(render=True)
        print(f"[warmup step] {i+1}")

    # 再检查是否成为有效传感器
    for p in ["/World/Lidar0", "/Lidar0"]:
        try:
            valid = lidar_iface.is_lidar_sensor(p)
            print(f"[lidar valid check] {p} -> {valid}")
        except Exception as e:
            print(f"[lidar valid check] {p} exception: {e}")

    # 再试着读取几项基础数据
    try:
        rows = int(lidar_iface.get_num_rows("/World/Lidar0"))
        cols = int(lidar_iface.get_num_cols("/World/Lidar0"))
        print(f"[lidar shape] rows={rows}, cols={cols}")
    except Exception as e:
        print("[lidar shape] /World/Lidar0 exception:", e)

    try:
        rows = int(lidar_iface.get_num_rows("/Lidar0"))
        cols = int(lidar_iface.get_num_cols("/Lidar0"))
        print(f"[lidar shape alt] rows={rows}, cols={cols}")
    except Exception as e:
        print("[lidar shape alt] /Lidar0 exception:", e)

    simulation_app.close()


if __name__ == "__main__":
    main()