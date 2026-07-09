from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


LITE3_ROOT = "/Lite3"
TORSO_PATH = f"{LITE3_ROOT}/TORSO"
JOINTS_PATH = f"{LITE3_ROOT}/joints"

YAW_LINK_PATH = f"{LITE3_ROOT}/yaw_link"
PITCH_LINK_PATH = f"{LITE3_ROOT}/pitch_link"
YAW_JOINT_PATH = f"{JOINTS_PATH}/yaw_joint"
PITCH_JOINT_PATH = f"{JOINTS_PATH}/pitch_joint"

SOURCE_CAMERA_XFORM_PATH = f"{TORSO_PATH}/SG2_OX03CC_5200_GMSL2_H60YA"
SOURCE_CAMERA_PATH = f"{SOURCE_CAMERA_XFORM_PATH}/Camera_SG2_OX03CC_5200_GMSL2_H60YA"

def _set_asset_root(lite3_root: str) -> None:
    """Rebase all authored launcher paths to the Lite3 root found in the USD."""
    global LITE3_ROOT
    global TORSO_PATH
    global JOINTS_PATH
    global YAW_LINK_PATH
    global PITCH_LINK_PATH
    global YAW_JOINT_PATH
    global PITCH_JOINT_PATH
    global SOURCE_CAMERA_XFORM_PATH
    global SOURCE_CAMERA_PATH

    root = str(lite3_root).rstrip("/")
    if not root.startswith("/"):
        root = "/" + root

    LITE3_ROOT = root
    TORSO_PATH = f"{LITE3_ROOT}/TORSO"
    JOINTS_PATH = f"{LITE3_ROOT}/joints"
    YAW_LINK_PATH = f"{LITE3_ROOT}/yaw_link"
    PITCH_LINK_PATH = f"{LITE3_ROOT}/pitch_link"
    YAW_JOINT_PATH = f"{JOINTS_PATH}/yaw_joint"
    PITCH_JOINT_PATH = f"{JOINTS_PATH}/pitch_joint"
    SOURCE_CAMERA_XFORM_PATH = f"{TORSO_PATH}/SG2_OX03CC_5200_GMSL2_H60YA"
    SOURCE_CAMERA_PATH = f"{SOURCE_CAMERA_XFORM_PATH}/Camera_SG2_OX03CC_5200_GMSL2_H60YA"


def _find_lite3_root(stage: Usd.Stage, requested_root: str | None = None) -> str:
    """Return the parent path of the Lite3 TORSO prim."""
    candidates: list[str] = []
    if requested_root is not None:
        root = str(requested_root).rstrip("/")
        if not root.startswith("/"):
            root = "/" + root
        candidates.append(f"{root}/TORSO")

    candidates.extend(
        [
            TORSO_PATH,
            "/Lite3/Lite3/TORSO",
            "/World/Lite3/TORSO",
            "/World/Lite3/Lite3/TORSO",
            "/World/Environment/Lite3/TORSO",
            "/World/Environment/Lite3/Lite3/TORSO",
        ]
    )

    for candidate in candidates:
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid():
            return str(prim.GetPath().GetParentPath())

    matches = []
    for prim in stage.Traverse():
        if prim.GetName() == "TORSO":
            path = prim.GetPath().pathString
            lower_path = path.lower()
            score = path.count("/")
            if "lite3" in lower_path:
                score -= 10
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                score -= 4
            matches.append((score, path))

    if matches:
        matches.sort()
        return str(Sdf.Path(matches[0][1]).GetParentPath())

    root_prims = [prim.GetPath().pathString for prim in stage.GetPseudoRoot().GetChildren()]
    raise RuntimeError(
        "Could not find a TORSO prim in the USD. "
        f"Top-level prims are: {root_prims}. "
        "Pass --lite3-root /path/to/the/Lite3/root if your asset uses a custom hierarchy."
    )

def _vec3d(values: tuple[float, float, float]) -> Gf.Vec3d:
    return Gf.Vec3d(float(values[0]), float(values[1]), float(values[2]))


def _vec3f(values: tuple[float, float, float]) -> Gf.Vec3f:
    return Gf.Vec3f(float(values[0]), float(values[1]), float(values[2]))


def _quatd(values: tuple[float, float, float, float]) -> Gf.Quatd:
    return Gf.Quatd(float(values[0]), _vec3d((values[1], values[2], values[3])))


def _quatf_identity() -> Gf.Quatf:
    return Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))


def _set_xform(
    prim: Usd.Prim,
    translate: tuple[float, float, float],
    orient_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(_vec3d(translate))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_quatd(orient_wxyz))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(_vec3f(scale))


def _set_color(prim: Usd.Prim, color: tuple[float, float, float]) -> None:
    if prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([_vec3f(color)])


def _apply_collision(prim: Usd.Prim, enabled: bool) -> None:
    if enabled:
        UsdPhysics.CollisionAPI.Apply(prim)


def _apply_rigid_body(
    prim: Usd.Prim,
    mass_kg: float,
    diagonal_inertia: tuple[float, float, float],
) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass_kg))
    mass_api.CreateDiagonalInertiaAttr(_vec3f(diagonal_inertia))


def _define_xform(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    orient_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    prim.SetActive(True)
    _set_xform(prim, translate, orient_wxyz, scale)
    return prim


def _define_cube(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    prim.SetActive(True)
    _set_xform(prim, translate=translate, scale=scale)
    _set_color(prim, color)
    _apply_collision(prim, collision)
    return prim


def _define_cylinder(
    stage: Usd.Stage,
    path: str,
    axis: str,
    translate: tuple[float, float, float],
    radius: float,
    height: float,
    color: tuple[float, float, float],
    collision: bool = False,
) -> Usd.Prim:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(axis)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    prim = cylinder.GetPrim()
    prim.SetActive(True)
    _set_xform(prim, translate=translate)
    _set_color(prim, color)
    _apply_collision(prim, collision)
    return prim


def _define_sphere(
    stage: Usd.Stage,
    path: str,
    translate: tuple[float, float, float],
    radius: float,
    color: tuple[float, float, float],
) -> Usd.Prim:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    prim = sphere.GetPrim()
    prim.SetActive(True)
    _set_xform(prim, translate=translate)
    _set_color(prim, color)
    return prim


def _define_revolute_joint(
    stage: Usd.Stage,
    path: str,
    body0_path: str,
    body1_path: str,
    axis: str,
    local_pos0: tuple[float, float, float],
    local_pos1: tuple[float, float, float],
    lower_deg: float,
    upper_deg: float,
    stiffness: float,
    damping: float,
    max_force: float,
) -> None:
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(_vec3f(local_pos0))
    joint.CreateLocalPos1Attr(_vec3f(local_pos1))
    joint.CreateLocalRot0Attr(_quatf_identity())
    joint.CreateLocalRot1Attr(_quatf_identity())
    joint.CreateLowerLimitAttr(float(lower_deg))
    joint.CreateUpperLimitAttr(float(upper_deg))

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateMaxForceAttr(float(max_force))


def _remove_if_present(stage: Usd.Stage, path: str) -> None:
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(Sdf.Path(path))
        if stage.GetPrimAtPath(path).IsValid():
            stage.OverridePrim(Sdf.Path(path)).SetActive(False)


def _remove_previous_launcher(stage: Usd.Stage) -> None:
    paths = [
        YAW_JOINT_PATH,
        PITCH_JOINT_PATH,
        YAW_LINK_PATH,
        PITCH_LINK_PATH,
        # Common prim names from the previous test7_1 turret asset.
        f"{TORSO_PATH}/_turret_fixed_01",
        f"{TORSO_PATH}/base_link",
        f"{TORSO_PATH}/yaw",
        f"{TORSO_PATH}/pitch",
        f"{LITE3_ROOT}/base_link",
        f"{LITE3_ROOT}/yaw",
        f"{LITE3_ROOT}/pitch",
        f"{TORSO_PATH}/launcher_base_plate",
        f"{TORSO_PATH}/launcher_top_plate",
        f"{TORSO_PATH}/launcher_camera_pod",
        f"{TORSO_PATH}/launcher_camera_lens",
        f"{TORSO_PATH}/launcher_lidar_pod",
        f"{TORSO_PATH}/launcher_lidar_lens",
        f"{TORSO_PATH}/launcher_sensor_mast",
    ]
    for path in paths:
        _remove_if_present(stage, path)


def _ensure_source_camera(stage: Usd.Stage) -> None:
    if stage.GetPrimAtPath(SOURCE_CAMERA_PATH).IsValid():
        return

    _define_xform(
        stage,
        SOURCE_CAMERA_XFORM_PATH,
        translate=(0.23, 0.0, 0.073),
    )
    camera = UsdGeom.Camera.Define(stage, SOURCE_CAMERA_PATH)
    camera.GetProjectionAttr().Set("perspective")
    camera.GetFocalLengthAttr().Set(10.0)
    camera.GetHorizontalApertureAttr().Set(5.76)
    camera.GetVerticalApertureAttr().Set(3.24)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1000000.0))
    _set_xform(camera.GetPrim(), translate=(0.0, 0.0, 0.0))


def _build_visual_launcher(
    stage: Usd.Stage,
    mount_pos: tuple[float, float, float],
    collision: bool,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mount_x, mount_y, mount_z = mount_pos

    # The pitch link origin is the barrel center. With muzzle_offset_b=(0.09, 0, 0),
    # the physics muzzle point lands at the front tip of the black tube.
    pitch_pivot_in_yaw = (0.050, 0.0, 0.115)
    pitch_link_origin_in_yaw = (0.140, 0.0, 0.115)
    pitch_link_origin_in_root = (
        mount_x + pitch_link_origin_in_yaw[0],
        mount_y + pitch_link_origin_in_yaw[1],
        mount_z + pitch_link_origin_in_yaw[2],
    )

    _define_cube(
        stage,
        f"{TORSO_PATH}/launcher_base_plate",
        translate=(mount_x + 0.030, mount_y, mount_z - 0.018),
        scale=(0.230, 0.150, 0.028),
        color=(0.45, 0.47, 0.46),
        collision=False,
    )
    _define_cube(
        stage,
        f"{TORSO_PATH}/launcher_top_plate",
        translate=(mount_x + 0.030, mount_y, mount_z + 0.005),
        scale=(0.155, 0.105, 0.014),
        color=(0.72, 0.73, 0.70),
        collision=False,
    )

    yaw_link = _define_xform(stage, YAW_LINK_PATH, translate=mount_pos)
    _apply_rigid_body(
        yaw_link,
        mass_kg=0.22,
        diagonal_inertia=(0.00055, 0.00050, 0.00042),
    )

    pitch_link = _define_xform(stage, PITCH_LINK_PATH, translate=pitch_link_origin_in_root)
    _apply_rigid_body(
        pitch_link,
        mass_kg=0.18,
        diagonal_inertia=(0.00020, 0.00062, 0.00062),
    )

    # Rotating yaw stack.
    _define_cylinder(
        stage,
        f"{YAW_LINK_PATH}/black_slew_ring",
        axis="Z",
        translate=(0.0, 0.0, 0.010),
        radius=0.044,
        height=0.030,
        color=(0.02, 0.025, 0.025),
        collision=collision,
    )
    _define_cube(
        stage,
        f"{YAW_LINK_PATH}/white_lower_motor",
        translate=(0.0, 0.0, 0.055),
        scale=(0.055, 0.055, 0.060),
        color=(0.86, 0.85, 0.80),
        collision=collision,
    )
    _define_cylinder(
        stage,
        f"{YAW_LINK_PATH}/black_vertical_post",
        axis="Z",
        translate=(0.0, 0.0, 0.095),
        radius=0.020,
        height=0.085,
        color=(0.04, 0.045, 0.045),
        collision=collision,
    )
    _define_cube(
        stage,
        f"{YAW_LINK_PATH}/pitch_yoke_cross",
        translate=pitch_pivot_in_yaw,
        scale=(0.070, 0.155, 0.052),
        color=(0.80, 0.79, 0.73),
        collision=collision,
    )
    _define_cube(
        stage,
        f"{YAW_LINK_PATH}/left_yoke_plate",
        translate=(pitch_pivot_in_yaw[0], 0.076, pitch_pivot_in_yaw[2]),
        scale=(0.068, 0.012, 0.075),
        color=(0.08, 0.085, 0.085),
        collision=collision,
    )
    _define_cube(
        stage,
        f"{YAW_LINK_PATH}/right_yoke_plate",
        translate=(pitch_pivot_in_yaw[0], -0.076, pitch_pivot_in_yaw[2]),
        scale=(0.068, 0.012, 0.075),
        color=(0.08, 0.085, 0.085),
        collision=collision,
    )

    # Pitching launcher body, styled after the reference image.
    _define_cube(
        stage,
        f"{PITCH_LINK_PATH}/gray_launcher_body",
        translate=(-0.055, 0.0, 0.006),
        scale=(0.170, 0.060, 0.055),
        color=(0.57, 0.58, 0.55),
        collision=collision,
    )
    _define_cube(
        stage,
        f"{PITCH_LINK_PATH}/raised_rear_housing",
        translate=(-0.125, 0.0, 0.050),
        scale=(0.075, 0.064, 0.040),
        color=(0.62, 0.63, 0.60),
        collision=collision,
    )
    _define_cube(
        stage,
        f"{PITCH_LINK_PATH}/rear_tail",
        translate=(-0.190, 0.0, 0.040),
        scale=(0.055, 0.040, 0.030),
        color=(0.48, 0.50, 0.48),
        collision=collision,
    )
    _define_cylinder(
        stage,
        f"{PITCH_LINK_PATH}/main_launch_tube",
        axis="X",
        translate=(0.000, 0.0, 0.0),
        radius=0.016,
        height=0.180,
        color=(0.035, 0.038, 0.038),
        collision=collision,
    )
    _define_cylinder(
        stage,
        f"{PITCH_LINK_PATH}/front_muzzle_ring",
        axis="X",
        translate=(0.087, 0.0, 0.0),
        radius=0.020,
        height=0.014,
        color=(0.12, 0.12, 0.11),
        collision=collision,
    )
    _define_sphere(
        stage,
        f"{PITCH_LINK_PATH}/muzzle",
        translate=(0.090, 0.0, 0.0),
        radius=0.010,
        color=(0.95, 0.18, 0.05),
    )
    _define_cylinder(
        stage,
        f"{PITCH_LINK_PATH}/trunnion_axle",
        axis="Y",
        translate=(-0.090, 0.0, 0.0),
        radius=0.018,
        height=0.165,
        color=(0.10, 0.10, 0.095),
        collision=collision,
    )

    # Fixed sensor tower, visually echoing the green camera boxes in the image.
    _define_cube(
        stage,
        f"{TORSO_PATH}/launcher_sensor_mast",
        translate=(mount_x + 0.040, mount_y, mount_z + 0.175),
        scale=(0.040, 0.040, 0.115),
        color=(0.08, 0.085, 0.085),
        collision=False,
    )
    _define_cube(
        stage,
        f"{TORSO_PATH}/launcher_lidar_pod",
        translate=(mount_x + 0.070, mount_y, mount_z + 0.255),
        scale=(0.060, 0.070, 0.050),
        color=(0.02, 0.75, 0.18),
        collision=False,
    )
    _define_cylinder(
        stage,
        f"{TORSO_PATH}/launcher_lidar_lens",
        axis="X",
        translate=(mount_x + 0.105, mount_y, mount_z + 0.255),
        radius=0.018,
        height=0.016,
        color=(0.015, 0.018, 0.018),
        collision=False,
    )
    _define_cube(
        stage,
        f"{TORSO_PATH}/launcher_camera_pod",
        translate=(mount_x + 0.075, mount_y, mount_z + 0.320),
        scale=(0.070, 0.065, 0.045),
        color=(0.02, 0.70, 0.20),
        collision=False,
    )
    _define_cylinder(
        stage,
        f"{TORSO_PATH}/launcher_camera_lens",
        axis="X",
        translate=(mount_x + 0.115, mount_y, mount_z + 0.320),
        radius=0.019,
        height=0.018,
        color=(0.01, 0.012, 0.012),
        collision=False,
    )

    return pitch_pivot_in_yaw, pitch_link_origin_in_yaw


def create_launcher_model(
    input_usd: Path,
    output_usd: Path,
    mount_pos: tuple[float, float, float],
    collision: bool,
    backup: bool,
    lite3_root: str | None = None,
) -> None:
    if not input_usd.exists():
        raise FileNotFoundError(input_usd)

    if input_usd.resolve() == output_usd.resolve() and backup:
        backup_path = input_usd.with_suffix(input_usd.suffix + ".bak")
        shutil.copy2(input_usd, backup_path)
        print(f"[launcher] backup written: {backup_path}")

    stage = Usd.Stage.Open(str(input_usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {input_usd}")

    resolved_root = _find_lite3_root(stage, lite3_root)
    _set_asset_root(resolved_root)
    print(f"[launcher] Lite3 root = {LITE3_ROOT}")
    print(f"[launcher] torso prim = {TORSO_PATH}")

    torso = stage.GetPrimAtPath(TORSO_PATH)
    if not torso.IsValid():
        raise RuntimeError(f"Could not find torso prim after auto-detect: {TORSO_PATH}")

    UsdGeom.Xform.Define(stage, JOINTS_PATH)
    UsdPhysics.ArticulationRootAPI.Apply(torso)

    _remove_previous_launcher(stage)
    _ensure_source_camera(stage)
    pitch_pivot_in_yaw, pitch_link_origin_in_yaw = _build_visual_launcher(
        stage,
        mount_pos=mount_pos,
        collision=collision,
    )

    _define_revolute_joint(
        stage,
        path=YAW_JOINT_PATH,
        body0_path=TORSO_PATH,
        body1_path=YAW_LINK_PATH,
        axis="Z",
        local_pos0=mount_pos,
        local_pos1=(0.0, 0.0, 0.0),
        lower_deg=-180.0,
        upper_deg=180.0,
        stiffness=20.0,
        damping=2.0,
        max_force=50.0,
    )
    _define_revolute_joint(
        stage,
        path=PITCH_JOINT_PATH,
        body0_path=YAW_LINK_PATH,
        body1_path=PITCH_LINK_PATH,
        axis="Y",
        local_pos0=pitch_pivot_in_yaw,
        local_pos1=(
            pitch_pivot_in_yaw[0] - pitch_link_origin_in_yaw[0],
            pitch_pivot_in_yaw[1] - pitch_link_origin_in_yaw[1],
            pitch_pivot_in_yaw[2] - pitch_link_origin_in_yaw[2],
        ),
        lower_deg=-60.0,
        upper_deg=60.0,
        stiffness=20.0,
        damping=2.0,
        max_force=50.0,
    )

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output_usd)):
        raise RuntimeError(f"Could not export USD: {output_usd}")

    print(f"[launcher] wrote: {output_usd}")
    print("[launcher] compatible TurretAimEnvCfg values:")
    print("  yaw_joint_name = 'yaw_joint'")
    print("  pitch_joint_name = 'pitch_joint'")
    print("  muzzle_body_name_expr = 'pitch_link'")
    print("  muzzle_forward_axis = (1.0, 0.0, 0.0)")
    print("  muzzle_offset_b = (0.09, 0.0, 0.0)")


def _parse_xyz(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use x,y,z, for example: 0.02,0,0.105")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build a reference-image-style launcher on test7_1/lite3_with_turret.usd "
            "while preserving yaw_joint, pitch_joint, and pitch_link for TurretAimEnv."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "lite3_with_turret.usd",
        help="Input USD asset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "lite3_with_launcher.usd",
        help="Output USD asset. Pass the same path as --input to overwrite.",
    )
    parser.add_argument(
        "--lite3-root",
        type=str,
        default=None,
        help="Optional Lite3 root prim path. Example: /Lite3 or /World/Environment/Lite3.",
    )
    parser.add_argument(
        "--mount-pos",
        type=_parse_xyz,
        default=(0.02, 0.0, 0.105),
        help="Yaw-axis mount position in the TORSO frame as x,y,z.",
    )
    parser.add_argument(
        "--collisions",
        action="store_true",
        help="Add collision API to launcher visual geometry. Off by default to avoid self-collision surprises.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file when overwriting the input USD.",
    )
    args = parser.parse_args()

    create_launcher_model(
        input_usd=args.input,
        output_usd=args.output,
        mount_pos=args.mount_pos,
        collision=bool(args.collisions),
        backup=not bool(args.no_backup),
        lite3_root=args.lite3_root,
    )


if __name__ == "__main__":
    main()



