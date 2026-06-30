import torch
import os
import glob
import argparse
import math


def inspect_pt_file(file_path):
    print(f"\n{'='*60}")
    print(f"🔍 正在解析文件: {file_path}")
    print(f"{'='*60}")

    try:
        # 强制加载到 CPU 内存中
        data = torch.load(file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return

    # 1. 打印 Meta 字典 (元数据)
    if "meta" in data:
        print("📊 [Meta 信息]")
        for k, v in data["meta"].items():
            if isinstance(v, list):
                print(f"  - {k}:")
                for item in v:
                    print(f"      {item}")
            else:
                print(f"  - {k}: {v}")
        print()

    # 2. 打印特征序列基本信息
    seq_data = None
    target_vel = None
    if "sequence_data" in data:
        seq_data = data["sequence_data"]
        print("📈 [特征序列 - sequence_data]")
        print(f"  - Shape: {tuple(seq_data.shape)}  <-- [样本数, 帧数, 特征维度]")
        print(f"  - Dtype: {seq_data.dtype}\n")

    # 3. 打印速度标签信息
    if "target_velocity_base" in data:
        target_vel = data["target_velocity_base"]
        print("🎯 [速度标签 - target_velocity_base]")
        print(f"  - Shape: {tuple(target_vel.shape)}  <-- [样本数, 3维速度]")
        if target_vel.shape[0] > 0:
            sample_label = [round(x, 4) for x in target_vel[0].tolist()]
            print(f"  - 示例 (第1个样本): {sample_label}\n")

    # 4. 计算并打印误差 (坐标 + 角度)
    if seq_data is not None and seq_data.shape[0] > 0:
        # A. 计算 3D 坐标误差：GT vs Refined
        gt_pos = seq_data[:, :, 1:4]
        refined_pos = seq_data[:, :, 7:10]

        euclidean_error = torch.norm(refined_pos - gt_pos, dim=-1)
        signed_error_x = refined_pos[:, :, 0] - gt_pos[:, :, 0]
        abs_error_x = torch.abs(refined_pos[:, :, 0] - gt_pos[:, :, 0])
        abs_error_y = torch.abs(refined_pos[:, :, 1] - gt_pos[:, :, 1])
        abs_error_z = torch.abs(refined_pos[:, :, 2] - gt_pos[:, :, 2])

        print("📏 [空间坐标系误差分析 (GT vs Refined)]")
        print(f"  - 整体平均 3D 误差: {euclidean_error.mean().item():.6f} 米")
        print(f"  - 最大单点 3D 误差: {euclidean_error.max().item():.6f} 米")
        print(f"  - 平均 X 轴绝对误差: {abs_error_x.mean().item():.6f} 米")
        print(f"  - 平均 Y 轴绝对误差: {abs_error_y.mean().item():.6f} 米")
        print(f"  - 平均 Z 轴绝对误差: {abs_error_z.mean().item():.6f} 米\n")
        print(f"  - mean(refined_x - gt_x): {signed_error_x.mean().item():.6f} 米")
        print(f"  - std(refined_x - gt_x) : {signed_error_x.std(unbiased=False).item():.6f} 米\n")

        # B. 计算角度误差：GT vs Refined
        refined_yaw = refined_pitch = gt_yaw = gt_pitch = dog_moving_flag = None

        if seq_data.shape[2] >= 26:
            dog_moving_flag = seq_data[:, :, 16]

            refined_yaw = seq_data[:, :, 22]
            refined_pitch = seq_data[:, :, 23]
            gt_yaw = seq_data[:, :, 24]
            gt_pitch = seq_data[:, :, 25]
        else:
            print("⚠️ 当前脚本的角度索引按 26 维特征定义编写，旧格式文件请单独适配。")

        if refined_yaw is not None:
            def yaw_error(pred, gt):
                diff = (pred - gt + math.pi) % (2 * math.pi) - math.pi
                return torch.abs(diff)

            def pitch_error(pred, gt):
                return torch.abs(pred - gt)

            err_yaw_rad = yaw_error(refined_yaw, gt_yaw)
            err_pitch_rad = pitch_error(refined_pitch, gt_pitch)

            mean_yaw_rad = err_yaw_rad.mean().item()
            mean_pitch_rad = err_pitch_rad.mean().item()

            print("📐 [角度误差分析 (GT vs Refined)]")
            print(f"  - 平均 Yaw 绝对误差  : {mean_yaw_rad:.6f} rad  (约 {math.degrees(mean_yaw_rad):.3f} 度)")
            print(f"  - 平均 Pitch 绝对误差: {mean_pitch_rad:.6f} rad  (约 {math.degrees(mean_pitch_rad):.3f} 度)")
            print(f"  - 最大 Yaw 单点误差  : {math.degrees(err_yaw_rad.max().item()):.3f} 度")
            print(f"  - 最大 Pitch 单点误差: {math.degrees(err_pitch_rad.max().item()):.3f} 度\n")

        # C. 打印单帧示例对比
        print("🔍 [单帧直观对比展示 (第1个样本 第1帧)]")
        sample_gt = [round(x, 5) for x in gt_pos[0, 0].tolist()]
        sample_refined = [round(x, 5) for x in refined_pos[0, 0].tolist()]
        sample_err = round(euclidean_error[0, 0].item(), 5)

        sample_target_vel = None
        if target_vel is not None and target_vel.shape[0] > 0:
            sample_target_vel = [round(x, 5) for x in target_vel[0].tolist()]

        sample_frame_count = int(seq_data.shape[1])
        sample_first_dt = None
        sample_gt_first_step_vel = None
        sample_refined_first_step_vel = None
        sample_last_dt = None
        sample_gt_last_step_vel = None
        sample_refined_last_step_vel = None
        sample_gt_fit_vel = None
        sample_refined_fit_vel = None

        def _round_vec(tensor):
            return [round(float(x), 5) for x in tensor.tolist()]

        if seq_data.shape[1] >= 2:
            dt = (seq_data[0, 1, 0] - seq_data[0, 0, 0]).item()
            if abs(dt) > 1e-8:
                sample_first_dt = dt
                gt_step_vel = (gt_pos[0, 1] - gt_pos[0, 0]) / dt
                refined_step_vel = (refined_pos[0, 1] - refined_pos[0, 0]) / dt
                sample_gt_first_step_vel = _round_vec(gt_step_vel)
                sample_refined_first_step_vel = _round_vec(refined_step_vel)

            last_dt = (seq_data[0, -1, 0] - seq_data[0, -2, 0]).item()
            if abs(last_dt) > 1e-8:
                sample_last_dt = last_dt
                gt_last_step_vel = (gt_pos[0, -1] - gt_pos[0, -2]) / last_dt
                refined_last_step_vel = (refined_pos[0, -1] - refined_pos[0, -2]) / last_dt
                sample_gt_last_step_vel = _round_vec(gt_last_step_vel)
                sample_refined_last_step_vel = _round_vec(refined_last_step_vel)

            times = seq_data[0, :, 0].to(torch.float32)
            centered_t = times - times.mean()
            denom = torch.sum(centered_t * centered_t)
            if denom.item() > 1e-12:
                centered_gt = gt_pos[0] - gt_pos[0].mean(dim=0, keepdim=True)
                centered_refined = refined_pos[0] - refined_pos[0].mean(dim=0, keepdim=True)
                gt_fit_vel = torch.sum(centered_t.unsqueeze(-1) * centered_gt, dim=0) / denom
                refined_fit_vel = torch.sum(centered_t.unsqueeze(-1) * centered_refined, dim=0) / denom
                sample_gt_fit_vel = _round_vec(gt_fit_vel)
                sample_refined_fit_vel = _round_vec(refined_fit_vel)

        print("  📍 坐标对比:")
        print(f"    * GT 坐标      : {sample_gt}")
        print(f"    * Refined 坐标 : {sample_refined}")
        print(f"    * 单点距差     : {sample_err} 米")

        if sample_target_vel is not None:
            print("  🏃 速度对比:")
            print(f"    * 保存的速度标签 target_velocity_base (窗口末帧): {sample_target_vel} m/s")
        if sample_gt_first_step_vel is not None and sample_refined_first_step_vel is not None:
            print(f"    * 第1->2帧时间间隔 dt       : {sample_first_dt:.5f} s")
            print(f"    * GT 第1->2帧差分速度      : {sample_gt_first_step_vel} m/s")
            print(f"    * Refined 第1->2帧差分速度 : {sample_refined_first_step_vel} m/s")
        if sample_gt_last_step_vel is not None and sample_refined_last_step_vel is not None:
            print(f"    * 第{sample_frame_count - 1}->{sample_frame_count}帧时间间隔 dt       : {sample_last_dt:.5f} s")
            print(f"    * GT 第{sample_frame_count - 1}->{sample_frame_count}帧差分速度      : {sample_gt_last_step_vel} m/s")
            print(f"    * Refined 第{sample_frame_count - 1}->{sample_frame_count}帧差分速度 : {sample_refined_last_step_vel} m/s")
        if sample_gt_fit_vel is not None and sample_refined_fit_vel is not None:
            print(f"    * GT 全窗口线性拟合速度      : {sample_gt_fit_vel} m/s")
            print(f"    * Refined 全窗口线性拟合速度 : {sample_refined_fit_vel} m/s")

        if refined_yaw is not None:
            print("  🎯 角度对比:")
            print(f"    * GT 角度      : Yaw= {gt_yaw[0, 0].item():.5f}, Pitch= {gt_pitch[0, 0].item():.5f} (rad)")
            print(f"    * Refined 角度 : Yaw= {refined_yaw[0, 0].item():.5f}, Pitch= {refined_pitch[0, 0].item():.5f} (rad)")
            print(f"    * 角度偏离     : Yaw= {math.degrees(err_yaw_rad[0, 0].item()):.3f}度, Pitch= {math.degrees(err_pitch_rad[0, 0].item()):.3f}度")
            if dog_moving_flag is not None:
                print(f"  🐕 运动状态      : dog_moving_flag= {dog_moving_flag[0, 0].item():.0f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查保存的 .pt 数据文件")
    parser.add_argument(
        "--dir",
        type=str,
        default="source/standalone/test10/speed_prediction/velocity_dataset",
        help="存放 .pt 文件的目录路径",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="直接指定单个 .pt 文件的路径",
    )

    args = parser.parse_args()

    if args.file:
        if os.path.exists(args.file):
            inspect_pt_file(args.file)
        else:
            print(f"找不到文件: {args.file}")
    else:
        if not os.path.exists(args.dir):
            print(f"找不到目录: {args.dir}")
        else:
            pt_files = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
            if not pt_files:
                print(f"在目录 {args.dir} 下没有找到任何 .pt 文件。")
            else:
                print(f"找到 {len(pt_files)} 个 .pt 文件，开始逐个解析：")
                for idx, pt_file in enumerate(pt_files, 1):
                    print(f"\n\n========== [{idx}/{len(pt_files)}] ==========")
                    inspect_pt_file(pt_file)
