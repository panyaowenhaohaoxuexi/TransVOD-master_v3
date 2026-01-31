import os
import re
import shutil
from pathlib import Path

# ================= 配置区域 =================
# 你的输入根目录
INPUT_ROOT = r'/mnt/f/Tri_modal_data/Temp_Tri_modal_data/labels_val_txt'
# 输出根目录
OUTPUT_ROOT = r'/mnt/f/Tri_modal_data/Temp_Tri_modal_data/labels/val'
# "F:\Tri_modal_data\Temp_Tri_modal_data\labels_train_txt"
# F:\Tri_modal_data\Temp_Tri_modal_data\labels\train
# ===========================================

def distribute_images(phase_name, start_video_id, max_suffix, offset_count):
    """
    :param phase_name: 文件夹名称 (如 diyiqi)
    :param start_video_id: 这个文件夹对应的起始 video 编号 (如 1 或 6)
    :param max_suffix: 最大的后缀数字 (如 5 或 10)
    :param offset_count: 计算目标编号的偏移量 (diyiqi是0, dierqi是5)
    """
    phase_dir = Path(INPUT_ROOT) / phase_name
    
    if not phase_dir.exists():
        print(f"[错误] 文件夹不存在: {phase_dir}")
        return

    print(f"--- 正在处理: {phase_name} ---")
    
    # 准备好目标文件夹
    # diyiqi: video_0001 ~ video_0005
    # dierqi: video_0006 ~ video_0015
    targets = {}
    for i in range(1, max_suffix + 1):
        # 核心逻辑：目标编号 = 偏移量 + 当前后缀数字
        # 例如 dierqi 的 _1.jpg -> 5 + 1 = 6 -> video_0006
        target_num = offset_count + i
        target_dir = Path(OUTPUT_ROOT) / f"video_{target_num:04d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        targets[i] = target_dir

    # 正则匹配：任意字符 + 下划线 + 数字 + 后缀
    pattern = re.compile(r".*_(\d+)\.(jpg|png|jpeg|txt)$", re.IGNORECASE)

    copied_count = 0
    
    files = sorted([f for f in os.listdir(phase_dir) if os.path.isfile(phase_dir / f)])

    for name in files:
        m = pattern.match(name)
        if not m:
            continue # 跳过不符合规则的文件
        
        suffix_num = int(m.group(1)) # 拿到 _ 后面的数字

        # 如果数字超出了范围（比如 diyiqi 里出现了 _6），则跳过
        if suffix_num < 1 or suffix_num > max_suffix:
            continue

        src = phase_dir / name
        dst_folder = targets[suffix_num]
        dst = dst_folder / name

        # 查重逻辑：如果目标文件夹已有同名文件，自动重命名避免覆盖
        if dst.exists():
            base = dst.stem
            ext = dst.suffix
            n = 1
            while True:
                new_dst = dst_folder / f"{base}_dup{n}{ext}"
                if not new_dst.exists():
                    dst = new_dst
                    break
                n += 1

        try:
            shutil.copy2(src, dst) # 使用 copy2 保留文件元数据
            copied_count += 1
            if copied_count % 100 == 0:
                print(f"  已复制 {copied_count} 张图片...")
        except Exception as e:
            print(f"[失败] {name} -> {dst_folder.name}: {e}")

    print(f"[{phase_name}] 处理完毕。共复制: {copied_count} 张。")


def main():
    # 1. 处理 diyiqi
    # 后缀 1-5，对应 video_0001 - video_0005 (偏移量为0)
    # _1 -> 0+1=1
    distribute_images(
        phase_name="diyiqi", 
        start_video_id=1, 
        max_suffix=5, 
        offset_count=0
    )

    print("\n" + "="*30 + "\n")

    # 2. 处理 dierqi
    # 后缀 1-10，对应 video_0006 - video_0015 (偏移量为5)
    # _1 -> 5+1=6
    distribute_images(
        phase_name="dierqi", 
        start_video_id=6, 
        max_suffix=10, 
        offset_count=5
    )
    
    print(f"\n全部完成！输出位置: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()