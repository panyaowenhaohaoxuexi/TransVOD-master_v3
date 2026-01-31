import os
import re
import shutil

def move_images(root_dir: str):
    root_dir = os.path.abspath(root_dir)

    # 目标子文件夹：video_0001 ~ video_0005
    targets = {i: os.path.join(root_dir, f"video_{i:04d}") for i in range(1, 6)}
    for p in targets.values():
        os.makedirs(p, exist_ok=True)

    # 匹配：0001_1.jpg 这类
    pattern = re.compile(r"^\d+_([1-5])\.jpg$", re.IGNORECASE)

    moved = 0
    skipped = 0

    for name in os.listdir(root_dir):
        src = os.path.join(root_dir, name)
        if not os.path.isfile(src):
            continue

        m = pattern.match(name)
        if not m:
            continue

        idx = int(m.group(1))  # 1~5
        dst = os.path.join(targets[idx], name)

        # 若目标已存在同名文件，避免覆盖：自动改名追加 _dupN
        if os.path.exists(dst):
            base, ext = os.path.splitext(name)
            n = 1
            while True:
                new_name = f"{base}_dup{n}{ext}"
                dst = os.path.join(targets[idx], new_name)
                if not os.path.exists(dst):
                    break
                n += 1

        try:
            shutil.move(src, dst)  # 剪切
            moved += 1
        except Exception as e:
            print(f"[ERROR] move failed: {src} -> {dst} ({e})")
            skipped += 1

    print(f"Done. moved={moved}, skipped={skipped}")
    print("Targets:")
    for i in range(1, 6):
        print(f"  {i}: {targets[i]}")

if __name__ == "__main__":
    folder = input("请输入文件夹路径: ").strip().strip('"').strip("'")
    if not os.path.isdir(folder):
        raise SystemExit(f"不是有效目录: {folder}")
    move_images(folder)
