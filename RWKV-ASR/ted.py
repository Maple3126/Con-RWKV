import os
import subprocess

stm_dir = "/home/TEDLIUM_release-3/speaker-adaptation/dev/stm"
sph_dir = "/home/TEDLIUM_release-3/speaker-adaptation/dev/sph"
output_root = "/home/TEDDATA/teddev"

os.makedirs(output_root, exist_ok=True)

for stm_file in os.listdir(stm_dir):
    if not stm_file.endswith(".stm"):
        continue

    base = stm_file.replace(".stm", "")
    parts = base.split("_")
    part1 = parts[0]
    part2 = "_".join(parts[1:])

    stm_path = os.path.join(stm_dir, stm_file)
    sph_path = os.path.join(sph_dir, base + ".sph")

    if not os.path.exists(stm_path):
        print(f"❌ 找不到 STM 檔案：{stm_path}")
        continue
    if not os.path.exists(sph_path):
        print(f"❌ 找不到 SPH 檔案：{sph_path}")
        continue

    out_dir = os.path.join(output_root, part1, part2)
    os.makedirs(out_dir, exist_ok=True)
    trans_path = os.path.join(out_dir, f"{part1}-{part2}.trans.txt")
    trans_lines = []

    success_count = 0
    skip_count = 0

    with open(stm_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith(";;") or len(line.strip()) == 0:
                skip_count += 1
                continue

            parts = line.strip().split()
            if len(parts) < 7:
                skip_count += 1
                continue

            try:
                start = float(parts[3])
                end = float(parts[4])
            except ValueError:
                print(f"❌ 時間格式錯誤：{line.strip()}")
                skip_count += 1
                continue

            transcript = " ".join(parts[6:])
            utt_id = f"{part1}-{part2}-{i+1:04d}"
            wav_path = os.path.join(out_dir, f"{utt_id}.wav")

            subprocess.run([
                "ffmpeg", "-i", sph_path,
                "-ss", str(start), "-to", str(end),
                "-ar", "16000", "-ac", "1", "-loglevel", "quiet",
                wav_path
            ])

            trans_lines.append(f"{utt_id} {transcript.upper()}")
            success_count += 1

    with open(trans_path, "w", encoding="utf-8") as f:
        f.write("\n".join(trans_lines))
        
    print(f"✅ {base}: 處理成功 {success_count} 句，跳過 {skip_count} 句。")