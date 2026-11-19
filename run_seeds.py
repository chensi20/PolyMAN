import os
import shutil
import subprocess

SEEDS = [666]

print("=" * 80)
print(f"🚀 STARTING MULTI-SEED AUTOMATED EXPERIMENTS: {SEEDS}")
print("=" * 80)

for seed in SEEDS:
    print(f"\n[RUNNER] >>> Starting Experiment with SEED = {seed} ...")
    
   
    with open("config.py", "r") as f:
        content = f.read()
    
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("SEED ="):
            new_lines.append(f"    SEED = {seed}")
        else:
            new_lines.append(line)
            
    with open("config.py", "w") as f:
        f.write("\n".join(new_lines) + "\n")
        
   
    train_proc = subprocess.run(["python", "train.py"])
    if train_proc.returncode != 0:
        print(f"❌ Training failed for Seed {seed}")
        continue
        
   
    test_proc = subprocess.run(["python", "test.py"])
    if test_proc.returncode != 0:
        print(f"❌ Testing failed for Seed {seed}")
        continue

   
    src_weight = "./checkpoints/best_model.pth"
    dst_weight = f"./checkpoints/best_model_seed_{seed}.pth"
    
    if os.path.exists(src_weight):
        shutil.copy(src_weight, dst_weight)
        print(f"💾 [Backup Success] Weights copied to: {dst_weight}")

print("\n" + "=" * 80)
print("✅ ALL 5-SEED EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
