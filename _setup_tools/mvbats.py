import glob, os, shutil
d = os.path.join(os.environ["USERPROFILE"], "Desktop")
old = os.path.join(d, "OOXX-robot-v1.1", "_setup_tools", "old_bats")
os.makedirs(old, exist_ok=True)
for f in glob.glob(os.path.join(d, "啟動OOXX*.bat")):
    shutil.move(f, os.path.join(old, os.path.basename(f)))
    print("moved", os.path.basename(f))
for f in glob.glob(os.path.join(d, "OOXX-robot-v1.1", "_setup_tools", "newbats", "*.bat")):
    shutil.copy(f, os.path.join(d, os.path.basename(f)))
    print("installed", os.path.basename(f))
print(sorted(os.listdir(d)))
