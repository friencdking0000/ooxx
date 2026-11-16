import hashlib, json, os, sys
root = sys.argv[1]
SKIP_DIRS = {"out", "Log", "voice_cache", "__pycache__", "_setup_tools", "dataset", "runs", ".git", "dist", "calib_archive"}
SKIP_FILES = {"cam_full.jpg", "cam_depth.png", "cam_error.txt", "depth_request", "access.txt", "cert.pem", "key.pem", "tuning.json", "cam_align.json", "bash.exe.stackdump", "reach_cache.json", "stock_count.json"}
res = {}
for d, ds, fs in os.walk(root):
    ds[:] = [x for x in ds if x not in SKIP_DIRS]
    for fn in fs:
        if fn in SKIP_FILES or fn.endswith((".tmp", ".pyc")): continue
        p = os.path.join(d, fn)
        rel = os.path.relpath(p, root).replace("\\", "/")
        with open(p, "rb") as f: b = f.read().replace(b"\r\n", b"\n")
        res[rel] = [hashlib.sha256(b).hexdigest()[:16], os.path.getsize(p), int(os.path.getmtime(p))]
json.dump(res, open(sys.argv[2], "w", encoding="utf-8") if len(sys.argv) > 2 else sys.stdout, ensure_ascii=True)
