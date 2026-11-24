# -*- coding: utf-8 -*-
import os, sys, subprocess, urllib.request, base64, hashlib, hmac, datetime, tarfile, time, traceback

AK = os.environ["AK_ID"]; SK = os.environ["AK_SEC"]
BUCKET = "depthpro2-janx-1953654729297796"
HOST = BUCKET + ".oss-cn-hangzhou.aliyuncs.com"
os.makedirs("/root/work", exist_ok=True)
os.chdir("/root/work")
LOG = open("/root/work/run.log", "a", buffering=1)

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()

def oss_put(key, filepath):
    d = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    body = open(filepath, "rb").read()
    s2s = "PUT\n\napplication/octet-stream\n%s\n/%s/%s" % (d, BUCKET, key)
    sig = base64.b64encode(hmac.new(SK.encode(), s2s.encode(), hashlib.sha1).digest()).decode()
    req = urllib.request.Request("https://%s/%s" % (HOST, key), data=body, method="PUT")
    req.add_header("Date", d); req.add_header("Content-Type", "application/octet-stream")
    req.add_header("Authorization", "OSS %s:%s" % (AK, sig))
    r = urllib.request.urlopen(req, timeout=180)
    log("oss_put", key, "->", r.status)

def download(url, path):
    log("download", url)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    log("  done %.1fMB %.0fs" % (os.path.getsize(path)/1e6, time.time()-t0))

def sh(cmd):
    log("$", cmd)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=LOG, stderr=LOG)
    log("  exit=%d %.0fs" % (r.returncode, time.time()-t0))
    if r.returncode != 0:
        raise RuntimeError("cmd failed: " + cmd)

try:
    log("=== runner start", time.strftime("%F %T"), "nproc", os.cpu_count())
    # 1) miniconda py311
    download("https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh", "/root/mc.sh")
    sh("bash /root/mc.sh -b -p /root/mc")
    sh("/root/mc/bin/conda create -y -n dp python=3.11")
    P = "/root/mc/envs/dp/bin"
    # 2) torch cpu + deps
    sh(P + "/pip install -q --no-deps https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.4.0%2Bcpu-cp311-cp311-linux_x86_64.whl https://mirrors.aliyun.com/pytorch-wheels/cpu/torchvision-0.19.0%2Bcpu-cp311-cp311-linux_x86_64.whl")
    sh(P + "/pip install -q numpy pillow scipy timm av pillow-heif click huggingface_hub matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple")
    # 3) weights from Apple CDN
    os.makedirs("torch_home/hub/checkpoints", exist_ok=True)
    for fn in ["depthpro2-fov-ew55zg4kq7.pt", "depthpro2-s-bgcc6f8bg3.pt", "depthpro2-l-tegstu7fim.pt"]:
        download("https://ml-site.cdn-apple.com/models/depth-pro-2/" + fn, "torch_home/hub/checkpoints/" + fn)
    # 4) inference
    sh(P + "/python -m depth_pro_2.cli predict -i input/yuantu.jpg -o output/ -m depthpro2-s --save-npz --save-png16")
    sh(P + "/python -m depth_pro_2.cli predict -i input/yuantu.jpg -o output/ -m depthpro2-l --save-npz --save-png16")
    # 5) push outputs
    for f in sorted(os.listdir("output")):
        oss_put("output/" + f, "output/" + f)
    log("=== ALL DONE")
except Exception:
    log("!!! RUNNER FAILED:")
    log(traceback.format_exc())
finally:
    try:
        oss_put("logs/run.log", "/root/work/run.log")
    except Exception as e:
        print("log upload failed:", e)
