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

def oss_signed_req(method, key, body=None):
    d = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    ct = "application/octet-stream" if body else ""
    s2s = "%s\n\n%s\n%s\n/%s/%s" % (method, ct, d, BUCKET, key)
    sig = base64.b64encode(hmac.new(SK.encode(), s2s.encode(), hashlib.sha1).digest()).decode()
    req = urllib.request.Request("https://%s/%s" % (HOST, key), data=body, method=method)
    req.add_header("Date", d)
    if ct: req.add_header("Content-Type", ct)
    req.add_header("Authorization", "OSS %s:%s" % (AK, sig))
    return urllib.request.urlopen(req, timeout=180)

def oss_put(key, filepath):
    r = oss_signed_req("PUT", key, open(filepath, "rb").read())
    log("oss_put", key, "->", r.status)

def download(url, path):
    log("download", url)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(path, "wb") as f:
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk: break
            f.write(chunk); n += len(chunk)
    log("  done %.1fMB %.0fs" % (n/1e6, time.time()-t0))

def sh(cmd):
    log("$", cmd)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=LOG, stderr=LOG)
    log("  exit=%d %.0fs" % (r.returncode, time.time()-t0))
    if r.returncode != 0:
        raise RuntimeError("cmd failed: " + cmd)

try:
    log("=== runner start", time.strftime("%F %T"), "py:", sys.version.split()[0], "nproc", os.cpu_count())
    # 0) python3.11 from OSS（python-build-standalone）
    if not os.path.exists("/root/py311/python/bin/python3"):
        r = oss_signed_req("GET", "code/cpython311.tar.gz")
        open("/root/py311.tar.gz", "wb").write(r.read())
        with tarfile.open("/root/py311.tar.gz") as t:
            t.extractall("/root/py311")
        log("python3.11 installed:", os.popen("/root/py311/python/bin/python3 -V").read().strip())
    P = "/root/py311/python/bin"
    # 1) 拉任务包（源码+输入图）：jsDelivr 优先，OSS 签名 GET 兜底
    try:
        download("https://cdn.jsdelivr.net/gh/janxland/dlc-task-pkg@3a6d8f1ea7a3/task.tar.gz", "task.tar.gz")
        log("task package from jsdelivr")
    except Exception as e:
        log("jsdelivr fail:", repr(e)[:150], " -> OSS fallback")
        r = oss_signed_req("GET", "task/task.tar.gz")
        open("task.tar.gz", "wb").write(r.read())
        log("task package from OSS")
    with tarfile.open("task.tar.gz") as t:
        t.extractall(".")
    log("extracted:", os.listdir("task")[:6])
    if not os.path.exists("task/depth_pro_2_src/cli.py"):
        raise RuntimeError("source missing after extract")
    # 2) torch cpu cp38 + 依赖（py3.8 兼容版本）
    log("pull torch wheels from OSS")
    r = oss_signed_req("GET", "code/torch311.whl"); open("/root/work/torch311.whl", "wb").write(r.read())
    r = oss_signed_req("GET", "code/torchvision311.whl"); open("/root/work/torchvision311.whl", "wb").write(r.read())
    sh(P + "/pip install -q --no-deps -U /root/work/torch311.whl /root/work/torchvision311.whl")
    sh(P + "/pip install -q 'numpy==1.24.4' 'pillow==10.3.0' 'scipy==1.10.1' 'timm>=1.0.27,<1.1' 'huggingface-hub' 'av<14' 'typing-extensions>=4.8' pillow-heif click matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple")
    # 3) 权重（Apple CDN，先查缓存不重复下）
    os.makedirs("torch_home/hub/checkpoints", exist_ok=True)
    for fn in ["depthpro2-fov-ew55zg4kq7.pt", "depthpro2-s-bgcc6f8bg3.pt", "depthpro2-l-tegstu7fim.pt"]:
        p = "torch_home/hub/checkpoints/" + fn
        if os.path.exists(p) and os.path.getsize(p) > 1e6:
            log("cached:", fn); continue
        download("https://ml-site.cdn-apple.com/models/depth-pro-2/" + fn, p)
    # 4) 推理（S + L）：把平铺源码组装成包目录
    os.makedirs("pkg", exist_ok=True)
    sh("cp -r task/depth_pro_2_src pkg/depth_pro_2")
    os.environ["PYTHONPATH"] = "/root/work/pkg"
    os.environ["TORCH_HOME"] = "/root/work/torch_home"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    sh(P + "/python3 -m depth_pro_2.cli predict -i task/input/yuantu.jpg -o output/ -m depthpro2-s --save-npz --save-png16")
    sh(P + "/python3 -m depth_pro_2.cli predict -i task/input/yuantu.jpg -o output/ -m depthpro2-l --save-npz --save-png16")
    # 5) 上传结果
    for f in sorted(os.listdir("output")):
        oss_put("output/" + f, "output/" + f)
    log("=== ALL DONE")
except Exception:
    log("!!! RUNNER FAILED:")
    log(traceback.format_exc())
finally:
    try:
        oss_put("logs/run.log", "/root/work/run.log")
        print("LOG_UPLOADED")
    except Exception as e:
        print("log upload failed:", repr(e)[:200])
