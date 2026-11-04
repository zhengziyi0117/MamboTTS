import os
import sys
import argparse
import json
import socket
import subprocess

from engine_contract import (
    DEFAULT_REF_TEXT, SOVITS_MODEL, GPT_MODEL, REF_WAV, API_HOST, API_PORT,
)


def _project_dir():
    return os.path.dirname(os.path.abspath(__file__))


def engine_paths(base=None):
    """集中返回引擎与模型文件路径，install_engine 生成 bat 时也复用同一清单。"""
    base = base or _project_dir()
    gsv_dir = os.path.abspath(os.environ.get("MAMBO_ENGINE_DIR", os.path.join(base, "GPT-SoVITS")))
    models_dir = os.path.join(base, "models")
    python_exe = (os.path.join(gsv_dir, "runtime", "python.exe") if os.name == "nt"
                  else os.path.join(base, ".venv", "bin", "python"))
    return {
        "gsv_dir": gsv_dir,
        "models_dir": models_dir,
        "python_exe": os.environ.get("MAMBO_ENGINE_PYTHON", python_exe),
        "api_py": os.path.join(gsv_dir, "api.py"),
        "sovits": os.path.join(models_dir, SOVITS_MODEL),
        "gpt": os.path.join(models_dir, GPT_MODEL),
        "ref_wav": os.path.join(models_dir, REF_WAV),
    }


def preflight_check(base=None):
    """启动前检查引擎与模型文件完整性，返回错误列表（空 = 就绪）。
    避免半安装状态下静默卡 60 秒后报「启动超时」，让人摸不着头脑。"""
    p = engine_paths(base)
    required = [
        ("GPT-SoVITS 整合包目录", p["gsv_dir"]),
        ("引擎入口 api.py", p["api_py"]),
        ("引擎 Python 环境", p["python_exe"]),
        ("SoVITS 模型", p["sovits"]),
        ("GPT 模型", p["gpt"]),
        ("参考音频", p["ref_wav"]),
    ]
    if os.name != "nt":
        pretrained = os.path.join(p["gsv_dir"], "GPT_SoVITS", "pretrained_models")
        required.extend([
            ("中文 BERT 权重", os.path.join(pretrained, "chinese-roberta-wwm-ext-large", "pytorch_model.bin")),
            ("HuBERT 权重", os.path.join(pretrained, "chinese-hubert-base", "pytorch_model.bin")),
            ("中文 G2PW 模型", os.path.join(p["gsv_dir"], "GPT_SoVITS", "text", "G2PWModel", "g2pW.onnx")),
        ])
    return [f"{name}: {path}" for name, path in required if not os.path.exists(path)]


def port_in_use(host=API_HOST, port=API_PORT):
    """探测端口是否已被其他进程占用（api.py 绑定失败会抛英文 traceback，
    提前用中文提示更好排障）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


# 在独立进程里探测：ROCm 初始化失败/崩溃不能拖垮启动器。
DEVICE_PROBE = r'''
import json, torch
info = {"available": False, "hip": torch.version.hip, "torch": torch.__version__}
if torch.cuda.is_available():
    info["name"] = torch.cuda.get_device_name(0)
    info["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    x = torch.ones((32, 32), device="cuda")
    assert (x @ x).mean().item() == 32
    conv = torch.nn.Conv1d(4, 4, 3).to("cuda")
    assert torch.isfinite(conv(torch.ones(1, 4, 32, device="cuda"))).all().item()
    torch.cuda.synchronize()
    info["available"] = True
print(json.dumps(info))
'''


def select_device(python_exe, requested="auto", env=None):
    """ROCm 在 PyTorch 中也使用 cuda 设备名；强制 GPU 模式失败时明确退出。"""
    if requested == "cpu":
        return "cpu", {"reason": "用户指定 CPU"}
    try:
        probe = subprocess.run([python_exe, "-c", DEVICE_PROBE], env=env,
                               capture_output=True, text=True, timeout=90)
        if probe.returncode:
            raise RuntimeError((probe.stderr or probe.stdout)[-1500:])
        info = json.loads(probe.stdout.strip().splitlines()[-1])
        if not info["available"]:
            raise RuntimeError("PyTorch 未检测到可用 GPU")
        if requested == "rocm" and not info["hip"]:
            raise RuntimeError("当前安装的是非 ROCm 版 PyTorch，请运行 ./install.sh --backend rocm")
        return "cuda", info
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        if requested != "auto":
            raise RuntimeError(f"{requested} 运算检测失败：{exc}") from exc
        return "cpu", {"reason": f"GPU 运算检测失败，自动使用 CPU：{exc}"}


def engine_env(threads=None):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    # 核显与 CPU 共用内存；避免 BLAS/ONNX 在 16 个逻辑核上争抢线程。
    count = str(threads or min(8, os.cpu_count() or 1))
    if threads is not None:
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = count
    else:
        env.setdefault("OMP_NUM_THREADS", count)
        env.setdefault("MKL_NUM_THREADS", count)
    nltk_dir = os.path.join(engine_paths()["gsv_dir"], "nltk_data")
    env.setdefault("NLTK_DATA", nltk_dir)
    # 语音长度变化会触发新的卷积形状，避免每次请求重新编译/穷举算法。
    env.setdefault("MIOPEN_FIND_MODE", "FAST")
    return env


def resolve_precision(requested, device, info):
    if requested != "auto":
        return requested
    if device == "cpu":
        return "full"
    if info.get("hip"):
        # 780M 的 FP16 已在完整中文合成中验证，其他 AMD 设备保守使用 FP32。
        if info.get("arch", "").startswith("gfx1103") or "780M" in info.get("name", ""):
            return "half"
        return "full"
    return "auto"


def build_command(paths, host, port, device, precision):
    # Linux 使用适配层加载原 API，缓存特征；Windows 整合包保留原入口。
    entry = paths["api_py"] if os.name == "nt" else os.path.join(_project_dir(), "engine_runtime.py")
    cmd = [paths["python_exe"], "-u", entry,
           "-a", host, "-p", str(port), "-s", paths["sovits"], "-g", paths["gpt"],
           "-dr", paths["ref_wav"], "-dt", DEFAULT_REF_TEXT, "-dl", "zh", "-d", device]
    if precision == "full":
        cmd.append("-fp")
    elif precision == "half":
        cmd.append("-hp")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="MamboTTS 推理服务（Linux AMD ROCm / CPU / Windows）")
    parser.add_argument("--host", default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    parser.add_argument("--device", choices=["auto", "cpu", "rocm", "cuda"],
                        default=os.environ.get("MAMBO_DEVICE", "auto"))
    parser.add_argument("--precision", choices=["auto", "full", "half"],
                        default=os.environ.get("MAMBO_PRECISION", "auto"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--check", action="store_true", help="检查文件和 GPU 运算后退出")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or (args.threads is not None and args.threads < 1):
        parser.error("端口须为 1–65535，线程数须为正整数")
    current_dir = _project_dir()
    p = engine_paths(current_dir)

    missing = preflight_check(current_dir)
    if missing:
        print("[错误] 引擎文件不完整，无法启动：")
        for item in missing:
            print(f"  - {item}")
        print("请运行 ./install.sh（Linux）或 MamboTTS.bat（Windows）完成安装。")
        sys.exit(1)

    env = engine_env(args.threads)
    try:
        device, info = select_device(p["python_exe"], args.device, env)
    except RuntimeError as exc:
        print(f"[错误] {exc}", flush=True)
        return 2
    precision = resolve_precision(args.precision, device, info)
    if device == "cpu" and precision == "half":
        parser.error("CPU 推理请使用 full 精度")
    print(f"[设备] {device}，精度: {precision}，检测: {json.dumps(info, ensure_ascii=False)}", flush=True)
    if args.check:
        print("[就绪] 引擎、模型与设备检查通过。", flush=True)
        return 0

    # 端口占用预检：若已被其他引擎实例或无关程序占住，明确提示，
    # 而不是让 api.py 抛一段 Uvicorn 的 Errno 10048 traceback
    if port_in_use(args.host, args.port):
        print(f"[错误] 端口 {args.host}:{args.port} 已被占用。")
        print("  · 若已有 MamboTTS/GPT-SoVITS 在运行，无需重复启动；")
        print("  · 否则请关闭占用该端口的程序后重试。")
        sys.exit(3)

    print("====================================================")
    print("         MamboTTS 本地语音推理引擎启动程序")
    print("====================================================")
    print(f"[状态] 正在加载 SoVITS 模型: {os.path.basename(p['sovits'])}")
    print(f"[状态] 正在加载 GPT 模型: {os.path.basename(p['gpt'])}")
    print(f"[提示] 推理服务即将启动在: http://{args.host}:{args.port}")
    print("====================================================\n", flush=True)

    # 构建启动命令（旧版 api.py 支持 -s/-g/-dr/-dt/-dl 参数）
    cmd = build_command(p, args.host, args.port, device, precision)

    # 在 GPT-SoVITS 目录下运行 API，保证工作目录正确
    try:
        if os.name != "nt":
            os.chdir(p["gsv_dir"])
            os.execve(p["python_exe"], cmd, env)
        proc = subprocess.Popen(cmd, cwd=p["gsv_dir"], env=env)
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[状态] 推理服务已关闭。")
    except Exception as e:
        print(f"\n[错误] 启动推理引擎失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
