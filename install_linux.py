"""Linux 推理环境安装：官方源码 + AMD gfx1103/CPU wheels + 最小模型。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from engine_contract import GPT_MODEL, SOVITS_MODEL, REF_WAV

BASE = Path(__file__).resolve().parent
ENGINE_COMMIT = "d7c2210da8c013e81a94bfc7b811a477c99fd506"
AMD_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"


def run(*args, **kwargs):
    print("[执行]", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def download(url, target, sha256=None):
    """失败保留 .part，下次通过 curl 续传；完整文件原子落盘。"""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (not sha256 or file_hash(target) == sha256):
        return
    part = target.with_name(target.name + ".part")
    run("curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "20",
        "--continue-at", "-", "--output", part, url)
    if sha256 and file_hash(part) != sha256:
        part.unlink()
        raise RuntimeError(f"下载校验失败：{target.name}，请重试")
    part.replace(target)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination / name).resolve().is_relative_to(destination):
                raise RuntimeError(f"压缩包路径非法：{name}")
        z.extractall(destination)


def install_models(source, engine):
    from huggingface_hub import HfApi, hf_hub_download
    endpoint = "https://huggingface.co" if source == "hf" else "https://hf-mirror.com"
    repo = "lj1995/GPT-SoVITS"
    # 固定本次快照，校验 LFS SHA256；只下载实际用于推理的两个编码器。
    revision = HfApi(endpoint=endpoint).model_info(repo).sha
    pretrained = engine / "GPT_SoVITS" / "pretrained_models"
    api = HfApi(endpoint=endpoint)
    for item in api.list_repo_tree(repo, revision=revision, recursive=True):
        if not item.path.startswith(("chinese-hubert-base/", "chinese-roberta-wwm-ext-large/")):
            continue
        if not hasattr(item, "size"):
            continue
        target = pretrained / item.path
        if item.lfs:
            download(f"{endpoint}/{repo}/resolve/{revision}/{item.path}", target, item.lfs.sha256)
        elif not target.is_file():
            path = hf_hub_download(repo, item.path, revision=revision, endpoint=endpoint)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    g2pw = engine / "GPT_SoVITS" / "text" / "G2PWModel"
    if not (g2pw / "g2pW.onnx").is_file():
        archive = BASE / "engine_temp" / "G2PWModel.zip"
        download(f"{endpoint}/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip", archive,
                 "46292be0374a49308069233cd5c147ae4c41806558e4781a2467a31a4d8099da")
        extract_zip(archive, g2pw.parent)

    # split-lang 的大语言识别模型默认到首次请求才下载；提前补齐支持离线合成。
    download("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
             pretrained / "fast_langdetect" / "lid.176.bin")

    models = BASE / "models"
    names = (GPT_MODEL, SOVITS_MODEL, REF_WAV)
    if not all((models / n).is_file() for n in names):
        archive = BASE / "engine_temp" / "MamboTTS-v1.2.1-full.zip"
        download("https://github.com/Tsukimisaka/MamboTTS/releases/download/v1.2.1/"
                 "MamboTTS-v1.2.1-full.zip", archive,
                 "8c4e67356a34a347628deeb99680db16ab5daf8f3321cf6e4cd97ebbcdd8bbe3")
        with zipfile.ZipFile(archive) as z:
            for name in names:
                if (models / name).is_file():
                    continue
                matches = [n for n in z.namelist() if Path(n).name == name]
                if len(matches) != 1:
                    raise RuntimeError(f"官方发布包缺少或含重复模型：{name}")
                target = models / name
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_suffix(target.suffix + ".part")
                with z.open(matches[0]) as src, part.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                part.replace(target)

    # 避免首次英文/混合语言推理临时联网。
    import nltk
    nltk_dir = engine / "nltk_data"
    for package in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"):
        if not nltk.download(package, download_dir=str(nltk_dir), quiet=True):
            raise RuntimeError(f"NLTK 数据下载失败：{package}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rocm", "cpu"), default="rocm")
    parser.add_argument("--source", choices=("hf", "hf-mirror"), default="hf")
    parser.add_argument("--desktop", action="store_true", help="同时安装 PySide6 图形客户端")
    parser.add_argument("--skip-deps", action="store_true", help="仅补齐模型文件")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.prefix == sys.base_prefix:
        parser.error("请在 Linux 下使用 ./install.sh，安装到项目独立虚拟环境")
    required_tools = ("git", "curl", "ffmpeg", "ffprobe", "uv")
    if not args.skip_deps:
        required_tools += ("gcc", "g++", "cmake")
    for tool in required_tools:
        if not shutil.which(tool):
            parser.error(f"缺少 {tool}，请先用系统包管理器安装（Arch: sudo pacman -S git curl ffmpeg uv base-devel cmake）")
    engine = Path(os.environ.get("MAMBO_ENGINE_DIR", str(BASE / "GPT-SoVITS"))).resolve()
    if not engine.exists():
        run("git", "clone", "--depth", "1", "--branch", "20250606v2pro",
            "https://github.com/RVC-Boss/GPT-SoVITS.git", engine)
    actual = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
    if actual != ENGINE_COMMIT:
        raise RuntimeError(f"引擎版本不符：{actual}；需要 {ENGINE_COMMIT}。可通过 MAMBO_ENGINE_DIR 指定独立目录。")
    if not args.skip_deps:
        pip = ("uv", "pip", "install", "--python", sys.executable)
        if args.backend == "rocm":
            run(*pip, "--index", AMD_INDEX, "torch[device-gfx1103]==2.11.0+rocm10.0.0",
                "torchaudio==2.11.0+rocm10.0.0")
        else:
            run(*pip, "--index", "https://download.pytorch.org/whl/cpu", "torch==2.6.0+cpu", "torchaudio==2.6.0+cpu")
        # 固定 torch 防止其他依赖把 AMD 版替换成 PyPI/CUDA 版。
        constraints = BASE / "engine_temp" / "torch-constraints.txt"
        constraints.parent.mkdir(exist_ok=True)
        import torch
        import torchaudio
        constraints.write_text(f"torch=={torch.__version__}\ntorchaudio=={torchaudio.__version__}\n")
        run(*pip, "-r", BASE / "requirements-engine.txt", "-c", constraints)
        if args.desktop:
            run(*pip, "-r", BASE / "requirements.txt", "-c", constraints)
    install_models(args.source, engine)
    (engine / ".mambo_install_ok").write_text(json.dumps({"commit": actual, "backend": args.backend}))
    print("[完成] ./run.sh 启动 Web 工作台；./run.sh --api 仅启动 API；./run.sh --check 检查设备。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[安装失败] {exc}", file=sys.stderr)
        sys.exit(1)
