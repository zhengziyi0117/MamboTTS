<h1 align="center">MamboTTS</h1>

<p align="center"><strong>面向 B 站 / 抖音曼波讲故事类解说场景的本地配音工具（Windows / Linux）</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-10%2F11_64位-0A0A0B?style=flat-square" alt="Windows" />
  <img src="https://img.shields.io/badge/Linux-AMD_780M_%2F_CPU-0A0A0B?style=flat-square" alt="Linux AMD 780M / CPU" />
  <img src="https://img.shields.io/badge/Engine-GPT--SoVITS-0A0A0B?style=flat-square" alt="GPT-SoVITS" />
  <img src="https://img.shields.io/badge/Version-v1.2.1-0A0A0B?style=flat-square" alt="v1.2.1" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0A0A0B?style=flat-square" alt="MIT" /></a>
</p>

> 基于 GPT-SoVITS 合成本地解说语音。Windows 使用 NVIDIA 引擎整合包（约 8.2GB）；Linux 使用源码与独立环境，支持 AMD Radeon 780M 和 CPU。首次安装需联网下载，中文合成阶段可离线。

## Linux / AMD Radeon 780M 部署

Linux 可通过官方 GPT-SoVITS 源码运行，支持 **780M（gfx1103）ROCm 加速**及 CPU 推理，
无需下载 Windows 整合包。使用项目独立的 Python 3.10 环境，不依赖 Arch 的系统 Python 版本。

```bash
# Arch Linux 安装基础工具；Debian / Ubuntu 用对应包管理器安装 git、curl、ffmpeg、uv
sudo pacman -S --needed git curl ffmpeg uv base-devel cmake

# 安装 AMD 官方 TheRock stable gfx1103 wheels、推理依赖和音色模型
./install.sh --backend rocm
# 下载 Hugging Face 文件较慢时可追加 --source hf-mirror

# 检查模型和 GPU 的矩阵乘法、卷积运算
./run.sh --check --device rocm

# 手动启动 Web 工作台，浏览器打开 http://127.0.0.1:7860
./run.sh
```

### Web 生成与试听

`./run.sh` 启动网页和推理引擎。浏览器访问 `http://127.0.0.1:7860`；
同一局域网的其他设备访问 `http://小主机IP:7860`。
页面支持输入文案、选择语言、调整语速、生成配音、波形预览、播放试听、下载 WAV，
并保留最近 50 条生成记录。文案草稿会保存在当前浏览器。

默认采用「完整句子」断句：逗号留在同一段中，句号、问号、感叹号和手动换行才分段；
过长句子在约 100 字内优先找逗号或空格，避免一次输入超过模型适合处理的长度。
小数、版本号、网址中的点不会当作句号拆开，英文单词也不会按字数切开。
也可以在网页切换「按逗号分句」，对比同一段文案的效果。

例如「直到全球气温断崖式暴跌，城市一夜冰封，我才明白那不是bug」现在作为完整一段合成，
原来的方式会切成 3 段，每段后由上游固定追加 300 毫秒静音。
客户端显式使用 `top_k=15 / top_p=0.6 / temperature=0.6`，尝试降低采样造成的韵律波动。
这不保证消除模型在「我才明白／那不是」之间自行产生的停顿；该位置原本没有程序切分。
句内短英文（如「这是bug导致的」）被上游额外补入的逗号音素会由 Linux 适配层移除，
文案中的实际标点和纯英文短句保留。句尾 `bug。` 没有触发这一补逗号逻辑。

所有程序都在当前终端中运行，按 **Ctrl+C** 关闭网页及本次启动的引擎。
如果引擎是你另外启动的，退出网页时会保留它。
音频与历史保存在 `outputs/web/`，安装脚本只负责安装，启动由你手动执行。

```bash
./run.sh --device rocm                 # 使用 780M
./run.sh --device cpu                  # 使用 CPU
./run.sh --host 127.0.0.1 --port 8080    # 只允许本机访问，自定义网页端口
./run.sh --api                         # 只启动推理 API，默认 9880
./run.sh --no-engine                   # 只启动网页，连接已经启动的本地引擎
```

Web 端口默认 `7860`，引擎端口默认 `9880`，可用 `--engine-port` 调整后者。
多设备同时提交时，生成任务会排队执行，避免同时占满核显内存。

首次安装会下载约数 GB 的依赖和模型。预留至少 12GB 磁盘空间，建议 16GB 以上内存。
安装器只安装推理所需依赖，G2PW 使用 CPU 版 ONNX Runtime，不会额外安装 NVIDIA CUDA 包。
曼波音色缺失时从本项目 `v1.2.1` 官方发布包提取，BERT / HuBERT 和 G2PW 使用上游公开模型。
`GPT-SoVITS` 固定为 `20250606v2pro` 对应提交，避免上游接口变化。

AMD 软件包采用 [TheRock 官方安装方式](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)，
安装包含 `device-gfx1103` 的原生内核，无需设置 `HSA_OVERRIDE_GFX_VERSION`。
Linux 需能访问 `/dev/kfd` 和 `/dev/dri/renderD*`；用户通常应属于 `render` / `video` 组。
具体机器以 `--check --device rocm` 运算结果为准。

默认 `--device auto` 会先在独立进程中检测 GPU，检测失败则使用 CPU。
780M 默认使用经过中文合成验证的 FP16；CPU 和其他 AMD 型号默认使用 FP32。
可用 `--precision full` 强制全精度。
强制 `--device rocm` 时检测失败会明确报错。

启动器默认设置 `MIOPEN_FIND_MODE=FAST`，降低变长语音反复选择卷积算法的开销，
含义见 [MIOpen 官方文档](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html)。
已有同名环境变量会被保留，可自行覆盖。

本机验证：Arch Linux、Ryzen 7 8745HS / Radeon 780M、PyTorch 2.11.0 + ROCm 10.0.0。
约 4.2 秒的中文音频在核显 FP16 + FAST 模式下耗时约 5.3 秒；
较长的约 13.2 秒音频耗时约 8.1 秒；CPU 模式同一短句约 5 秒。
这些是单次实测，耗时会随文案、首次算子准备和系统负载变化。

Linux 入口还会缓存同一参考音频的 HuBERT 特征、参考频谱和最近 8 条短文案的音素 / BERT 特征，
减少连续生成时的重复计算；参考文件变化时自动失效，缓存大小有上限。
整句合成减少了推理初始化次数，但单段变长也会增加自回归上下文长度，净提速需在实际文案上测量。
这些新改动已验证断句、缓存命中与失效逻辑，尚未测量新版本的真实合成耗时和听感；
上面的性能数字来自改动前。你下次手动重新启动网页及引擎后才会加载适配层。

```bash
# 强制使用 780M / 强制使用 CPU（已安装 ROCm 环境也可以直接运行 CPU）
./run.sh --device rocm
./run.sh --device cpu

# 完全不安装 GPU 依赖的 CPU 环境
./install.sh --backend cpu

# 可选图形客户端（服务部署无需 Qt 或显示器）
./install.sh --backend rocm --desktop
./run.sh --desktop

# 仅启动 API 并允许局域网访问
./run.sh --api --device rocm --host 0.0.0.0 --port 9880
```

合成示例（省略参考音频参数时使用服务器的默认曼波参考音频）：

```bash
curl --fail http://127.0.0.1:9880/ \
  -H 'Content-Type: application/json' \
  -d '{"text":"大家好，这是在小主机上生成的曼波配音。","text_language":"zh","speed":1.0,"cut_punc":"","top_p":0.6,"temperature":0.6}' \
  --output test.wav
```

可配置环境变量：`MAMBO_DEVICE`（auto / cpu / rocm / cuda）、`MAMBO_PRECISION`（auto / full / half）、
`MAMBO_ENGINE_DIR`（源码目录）、`MAMBO_ENGINE_PYTHON`（引擎解释器）、
`OMP_NUM_THREADS` / `MKL_NUM_THREADS`（默认 8 线程以内）、`MAMBO_TTS_TIMEOUT`（客户端读取超时，Linux 默认 600 秒）。

## 功能

- **文案合成 —** 输入文案，调用本地 GPT-SoVITS 引擎合成 WAV；保留句内上下文，超长文本自动分段。
- **语速调节 —** 0.1x–3.0x 连续滑条调节（引擎原生变速不变调）。
- **播放试听 —** 内置播放器，支持进度条拖动与实时时间预览。
- **历史记录 —** 合成历史跨重启保留（自动过滤已删除文件），支持回放 / 另存 / 删除。
- **运行日志 —** 主界面实时显示运行日志，引擎状态一目了然，问题可自查。
- **文件命名 —** 默认以文案前 20 字作为文件名，可自定义文件名与保存目录。
- **深浅主题 —** 深色 / 浅色一键切换，选择自动记忆。
- **单一入口 —** 双击 `MamboTTS.bat` 自动完成依赖安装、引擎下载与启动、拉起主界面。

## 快速开始

### 1. 下载并解压

前往 [Releases](https://github.com/Tsukimisaka/MamboTTS/releases) 下载 `MamboTTS-v1.2.1-full.zip`，解压到全英文路径，例如 `D:\MamboTTS\`。

| 内容 | 说明 |
| --- | --- |
| 客户端 | 发布包内含桌面图形客户端与预设音色 |
| 引擎本体 | 发布包不含，首次运行时自动下载（约 8.2GB） |

### 2. 双击启动

双击 `MamboTTS.bat`。

- 首次运行自动安装依赖并下载引擎：多源自动测速选最快，支持断点续传与失败自动换源，界面实时显示下载百分比 / 速度 / 剩余量。
- 引擎就绪后进入主界面（首次约 10–30 秒），关闭窗口时引擎自动退出。

### 3. 输入并合成

在文本框输入解说文案 → 调节语速 → 点击「合成配音」。完成后可在播放器试听，历史记录支持回放与另存。

## 系统要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows 10 / 11 64 位 |
| 显卡 | NVIDIA GPU（50 系与其余 NVIDIA 对应不同整合包，安装时按 nvidia-smi 自动预选） |
| 网络 | 首次运行需联网下载引擎约 8.2GB；合成阶段可离线 |
| 磁盘 | 需容纳下载包及解压后的引擎与模型，建议预留 30GB+ |
| 解压路径 | 须为全英文路径，中文路径可能导致引擎加载异常 |

## 常见问题

<details>
<summary>安装很慢或看起来卡住？</summary>

安装器内置多源下载（ModelScope / hf-mirror / HuggingFace 等自动测速选最快），下载期间实时显示百分比、速度与剩余量；单源失败自动切换，中断后重新双击可断点续传。
</details>

<details>
<summary>所有下载源都失败怎么办？</summary>

两种自救方式：在 `mirrors.txt` 中添加可用镜像；或手动下载整合包放入 `engine_temp/` 后重试安装（安装器会跳过下载直接解压）。详见「高级配置」。
</details>

<details>
<summary>一直显示引擎启动中？</summary>

首次加载约需 10–30 秒属正常。若超时，界面提供「重启引擎」入口，并可在运行日志中查看具体原因。
</details>

## 工作原理

MamboTTS 提供 Windows 图形客户端与 Linux Web 工作台。本仓库不包含 GPT-SoVITS 权重或源码：安装时下载引擎，客户端通过本地 HTTP API（`127.0.0.1:9880`）调用。Linux 适配层在加载时安装推理优化，不修改下载的源码文件；接口契约见 `engine_contract.py`。

## 编辑器扩展

仓库 `Premiere-Pro-UXP/` 目录提供用于 Premiere Pro 的 UXP 配音插件，由 [@2710165659](https://github.com/2710165659) 贡献，为独立于主客户端的编辑器扩展，适用于在 Premiere Pro 内发起配音与试听。

| 文件 / 目录 | 说明 |
| --- | --- |
| `index.html` / 相关 JS | 插件面板与逻辑 |
| `styles.css` | 面板样式 |
| `manifest.json` | UXP 插件清单 |
| `install.ps1` | Windows 安装脚本 |
| `preview-player/` | 基于 Rust 的预览播放器 |

Premiere Pro 为 Adobe 商标，本项目与其不存在隶属或背书关系。

## 高级配置

- **自定义下载镜像**：编辑 `mirrors.txt`，每行一个 URL 模板（`{file}` 为整合包文件名占位符）。
- **手动放置整合包**：将 7z 文件放入 `engine_temp/` 后重新运行安装，即可跳过下载。
- **自建 / 远程引擎**：主界面「引擎接口」可填写自定义服务地址（对应 `config.json` 的 `api_url`）。
- **配置文件**：`config.json` 由程序自动生成，模板见 `config.example.json`。

## 开发者

| 模块 | 职责 |
| --- | --- |
| `app.py` | 主界面：合成 / 播放器 / 历史 / 日志 / 主题 |
| `web_app.py` · `web/` | Web 工作台：生成队列 / 波形试听 / 下载 / 历史记录 |
| `launcher.py` · `bootstrap.py` | 启动引导与一键安装 |
| `install_engine.py` | 引擎多源下载与解压 |
| `inference.py` | 引擎 HTTP 客户端 |
| `text_processing.py` | 句子与长文本分段，避免逗号过度切分 |
| `engine_runtime.py` | Linux 引擎适配：特征缓存与中英短词停顿修正 |
| `theme.py` | 深浅双主题样式 |

Linux 下运行测试：`uv pip install --python .venv/bin/python -r requirements-dev.txt`，
然后执行 `.venv/bin/python -m unittest discover -s tests -v`。

## 免责声明

本软件仅供个人学习、技术研究与学术交流使用。用户需自行确保参考音频与生成内容的权利合规，并承担相应使用责任；严禁将生成的声音用于诈骗、仿冒、侵犯他人声音权 / 肖像权等非法用途。

## 致谢与许可

引擎基于 [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。

感谢 [@2710165659](https://github.com/2710165659) 贡献的 Premiere Pro UXP 插件支持。

本项目基于 [MIT](LICENSE) 许可开源；GPT-SoVITS 引擎本体遵循其上游许可。
