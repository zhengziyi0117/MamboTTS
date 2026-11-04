"""MamboTTS ↔ GPT-SoVITS 引擎契约常量（单一事实来源）。

本模块只放「与引擎约定相关、GUI 无关」的常量，被 inference.py（HTTP 客户端）
与 run_engine.py（引擎启动器）共同引用。此前 DEFAULT_REF_TEXT 定义在
inference.py 里、被 run_engine.py 反向 import，容易让人误以为启动器依赖 HTTP 层；
抽出来之后依赖方向回归正常。

注意：GPT-SoVITS 目录是云端下载的整合包，本文件不修改它，只约定调用契约。
"""

# GPT-SoVITS api.py 的 dict_language 支持的语种，用于客户端侧预校验，
# 避免传入不支持的语种触发服务端 KeyError → HTTP 500（对应上游 B10 防御）
SUPPORTED_LANGS = {
    "zh", "en", "ja", "ko", "yue", "auto", "auto_yue",
    "all_zh", "all_yue", "all_ja", "all_ko",
    "中文", "粤语", "英文", "日文", "韩文",
    "中英混合", "粤英混合", "日英混合", "韩英混合",
    "多语种混合", "多语种混合(粤语)",
}

# 默认参考音频转写文本：必须与 models/refer.wav 内容一字不差。
# 仅在此处定义一份，inference/run_engine/install_engine 从这里导入，
# 避免多处副本漂移（ref_text 与音频不匹配会静默劣化克隆音色）。
DEFAULT_REF_TEXT = "大家好，欢迎来到我的频道，今天给大家分享一个有趣的内容"

# 曼波音色模型清单（相对 models/ 目录的文件名）。
# run_engine 启动、install_engine 生成 go-api-mambo.bat、launcher 安装检测
# 都从这里取，消灭「模型改名要改三处」的隐患。
SOVITS_MODEL = "manbo_e8_s168.pth"
GPT_MODEL = "manbo-e10.ckpt"
REF_WAV = "refer.wav"

# 引擎 API 服务地址（本机回环，固定端口）
API_HOST = "127.0.0.1"
API_PORT = 9880
DEFAULT_API_URL = f"http://{API_HOST}:{API_PORT}"

# 旧版逐标点切分清单，保留供外部调用方引用。
# 当前客户端通过 text_processing 规划段落，向引擎传 cut_punc=""。
CUT_PUNC = "，。？！；：、…,.;?!"
