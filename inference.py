import os
import logging
import requests

from engine_contract import (
    SUPPORTED_LANGS, DEFAULT_REF_TEXT, CUT_PUNC, DEFAULT_API_URL, REF_WAV,
)  # 契约常量单一来源（engine_contract.py），这里透传保持旧引用兼容
from text_processing import split_speech_text

logger = logging.getLogger("mambotts.inference")


def _is_japanese(t):
    """检测文本是否含日语假名（平假名/片假名）"""
    for char in t:
        cp = ord(char)
        if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
            return True
    return False


def _looks_like_wav(data: bytes) -> bool:
    """WAV 结构校验：防止把 HTTP 200 的 JSON 错误页存成 .wav。

    按 RIFF chunk 链逐块遍历，确认存在 data 块且其声明长度 > 0。
    只读文件头即可判定（流式落盘后不可能拿全量做长度自洽校验）：
    JSON 错误页 / 截断假头在此都会被拒绝，正常 WAV 头在几百字节内必含 data 块。"""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    pos = 12
    # 只看头 8KB 内的 chunk 链：正常 WAV 的 fmt/data 声明都在文件最前面
    limit = min(len(data), 8192)
    while pos + 8 <= limit:
        chunk_id = data[pos:pos + 4]
        try:
            chunk_size = int.from_bytes(data[pos + 4:pos + 8], "little")
        except Exception:
            return False
        if chunk_id == b"data":
            # 音频数据必须真实存在：声明非空即认可（完整性交给上层可选的大小校验）
            return chunk_size > 0
        # 跳过整个 chunk（可能很大，比如 LIST 头）——跳到下一块
        pos += 8 + chunk_size + (chunk_size & 1)
    # 8KB 内没找到 data 块：可疑（正常 WAV 不会这样）
    return False


class TTSClient:
    def __init__(self, api_url=DEFAULT_API_URL):
        self.api_url = api_url
        # GUI 日志回调：由 app.py 注入（须线程安全——合成在后台线程进行）
        self._log_handler = None
        # 默认参考音频：路径相对于本文件所在目录
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_ref_audio = os.path.join(_base_dir, "models", REF_WAV)
        self.default_ref_text = DEFAULT_REF_TEXT

    def set_log_handler(self, handler):
        """注入日志回调，handler 签名为 handler(level: str, message: str)。
        注意：该回调会从后台合成线程调用，实现方必须自行 marshal 回 GUI 线程。"""
        self._log_handler = handler

    def _log(self, level, message):
        """同时输出到 logger 与 GUI 日志回调"""
        getattr(logger, level.lower(), logger.info)(message)
        if self._log_handler:
            try:
                self._log_handler(level, message)
            except Exception:
                # 日志回调自身异常不应影响主流程
                pass

    def is_api_running(self, timeout=1.5, api_url=None):
        """检查本地 GPT-SoVITS API 服务是否在线。

        判定依据：GPT-SoVITS（FastAPI）对 GET /control 的任何响应——正常
        JSON、参数校验 422、路由 404——响应体都是 JSON（无参数时甚至就是
        字面量 null）。因此「能拿到可解析的 JSON」即可确认端口上是引擎；
        若返回 HTML/明文（反代、其他本地服务占端口），判为离线并记日志，
        避免后续所有合成请求都报莫名其妙的错。"""
        base_url = (api_url or self.api_url).rstrip("/")
        try:
            # 强制直连、绕过系统代理：本机 Clash/ghost 等代理工具会劫持 127.0.0.1
            # 流量，长文本合成期间代理空闲超时会把连接掐断，导致
            # "ChunkedEncodingError: Response ended prematurely"。
            response = requests.get(f"{base_url}/control", timeout=timeout,
                                    proxies={"http": None, "https": None})
        except requests.RequestException:
            return False
        try:
            response.json()
            if getattr(self, "_non_json_warned", False):
                self._non_json_warned = False  # 引擎恢复，允许下次异常再提示一次
            return True
        except ValueError:
            # 只警告一次，避免 10s 心跳轮询被同一句刷屏淹没关键日志；
            # 状态翻转（JSON→非JSON）时才再提示
            if not getattr(self, "_non_json_warned", False):
                self._non_json_warned = True
                self._log("WARNING",
                          f"端口 {base_url} 有 HTTP 响应但内容不是 JSON，"
                          f"不像 GPT-SoVITS 引擎（可能被其他程序占用），按离线处理。")
            return False

    @staticmethod
    def _check_writable_dir(save_path):
        """写文件前预检：父目录必须存在且可写。返回错误信息或 None。"""
        parent_dir = os.path.dirname(save_path) or "."
        if not os.path.exists(parent_dir):
            return f"输出目录不存在：{parent_dir}\n请点击「浏览...」重新选择保存路径。"
        if not os.path.isdir(parent_dir):
            return f"输出路径不是文件夹：{parent_dir}"
        if not os.access(parent_dir, os.W_OK):
            return f"没有写入权限的目录：{parent_dir}\n请选择其他位置，或用管理员身份运行。"
        return None

    def generate_speech(self, text, save_path, ref_audio="", prompt_text="",
                        text_lang="zh", prompt_lang="zh", speed=1.0, api_url=None,
                        segmentation="sentence"):
        """
        调用 GPT-SoVITS 本地 API 生成语音。
        api_url: 显式快照，避免合成进行中主线程修改 self.api_url 引发竞态。
        返回: (success: bool, message: str)
        """
        if not save_path:
            return False, "保存路径为空，请选择有效的输出路径。"

        err = self._check_writable_dir(save_path)
        if err:
            return False, err

        try:
            segments = split_speech_text(text, segmentation)
        except ValueError as exc:
            return False, str(exc)
        if not segments:
            return False, "请输入包含文字的配音文案。"

        base_url = (api_url or self.api_url).rstrip("/")
        online = self.is_api_running(api_url=base_url)
        if not online:
            self._log("WARNING", "GPT-SoVITS API 处于离线状态，无法合成。")
            return self._offline_failure(save_path)

        # 自动检测语种：如果文本中包含日语平假名或片假名，自动切换为日语合成
        if text_lang == "zh" and _is_japanese(text):
            text_lang = "ja"
            self._log("INFO", "自动检测文案为日语，已切换为日语合成 (ja)")
        if prompt_text and prompt_lang == "zh" and _is_japanese(prompt_text):
            prompt_lang = "ja"

        # 客户端侧语种预校验：避免传入不支持的语种触发服务端 KeyError → 500
        if text_lang not in SUPPORTED_LANGS:
            return False, f"不支持的文本语种: {text_lang}\n支持的语种: zh/en/ja/ko/yue 等"
        if prompt_lang not in SUPPORTED_LANGS:
            return False, f"不支持的参考音频语种: {prompt_lang}\n支持的语种: zh/en/ja/ko/yue 等"

        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0

        # GPT-SoVITS api.py 字段名（旧版）: refer_wav_path / prompt_language /
        # text_language / speed / cut_punc
        # 客户端已规划段落，不再让引擎按逗号二次切分。
        # 上游 HTTP 默认 top_p/temperature=1，使用较保守的采样降低韵律波动。
        params = {
            "text": "\n".join(segments),
            "text_language": text_lang,
            "speed": speed,
            "cut_punc": "",
            "top_k": 15,
            "top_p": 0.6,
            "temperature": 0.6,
        }

        # 参考音频：优先用调用方传入的，否则用默认 refer.wav
        effective_ref_audio = ref_audio if ref_audio else self.default_ref_audio
        effective_ref_text = prompt_text if prompt_text else self.default_ref_text
        if effective_ref_audio and effective_ref_text:
            params["refer_wav_path"] = effective_ref_audio
            params["prompt_text"] = effective_ref_text
            params["prompt_language"] = prompt_lang

        # 读超长按文本量伸缩：约每字 1 秒，下限 120s，上限 600s。
        # 长文案在消费级 GPU 上合成远超 60 秒，固定超时必然误报。
        default_timeout = 120.0 if os.name == "nt" else 600.0
        try:
            configured_timeout = float(os.environ.get("MAMBO_TTS_TIMEOUT", default_timeout))
            if not 1 <= configured_timeout <= 86400:
                raise ValueError
        except ValueError:
            configured_timeout = default_timeout
        read_timeout = max(configured_timeout, min(600.0, len(text) * 1.0))

        try:
            self._log("INFO", f"正在合成 {len(segments)} 段文案（{segmentation}）: {text[:20]}...")
            # POST / + JSON body：与 GET 字段完全一致（见 api.py tts_endpoint）。
            # 不用 GET 是避免长文案 URL 编码后超长（h11 行上限 16KB，
            # 约 600 个汉字即触发 414/400）。
            # proxies=None 强制直连：本机代理工具（Clash/ghost 等）会劫持
            # 127.0.0.1 流量并在长合成期间掐断连接（ChunkedEncodingError）。
            # stream=True + iter_content 边下边写：几十 MB 的 WAV 不必整体进内存
            with requests.post(f"{base_url}/", json=params, stream=True,
                               timeout=(10, read_timeout),
                               proxies={"http": None, "https": None}) as response:
                if response.status_code == 200:
                    try:
                        with open(save_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=1024 * 256):
                                if chunk:
                                    f.write(chunk)
                    except PermissionError:
                        return False, f"文件被占用或无写入权限：{save_path}\n请关闭正在使用该文件的程序后重试。"
                    except OSError as e:
                        return False, f"保存文件失败：{e}\n路径：{save_path}"
                    # 写完再校验 WAV 结构：流式模式下无法提前看 content
                    try:
                        with open(save_path, "rb") as f:
                            head = f.read(8192)
                    except OSError:
                        head = b""
                    if not _looks_like_wav(head):
                        snippet = f"(文件大小 {os.path.getsize(save_path)} 字节)"
                        self._log("ERROR", f"API 返回 200 但内容不是有效 WAV：{snippet}")
                        try:
                            os.remove(save_path)  # 别留坏文件
                        except OSError:
                            pass
                        return False, "引擎返回了无效音频数据（状态码 200）。\n请查看引擎日志排查模型/参考音频配置。"
                    self._log("INFO", f"语音合成成功，已保存至: {save_path}")
                    return True, f"语音合成成功，已保存至: {save_path}"
                else:
                    # 把后端返回的错误体也带给用户，便于排查
                    err_body = response.text[:300] if response.text else "(无响应内容)"
                    self._log("ERROR", f"API 响应失败，状态码: {response.status_code}, 内容: {err_body}")
                    return False, f"API 返回错误（状态码 {response.status_code}）：\n{err_body}"

        # 注：ChunkedEncodingError 与 ConnectionError 是 RequestException 的并列子类，
        # 并非继承关系，先捕获哪个都不影响可达性；放前面只为可读性。
        except requests.exceptions.ChunkedEncodingError:
            # 响应中途断连：请求已是直连模式仍中断，多为引擎侧崩溃/OOM。
            # 流式写盘可能已落下半截 WAV，清掉避免留下坏文件
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except OSError:
                pass
            self._log("ERROR", "API 响应中途断开（ChunkedEncodingError）：引擎可能在长文本推理中崩溃。")
            return False, ("引擎在合成中途断开了连接（直连模式下仍中断，多半是长文本推理时"
                           "显存不足或进程崩溃）。\n建议缩短文案分段合成，并查看引擎日志定位。")
        except requests.exceptions.ConnectionError:
            return False, f"无法连接到 API 服务：{base_url}\n引擎可能正在退出，请等待引擎状态变绿后重试。"
        except requests.exceptions.Timeout:
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except OSError:
                pass
            return False, f"API 请求超时（{int(read_timeout)}秒），文案过长或显卡负载过高，可分段合成后拼接。"
        except Exception as e:
            self._log("ERROR", f"调用 API 异常: {str(e)}")
            return False, f"调用 API 异常：{type(e).__name__}: {e}"

    def _offline_failure(self, save_path):
        """引擎离线：不写任何文件，仅校验目录可写并给出可操作的提示。
        历史版本会写 1 秒静音 WAV 冒充成功，导致目录里堆积垃圾文件。"""
        err = self._check_writable_dir(save_path)
        if err:
            return False, err
        self._log("WARNING", f"引擎离线，未生成文件，目标目录可写：{os.path.dirname(save_path) or '.'}")
        return False, ("本地 GPT-SoVITS 引擎尚未就绪，无法合成配音。\n"
                       "引擎通常在客户端启动后 10-30 秒内自动上线（看顶部状态灯）；\n"
                       "若状态灯显示失败/超时，可点击旁边的「↻ 重启引擎」；\n"
                       "若从未安装过引擎，请通过 MamboTTS.bat 完成一键安装。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    client = TTSClient()
    print("API 是否运行:", client.is_api_running())
    ok, msg = client.generate_speech("测试一下引擎离线时的失败提示", "test.wav")
    print(f"结果: success={ok}, message={msg}")
