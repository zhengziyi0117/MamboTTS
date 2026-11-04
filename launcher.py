"""
MamboTTS 统一启动器
- 检测安装状态
- 未安装：弹窗询问 → 安装界面（4 阶段进度条 + 实时日志）
- 已安装：直接启动主界面 app.py
"""
import os
import sys
import re
import subprocess

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QMessageBox, QFrame, QDialog, QRadioButton
)
from PySide6.QtCore import QThread, Signal

from theme import THEME, header_qss
from install_engine import GPU_PACKAGES


# ============================================================
# 安装状态检测
# ============================================================

def _project_dir():
    return os.path.dirname(os.path.abspath(__file__))


INSTALL_MARKER = ".mambo_install_ok"


def is_installed():
    """
    判定引擎是否已安装：
    - 核心文件存在（api.py + runtime/python.exe）
    - 且满足其一：存在安装完成标记（新安装）/ runtime 内含 torch（老安装兼容）

    设计约束：
    1. 不能把 models/ 纳入判定——整合包不含曼波模型，模型缺失走安装流程
       会形成「装完依旧未安装→反复下载 8GB」的死循环；模型缺失由
       run_engine.preflight_check 在引擎启动阶段报错（GUI 日志可见）。
    2. 半途中断的解压可能留下 api.py + python.exe 但缺 torch，
       用标记/torch 双重校验识别「半成品」。
    """
    base = _project_dir()
    if sys.platform != "win32":
        from run_engine import engine_paths
        paths = engine_paths(base)
        return all(os.path.exists(paths[key]) for key in ("api_py", "python_exe")) and os.path.isfile(
            os.path.join(paths["gsv_dir"], INSTALL_MARKER))
    gsv_dir = os.path.join(base, "GPT-SoVITS")
    api_py = os.path.join(gsv_dir, "api.py")
    runtime_python = os.path.join(gsv_dir, "runtime", "python.exe")
    if not (os.path.exists(api_py) and os.path.exists(runtime_python)):
        return False
    if os.path.exists(os.path.join(gsv_dir, INSTALL_MARKER)):
        return True
    # 老安装（标记机制引入前完成）：用 runtime 内 torch 目录作为完整性代理
    torch_dir = os.path.join(gsv_dir, "runtime", "Lib", "site-packages", "torch")
    return os.path.isdir(torch_dir)


# ============================================================
# 显卡型号选择对话框
# ============================================================

def detect_gpu_type():
    """用 nvidia-smi 探测本机 NVIDIA 显卡，返回建议的整合包类型。

    命中 RTX 50 系 → 'nvidia50'，其余 NVIDIA → 'general'；
    探测失败（无 nvidia-smi/无 A 卡等）返回 None，交由默认值兜底。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=6,
        ).decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if re.search(r"RTX\s*50\d0|50\d0\s+Ti|B\d\d\d", out, re.IGNORECASE):
        return "nvidia50"
    if "NVIDIA" in out or "RTX" in out or "GTX" in out:
        return "general"
    return None


class GpuSelectionDialog(QDialog):
    """让用户选择显卡型号，决定下载哪个整合包"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择显卡型号")
        # 不用 setFixedSize：高 DPI 缩放下固定尺寸会裁切文案，最小尺寸 + 布局自适应更安全
        self.setMinimumSize(520, 380)
        self.resize(520, 400)
        # 默认预选；探测到本机显卡后自动选中更贴合的档位（仍可手动改）。
        # 探测失败（无 nvidia-smi / 核显 / AMD）时默认 general 而非 nvidia50：
        # general 兼容面更大，避免 30/40 系或无卡用户被推错 50 系专用包
        detected = detect_gpu_type()
        self._gpu_type = detected or "general"
        self._build_ui(default_gpu=detected)

    def _build_ui(self, default_gpu=None):
        self.setStyleSheet(header_qss())
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("请选择您的显卡型号")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {THEME['text']};")
        layout.addWidget(title)

        n50 = GPU_PACKAGES["nvidia50"]
        gen = GPU_PACKAGES["general"]
        n50_gb = n50["size_bytes"] / (1024 ** 3)
        gen_gb = gen["size_bytes"] / (1024 ** 3)
        # 与 install_engine 磁盘预检同一公式（整包 ×3.6 含解压产物），
        # 文案随实际包大小自动更新，不再写死数字
        need_gb = max(n50["size_bytes"], gen["size_bytes"]) * 3.6 / (1024 ** 3)

        desc = QLabel(
            "不同显卡需要下载对应的 GPT-SoVITS 整合包，选错可能导致无法正常推理。\n"
            f"两种整合包体积分别约 {n50_gb:.1f}GB / {gen_gb:.1f}GB，安装前请确认\n"
            f"磁盘剩余空间 ≥ {need_gb:.0f}GB（含下载与解压产物）且网络畅通。"
        )
        desc.setStyleSheet(f"color: {THEME['dim']}; font-size: 12px;")
        layout.addWidget(desc)

        self.rb_nvidia50 = QRadioButton(
            f"NVIDIA 50 系显卡 (RTX 5070 / 5080 / 5090 等) —— 约 {n50_gb:.1f}GB"
        )
        self.rb_other = QRadioButton(
            f"NVIDIA 30 系 / 入门 40 系显卡 (RTX 3060 / 4060 等) —— 约 {gen_gb:.1f}GB"
        )
        # 按探测结果预选（探测失败默认 general 兼容包），并明确告知用户
        self.rb_nvidia50.setChecked(self._gpu_type == "nvidia50")
        self.rb_other.setChecked(self._gpu_type == "general")
        layout.addWidget(self.rb_nvidia50)
        layout.addWidget(self.rb_other)

        if default_gpu:
            hint = QLabel("● 已通过 nvidia-smi 自动识别并预选，如与实际不符可手动改。")
            hint.setStyleSheet(f"color: {THEME['muted']}; font-size: 12px;")
        else:
            hint = QLabel("○ 未能自动识别显卡，请确认型号后手动选择。")
            hint.setStyleSheet(f"color: {THEME['muted']}; font-size: 12px;")
        layout.addWidget(hint)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_cancel = QPushButton("取消")
        btn_cancel.setObjectName("SecondaryBtn")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("确认下载")
        self.btn_ok = btn_ok
        btn_ok.clicked.connect(self._on_confirm)
        # 按钮上明示所选包大小：把最终确认信息推到离点击最近的位置
        self.rb_nvidia50.toggled.connect(self._update_ok_label)
        self._update_ok_label()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        layout.addLayout(btn_layout)

    def _update_ok_label(self):
        pkg = GPU_PACKAGES["nvidia50" if self.rb_nvidia50.isChecked() else "general"]
        gb = pkg["size_bytes"] / (1024 ** 3)
        self.btn_ok.setText(f"确认下载（约 {gb:.1f}GB）")

    def _on_confirm(self):
        if self.rb_nvidia50.isChecked():
            self._gpu_type = "nvidia50"
        else:
            self._gpu_type = "general"
        self.accept()

    def get_gpu_type(self):
        return self._gpu_type


# ============================================================
# 安装线程：后台执行 install_engine.install()
# ============================================================

class InstallWorker(QThread):
    """后台安装线程，避免阻塞 GUI。

    协作式取消：request_cancel() 只置标志，install() 在阶段边界检查。
    9GB 下载中途被强杀会留下 .incomplete 缓存，重新安装可续传。
    """
    progress_signal = Signal(float)         # 整体进度 0.0~1.0
    log_signal = Signal(str)                # 日志文本
    state_signal = Signal(str)              # 当前阶段名
    finished_signal = Signal(bool, str)     # (success, message)

    def __init__(self, gpu_type="nvidia50"):
        super().__init__()
        self.gpu_type = gpu_type
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def is_cancel_requested(self):
        return self._cancel

    def run(self):
        # 延迟导入：装依赖/报错都在工作线程里处理，不阻塞主线程启动
        try:
            from install_engine import install
        except Exception as e:
            self.finished_signal.emit(False, f"加载安装模块失败: {e}")
            return

        success, message = install(
            progress_cb=lambda p: self.progress_signal.emit(p),
            log_cb=lambda m: self.log_signal.emit(m),
            state_cb=lambda s: self.state_signal.emit(s),
            gpu_type=self.gpu_type,
            cancel_cb=self.is_cancel_requested,
        )
        # 成功但取消标志恰好置位的竞态：不往成功消息上挂「已取消」前缀
        if self._cancel and not success:
            message = "安装已取消：" + message
        self.finished_signal.emit(success, message)


# ============================================================
# 安装界面
# ============================================================

class InstallerWindow(QWidget):
    """安装界面：分阶段进度条 + 实时日志"""

    def __init__(self, on_done_callback=None, gpu_type="nvidia50"):
        super().__init__()
        self.on_done_callback = on_done_callback
        self.gpu_type = gpu_type
        self._worker = None
        self._cancel_requested = False
        self._close_attempts = 0
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("MamboTTS 引擎安装")
        self.resize(700, 560)
        self.setMinimumSize(600, 480)

        # 与主界面共享同一套 Catppuccin 主题（theme.py 单一定义源）
        self.setStyleSheet(header_qss())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Header（扁平深灰卡片，白竖条品牌高光）
        header = QFrame()
        header.setObjectName("HeaderCard")
        header.setFixedHeight(64)

        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 10, 16, 10)
        h_layout.setSpacing(12)
        accent_bar = QFrame()
        accent_bar.setObjectName("AccentBar")
        accent_bar.setFixedWidth(3)
        h_layout.addWidget(accent_bar)
        h_text = QVBoxLayout()
        h_text.setSpacing(2)
        title = QLabel("MamboTTS 引擎安装")
        title.setStyleSheet(f"color: {THEME['text']}; font-size: 17px; font-weight: 800; background: transparent;")
        pkg_gb = GPU_PACKAGES.get(self.gpu_type, GPU_PACKAGES["nvidia50"])["size_bytes"] / (1024 ** 3)
        subtitle = QLabel(f"首次使用需下载 GPT-SoVITS 整合包（约 {pkg_gb:.1f}GB），请保持网络畅通")
        subtitle.setStyleSheet(f"color: {THEME['muted']}; font-size: 12px; background: transparent;")
        h_text.addWidget(title)
        h_text.addWidget(subtitle)
        h_layout.addLayout(h_text)
        h_layout.addStretch()
        layout.addWidget(header)

        # 当前阶段
        self.stage_label = QLabel("准备中...")
        self.stage_label.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {THEME['text']};"
        )
        layout.addWidget(self.stage_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 日志面板
        log_label = QLabel("安装日志:")
        layout.addWidget(log_label)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        # 限制最大行数：安装日志动辄上万行，无上限会持续吃内存
        self.log_view.document().setMaximumBlockCount(5000)
        layout.addWidget(self.log_view, 1)

        # 按钮栏
        btn_layout = QHBoxLayout()
        self.btn_install = QPushButton("开始安装")
        self.btn_install.setFixedHeight(40)
        self.btn_install.clicked.connect(self.start_install)
        self.btn_close = QPushButton("退出")
        self.btn_close.setObjectName("SecondaryBtn")
        # 固定最小宽度：取消流程中文案长度变化会引起按钮/布局跳动
        self.btn_close.setMinimumWidth(150)
        self.btn_close.setFixedHeight(40)
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_install, 2)
        btn_layout.addWidget(self.btn_close, 1)
        layout.addLayout(btn_layout)

    def start_install(self):
        """启动后台安装线程"""
        # 重试安装时重置取消状态：否则上次取消留下的 _close_attempts>=1
        # 会让本次下载中第一次点 X 就直接进入"强制终止"分支
        self._close_attempts = 0
        self._cancel_requested = False
        self.btn_close.setText("退出")
        self.btn_close.setEnabled(True)
        self.btn_install.setEnabled(False)
        self.btn_install.setText("◐ 安装进行中...")
        self.progress_bar.setValue(0)

        self._worker = InstallWorker(gpu_type=self.gpu_type)
        self._worker.progress_signal.connect(self.on_progress)
        self._worker.log_signal.connect(self.on_log)
        self._worker.state_signal.connect(self.on_state)
        self._worker.finished_signal.connect(self.on_finished)
        self._worker.start()

    def on_progress(self, p):
        self.progress_bar.setValue(int(p * 100))

    def on_log(self, msg):
        # 日志：只有级别标签着色，正文保持浅灰（避免整行飘红刺眼）；
        # 行首用 ASCII 符号。msg 自带 [状态]/[错误] 前缀 → 识别后剥离，避免标签重复
        tag, tag_color = "", ""
        body = msg
        for t, c in (("[错误]", THEME["danger"]), ("[警告]", THEME["warning"]),
                     ("[状态]", THEME["border_active"]), ("[提示]", THEME["dim"])):
            if body.startswith(t):
                tag, tag_color = t, c
                body = body[len(t):].lstrip()
                break
        safe_body = (body.replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
        self.log_view.append(
            f'<span style="color:{THEME["dim"]};">· </span>'
            f'<span style="color:{tag_color}; font-weight:bold;">{tag}</span> '
            f'<span style="color:{THEME["muted"]};">{safe_body}</span>'
        )
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_state(self, stage):
        self.on_log(f"当前阶段: {stage}")

    def on_finished(self, success, message):
        self._cancel_requested = False
        if success:
            self.stage_label.setText("✓ 安装完成")
            self.stage_label.setStyleSheet(
                f"font-size: 13px; font-weight: 700; color: {THEME['text']};")
            self.btn_install.setText("完成")
            QMessageBox.information(self, "安装完成", "GPT-SoVITS 引擎安装成功！\n即将启动 MamboTTS 主界面。")
            # 先关闭安装器再开主窗口：顺序反了会出现两窗共存一帧的闪烁/焦点错乱
            self.close()
            if self.on_done_callback:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, self.on_done_callback)
        else:
            # 取消判定走显式标志而非字符串猜测：install() 各阶段取消时的
            # 返回文案并非都以「已取消」开头，字符串匹配会把取消误报成失败
            cancelled = self._worker is not None and self._worker.is_cancel_requested()
            self.stage_label.setText("已取消" if cancelled else "✕ 安装失败")
            # 反白（白底黑字）只给错误；取消用灰字
            self.stage_label.setStyleSheet(
                f"font-size: 13px; font-weight: 700; color: {THEME['accent_text']};"
                f"background-color: {THEME['accent']}; border-radius: 6px; padding: 2px 10px;"
                if not cancelled else
                f"font-size: 13px; font-weight: 400; color: {THEME['muted']};"
            )
            self.btn_install.setEnabled(True)
            self.btn_install.setText("重新尝试安装")
            if cancelled:
                self.on_log("安装已取消。已下载的部分会保留，重试安装可续传。")
            else:
                # 原始异常可能是一大段英文 traceback，小白无法据此行动：
                # 弹窗给常见原因清单，原文只进日志面板（截断显示）
                brief = message if len(message) <= 300 else message[:300] + "…"
                self.on_log(f"[错误] 安装失败详情：{message}")
                QMessageBox.critical(
                    self, "安装失败",
                    "安装未完成。常见原因：\n"
                    "1. 磁盘空间不足（需预留约 30GB+）\n"
                    "2. 网络中断或镜像源不可达（重试可续传）\n"
                    "3. 杀毒软件拦截了解压/pip\n\n"
                    f"错误摘要：\n{brief}\n\n完整信息见上方安装日志。"
                )

    def closeEvent(self, event):
        """安装进行中不直接关闭（会留下半成品）。

        三段式退出：第 1 次询问确认 → 请求协作式取消并保持窗口；
        第 2 次提示「正在等待收尾」；第 3 次强杀线程并接受关闭。
        任何一次都不会出现「点了退不出去也没反应」的死路。
        """
        if self._worker and self._worker.isRunning():
            self._close_attempts += 1
            if self._close_attempts == 1:
                reply = QMessageBox.question(
                    self, "确认取消",
                    "安装正在进行中。\n取消会停止安装，已下载部分将保留（重试自动续传）。\n"
                    "下载阶段取消在数秒内生效。\n确定要取消吗？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    event.ignore()
                    return
                self._cancel_requested = True
                self._worker.request_cancel()
                self.btn_install.setText("◐ 正在等待取消生效...")
                self.btn_close.setText("正在取消…再点「退出」可强制结束")
                event.ignore()
                return
            if self._close_attempts == 2:
                QMessageBox.warning(
                    self, "等待收尾",
                    "已请求取消，正在等待安装线程收尾（通常数秒内）。\n"
                    "再点一次「退出」将立即强制终止。"
                )
                self.btn_close.setText("⚠ 再点一次「退出」立即强制终止")
                event.ignore()
                return
            # 第 3 次：强制退出
            try:
                self._worker.terminate()
                self._worker.wait(3000)
            except Exception:
                pass
        event.accept()


# ============================================================
# 主入口
# ============================================================

# 持有主窗口引用防止被 GC（Qt 对象在无 Python 引用时会被销毁）
_main_window_holder = []


def _show_main_window():
    """创建并显示主界面（同进程内 import，避免再起子进程）"""
    from app import MamboTTSApp
    win = MamboTTSApp()
    win.show()
    _main_window_holder.append(win)
    return win


def main():
    app = QApplication(sys.argv)
    # 应用用户持久化的主题偏好（dark/light），安装窗/对话框跟随
    try:
        from theme import apply_theme
        from config_store import load as _cfg_load
        cfg = _cfg_load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
        apply_theme(cfg["theme"])
    except Exception:
        pass  # 主题读取失败不影响启动（默认深色）

    if is_installed():
        # 已安装：直接进主界面
        _show_main_window()
        sys.exit(app.exec())

    if sys.platform != "win32":
        QMessageBox.information(None, "安装 Linux 引擎",
                                "请先在项目目录运行 ./install.sh --backend rocm --desktop。\n"
                                "安装后使用 ./run.sh --desktop 启动客户端，或 ./run.sh 启动 API。")
        return

    # 未安装：弹窗提示（用最大包估算，避免提示偏小导致中途盘满）
    max_need_gb = max(p["size_bytes"] for p in GPU_PACKAGES.values()) * 3.6 / (1024 ** 3)
    reply = QMessageBox.question(
        None, "首次使用",
        "检测到 GPT-SoVITS 引擎尚未安装。\n\n"
        "安装需要下载约 8.2GB 的整合包（国内多源镜像，自动选最快），\n"
        f"并预留约 {max_need_gb:.0f}GB 磁盘空间（含下载与解压产物）。\n"
        "是否立即开始安装？",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
    )
    if reply == QMessageBox.No:
        sys.exit(0)

    # 选择显卡型号，决定下载哪个整合包
    gpu_dialog = GpuSelectionDialog()
    if gpu_dialog.exec() != QDialog.Accepted:
        sys.exit(0)
    gpu_type = gpu_dialog.get_gpu_type()

    # 启动安装界面；安装完成后同进程直接切换到主界面
    installer = InstallerWindow(gpu_type=gpu_type)

    def on_done():
        _show_main_window()

    installer.on_done_callback = on_done
    installer.show()
    # 选完显卡后直接开始下载，无需再手动点击"开始安装"
    installer.start_install()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
