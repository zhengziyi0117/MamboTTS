"use strict";

const $ = (id) => document.getElementById(id);
const form = $("speech-form"),
  text = $("speech-text"),
  speed = $("speed"),
  language = $("language"),
  segmentation = $("segmentation");
const button = $("generate-button"),
  audio = $("audio-player"),
  canvas = $("waveform");
let online = false,
  busy = false,
  records = [],
  selected = null,
  peaks = null;
let clock = null,
  started = 0,
  waveformVersion = 0;

const storage = {
  get(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {}
  },
  remove(key) {
    try {
      localStorage.removeItem(key);
    } catch {}
  },
};
function duration(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
function updateForm() {
  $("text-count").textContent = text.value.length.toLocaleString("en-US");
  $("speed-value").textContent = `${Number(speed.value).toFixed(2)}×`;
  $("segmentation-hint").textContent =
    segmentation.value === "sentence"
      ? "保留句内上下文；需要停下来时，可以手动换行。"
      : "逗号会分开合成，适合需要明显停顿的文案。";
  button.disabled = busy || !online || !text.value.trim();
  button.title = busy
    ? "配音正在生成"
    : !online
      ? "正在等待引擎就绪"
      : "Ctrl + Enter 生成配音";
  storage.set(
    "mambo-draft",
    JSON.stringify({
      text: text.value,
      speed: speed.value,
      language: language.value,
      segmentation: segmentation.value,
    }),
  );
}
function error(message = "") {
  $("error-message").textContent = message;
  $("error-message").hidden = !message;
}
function setBusy(value) {
  busy = value;
  button.classList.toggle("is-working", value);
  button.setAttribute("aria-busy", String(value));
  $("generate-label").textContent = value ? "正在生成" : "生成配音";
  $("progress-note").hidden = !value;
  if (clock) clearInterval(clock);
  if (value) {
    started = Date.now();
    $("elapsed-time").textContent = "00:00";
    clock = setInterval(() => {
      $("elapsed-time").textContent = duration((Date.now() - started) / 1000);
    }, 1000);
  }
  updateForm();
}
async function request(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : "请检查文案、语言和语速后重试";
      const err = new Error(detail);
      err.status = response.status;
      throw err;
    }
    return data;
  } catch (err) {
    if (err.name === "AbortError" || err instanceof TypeError)
      throw new Error("连接中断，请检查工作台是否仍在运行");
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}
async function checkStatus() {
  try {
    const status = await request("/api/status");
    online = status.state === "online";
    $("engine-state").dataset.state = status.state;
    $("engine-label").textContent = online
      ? "本地引擎已就绪"
      : status.state === "loading"
        ? "引擎正在准备"
        : "引擎离线 · 请查看启动终端";
  } catch {
    online = false;
    $("engine-state").dataset.state = "offline";
    $("engine-label").textContent = "工作台连接中断";
  }
  updateForm();
  setTimeout(checkStatus, online ? 5000 : 2000);
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}
function renderHistory() {
  const list = $("history-list");
  list.replaceChildren();
  $("history-count").textContent = String(records.length).padStart(2, "0");
  if (!records.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "还没有生成记录。写一段文案，试试曼波的声音。";
    list.append(empty);
    return;
  }
  records.forEach((record, index) => {
    const row = document.createElement("div");
    row.className = "history-row";
    row.classList.toggle("is-active", selected?.id === record.id);
    const number = document.createElement("span");
    number.className = "history-index";
    number.textContent = String(index + 1).padStart(2, "0");
    const play = document.createElement("button");
    play.type = "button";
    play.className = "history-play";
    play.setAttribute("aria-label", `试听：${record.text}`);
    play.append(icon("play"));
    play.addEventListener("click", () => selectRecord(record, true));
    const info = document.createElement("div");
    info.className = "history-info";
    const title = document.createElement("button");
    title.type = "button";
    title.className = "history-text";
    title.textContent = record.text;
    title.addEventListener("click", () => selectRecord(record));
    const meta = document.createElement("div");
    meta.className = "history-meta";
    const date = new Date(record.created_at);
    meta.textContent = `${Number.isNaN(date.getTime()) ? "" : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}  ·  ${Number(record.speed).toFixed(2)}×  ·  WAV`;
    info.append(title, meta);
    const length = document.createElement("span");
    length.className = "history-duration";
    length.textContent = duration(record.duration);
    const download = document.createElement("a");
    download.className = "history-download";
    download.href = record.download_url;
    download.download = record.name;
    download.setAttribute("aria-label", `下载：${record.text}`);
    download.append(icon("download"));
    row.append(number, play, info, length, download);
    list.append(row);
  });
}
async function loadHistory() {
  records = await request("/api/history");
  renderHistory();
  if (!selected && records.length) selectRecord(records[0]);
}
async function selectRecord(record, play = false) {
  selected = record;
  peaks = null;
  const version = ++waveformVersion;
  audio.src = record.audio_url;
  audio.hidden = false;
  $("preview-empty").hidden = true;
  $("audio-surface").classList.add("has-audio");
  $("preview-text").textContent = record.text;
  $("preview-text").hidden = false;
  $("audio-number").textContent = duration(record.duration);
  $("audio-details").textContent =
    `${Number(record.speed).toFixed(2)}×  /  ${Number(record.duration).toFixed(1)} 秒`;
  $("preview-caption").textContent = "配音已就绪";
  $("download-button").href = record.download_url;
  $("download-button").download = record.name;
  $("download-button").hidden = false;
  renderHistory();
  drawWaveform();
  if (play)
    audio
      .play()
      .catch(() => error("浏览器暂未开始播放，请点击播放器的播放按钮"));
  let context;
  try {
    const response = await fetch(record.audio_url);
    if (!response.ok) throw new Error("音频文件已被移除，请重新生成");
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) return;
    context = new Context();
    const buffer = await context.decodeAudioData(await response.arrayBuffer());
    if (version !== waveformVersion) return;
    const channel = buffer.getChannelData(0),
      count = 80;
    const step = Math.max(1, Math.floor(channel.length / count));
    peaks = Array.from({ length: count }, (_, i) => {
      let peak = 0;
      for (let j = i * step; j < Math.min((i + 1) * step, channel.length); j++)
        peak = Math.max(peak, Math.abs(channel[j]));
      return peak;
    });
    drawWaveform();
  } catch (err) {
    if (version === waveformVersion)
      error(err.message || "波形暂时无法加载，仍可使用播放器试听");
  } finally {
    if (context) await context.close().catch(() => {});
  }
}
function drawWaveform() {
  const rect = canvas.getBoundingClientRect(),
    ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  if (!peaks) return;
  const max = Math.max(...peaks, 0.01),
    step = rect.width / peaks.length;
  const progress = audio.duration ? audio.currentTime / audio.duration : 0;
  peaks.forEach((peak, i) => {
    const height = Math.max(3, (peak / max) * rect.height * 0.8);
    ctx.fillStyle = i / peaks.length < progress ? "#ed673d" : "#a7b58f";
    ctx.fillRect(
      i * step,
      (rect.height - height) / 2,
      Math.max(1, step - 2),
      height,
    );
  });
}

async function pollJob(id) {
  try {
    const job = await request(`/api/jobs/${id}`);
    if (job.status === "done") {
      storage.remove("mambo-pending");
      setBusy(false);
      $("generation-status").textContent =
        `生成完成 · 用时 ${Number(job.elapsed).toFixed(1)} 秒`;
      selectRecord(job.record);
      loadHistory().catch((err) => error(err.message));
      return;
    }
    if (job.status === "error")
      throw new Error(job.error || "配音生成失败，请重试");
    $("progress-title").textContent =
      job.status === "queued" ? "任务正在排队" : "正在生成声音";
    $("generation-status").textContent = "";
    setTimeout(() => pollJob(id), 800);
  } catch (err) {
    if (err.message.startsWith("连接中断")) {
      $("generation-status").textContent = "连接中断，正在重试…";
      setTimeout(() => pollJob(id), 2000);
      return;
    }
    storage.remove("mambo-pending");
    setBusy(false);
    error(err.message);
  }
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !online || !text.value.trim()) return;
  error();
  $("generation-status").textContent = "";
  setBusy(true);
  try {
    const job = await request("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text.value.trim(),
        speed: Number(speed.value),
        language: language.value,
        segmentation: segmentation.value,
      }),
    });
    storage.set("mambo-pending", job.id);
    pollJob(job.id);
  } catch (err) {
    setBusy(false);
    error(err.message);
  }
});
text.addEventListener("input", updateForm);
speed.addEventListener("input", updateForm);
language.addEventListener("change", updateForm);
segmentation.addEventListener("change", updateForm);
text.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    form.requestSubmit();
  }
});
$("example-button").addEventListener("click", () => {
  text.value = "大家好，欢迎来到我的频道。今天我们来讲一个有趣的故事。";
  updateForm();
  text.focus();
});
$("reset-speed").addEventListener("click", () => {
  speed.value = "1";
  updateForm();
});
audio.addEventListener("timeupdate", drawWaveform);
audio.addEventListener("error", () => {
  if (audio.getAttribute("src"))
    error("音频加载失败，请重新选择记录或重新生成");
});
window.addEventListener("resize", drawWaveform);

try {
  const draft = JSON.parse(storage.get("mambo-draft") || "null");
  if (draft && typeof draft.text === "string") {
    text.value = draft.text.slice(0, 2000);
    speed.value = String(Math.min(2, Math.max(0.5, Number(draft.speed) || 1)));
    if (["zh", "en", "ja", "auto"].includes(draft.language))
      language.value = draft.language;
    if (["sentence", "punctuation"].includes(draft.segmentation))
      segmentation.value = draft.segmentation;
  }
} catch {}
updateForm();
checkStatus();
loadHistory().catch((err) => error(err.message));
const pending = storage.get("mambo-pending");
if (pending && /^[0-9a-f]{32}$/.test(pending)) {
  setBusy(true);
  pollJob(pending);
}
