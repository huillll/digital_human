#!/usr/bin/env python3
"""
CyberVerse 数字人演示
语音/文字输入 → LLM → TTS → FlashHead → 说话视频输出
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

# ─── 环境配置 ────────────────────────────────────────────────────────────────
os.environ.setdefault("NO_PROXY", "192.168.1.101,localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "192.168.1.101,localhost,127.0.0.1")

REPO_ROOT = Path("/home/test/CyberVerse/CyberVerse-main")
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
# Borrow torch / mediapipe from LatentSync conda env
_latentsync_site = "/home/test/.conda/envs/LatentSync/lib/python3.10/site-packages"
if _latentsync_site not in sys.path:
    sys.path.insert(1, _latentsync_site)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 输出目录（Gradio 从固定路径读取视频）
OUTPUT_DIR = Path("/home/test/CyberVerse/demo_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 克隆声音参考音频存放目录
CLONED_VOICES_DIR = Path("/home/test/CyberVerse/cloned_voices")
CLONED_VOICES_DIR.mkdir(parents=True, exist_ok=True)

# ─── 配置 ─────────────────────────────────────────────────────────────────────
LM_STUDIO_URL   = "http://192.168.1.101:1234/v1"
LM_STUDIO_MODEL = "holo-3.1-35b-a3b"
TTS_VOICE       = "zh-CN-XiaoxiaoNeural"
DEFAULT_AVATAR  = str(REPO_ROOT / "examples" / "girl.png")
SR_16K          = 16000

SYSTEM_PROMPT = (
    "你是一个友好、自然的中文数字人助手。回答简洁，口语化，适合语音播报。"
    "不超过100字。"
)

# ─── 后台 asyncio 事件循环（供 FlashHead 使用）──────────────────────────────
_bg_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
_bg_thread.start()


def _run(coro, timeout: float = 600.0):
    """在后台事件循环中运行协程，阻塞等待结果。"""
    fut = asyncio.run_coroutine_threadsafe(coro, _bg_loop)
    return fut.result(timeout=timeout)


# ─── 全局模型状态 ─────────────────────────────────────────────────────────────
_flash_plugin = None          # FlashHeadAvatarPlugin
_whisper_pipeline = None      # transformers ASR pipeline
_current_avatar: str = DEFAULT_AVATAR
_plugin_lock = threading.Lock()

WHISPER_MODEL = "/home/test/money_printer_turbo/MoneyPrinterTurbo/models/whisper-large-v3"
VOXCPM_MODEL  = "openbmb/VoxCPM2"   # HuggingFace model id

_voxcpm_model = None
_voxcpm_lock  = threading.Lock()


def _load_whisper():
    global _whisper_pipeline
    if _whisper_pipeline is not None:
        return _whisper_pipeline
    from faster_whisper import WhisperModel
    logger.info("加载 Whisper large-v3 (CPU int8) …")
    _whisper_pipeline = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    logger.info("Whisper 加载完成")
    return _whisper_pipeline


def _load_voxcpm():
    """懒加载 VoxCPM2，线程安全。需要 GPU (~8GB VRAM)。"""
    global _voxcpm_model
    if _voxcpm_model is not None:
        return _voxcpm_model
    with _voxcpm_lock:
        if _voxcpm_model is not None:
            return _voxcpm_model
        from voxcpm import VoxCPM
        logger.info("加载 VoxCPM2 声音克隆模型…")
        _voxcpm_model = VoxCPM.from_pretrained(VOXCPM_MODEL, load_denoiser=False)
        logger.info("VoxCPM2 加载完成")
    return _voxcpm_model


async def _init_flash_head(avatar_path: str):
    global _flash_plugin, _current_avatar
    from inference.core.config import load_config
    from inference.core.types import PluginConfig
    from inference.plugins.avatar.flash_head_plugin import FlashHeadAvatarPlugin

    raw = load_config(str(REPO_ROOT / "cyberverse_config.yaml"))
    avatar_cfg = raw["inference"]["avatar"]
    section = avatar_cfg["flash_head"]
    runtime = avatar_cfg.get("runtime", {})
    params = {k: v for k, v in {**runtime, **section}.items() if k != "plugin_class"}
    for key in ("checkpoint_dir", "wav2vec_dir", "models_dir"):
        if key in params and params[key]:
            p = Path(params[key])
            if not p.is_absolute():
                params[key] = str(REPO_ROOT / p)

    plugin = FlashHeadAvatarPlugin()
    logger.info("初始化 FlashHead …")
    await plugin.initialize(PluginConfig(plugin_name="avatar.flash_head", params=params))
    logger.info("设置头像: %s", avatar_path)
    await plugin.set_avatar(avatar_path, use_face_crop=False)
    _flash_plugin = plugin
    _current_avatar = avatar_path
    logger.info("FlashHead 就绪")


def _to_png_if_needed(path: str) -> str:
    """如果图片是 WebP 或其他格式，转成 PNG 再传给 FlashHead。"""
    if path.lower().endswith(".png"):
        return path
    import cv2
    img = cv2.imread(path)
    if img is None:
        return path  # 读取失败时原样返回，让 FlashHead 自己报错
    fd, png_path = tempfile.mkstemp(suffix=".png", prefix="cv_avatar_")
    os.close(fd)
    cv2.imwrite(png_path, img)
    return png_path


def _ensure_flash_head(avatar_path: str | None = None):
    """延迟初始化 FlashHead；头像变化时重新 set_avatar（不重载权重）。"""
    global _flash_plugin, _current_avatar
    raw = avatar_path or DEFAULT_AVATAR
    target = _to_png_if_needed(raw)
    with _plugin_lock:
        if _flash_plugin is None:
            _run(_init_flash_head(target))
        elif target != _current_avatar:
            _run(_flash_plugin.set_avatar(target, use_face_crop=False))
            _current_avatar = target


# ─── URL 文章抓取 & 摘要 ──────────────────────────────────────────────────────
_MPT_PATH = "/home/test/money_printer_turbo/MoneyPrinterTurbo"

_SUMMARIZE_PROMPT = """\
你是短视频文案创作者。根据下面的文章，写一段适合数字人播报的口播文案。

要求：
- 200~300字，口语化，有观点，有共鸣感
- 不要说"本文"、"文章"、"作者"等字眼
- 结尾可加一句互动引导
- 只输出文案本身，不要任何解释

文章标题：{title}

文章内容：
{content}"""


def fetch_and_summarize(url: str) -> tuple[str, str]:
    """抓取公众号/普通文章 URL，LLM 生成摘要，返回 (摘要, 状态)。"""
    if _MPT_PATH not in sys.path:
        sys.path.insert(0, _MPT_PATH)
    from app.utils.wechat_article import fetch_article

    article = fetch_article(url)
    title   = article.get("title", "")
    content = article.get("content", "")
    logger.info("抓取文章: %s (%d 字)", title, len(content))

    prompt = _SUMMARIZE_PROMPT.format(title=title, content=content[:3000])
    summary = call_llm([{"role": "user", "content": prompt}])
    status  = f"✅ {title}（原文 {len(content)} 字 → 摘要 {len(summary)} 字）"
    return summary, status


# ─── ASR ─────────────────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str) -> str:
    """将音频文件转成文字（faster-whisper large-v3）。"""
    model = _load_whisper()
    segments, _ = model.transcribe(
        audio_path,
        language="zh",
        beam_size=5,
        vad_filter=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    logger.info("ASR: %s", text)
    return text


# ─── LLM ─────────────────────────────────────────────────────────────────────
_llm_client = None
_llm_model: str = LM_STUDIO_MODEL


def _get_llm_client():
    """懒创建 OpenAI client，并自动检测 LM Studio 当前加载的模型。"""
    global _llm_client, _llm_model
    if _llm_client is not None:
        return _llm_client, _llm_model
    from openai import OpenAI
    _llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key="lmstudio")
    try:
        models = _llm_client.models.list()
        if models.data:
            _llm_model = models.data[0].id
            logger.info("LM Studio 当前模型: %s", _llm_model)
    except Exception:
        logger.warning("无法检测 LM Studio 模型，使用默认: %s", _llm_model)
    return _llm_client, _llm_model


def _is_qwen(model_id: str) -> bool:
    return "qwen" in model_id.lower()


def call_llm(messages: list[dict]) -> str:
    """调用 LM Studio OpenAI 接口。

    Qwen3 MoE 的思考模式需要特殊处理：
    - system 只能是 /no_think（否则模型忽略该标记继续思考）
    - 原 system prompt 移入第一条 user 消息前缀
    - max_tokens 需足够大（thinking 会先消耗大量 tokens）
    """
    client, model = _get_llm_client()
    msgs = list(messages)

    if _is_qwen(model):
        # 提取原 system 内容，整合到 user 消息里
        sys_content = ""
        if msgs and msgs[0]["role"] == "system":
            sys_content = msgs[0]["content"]
            msgs = msgs[1:]  # 去掉原 system
        # 插入 /no_think system
        msgs = [{"role": "system", "content": "/no_think"}] + msgs
        # 将 system 指令前缀注入第一条 user 消息
        for i, m in enumerate(msgs):
            if m["role"] == "user":
                if sys_content:
                    msgs[i] = {**m, "content": f"[指令] {sys_content}\n\n{m['content']}"}
                break
        max_tokens = 3000  # Qwen3 thinking 消耗大，需要更多空间
    else:
        max_tokens = 500

    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    logger.info("LLM [%s]: %s", model, text[:80])
    return text


# ─── 声音克隆工具 (VoxCPM) ───────────────────────────────────────────────────
CLONED_VOICE_PREFIX = "CLONED:"


def extract_reference_audio(media_path: str, out_dir: Path = CLONED_VOICES_DIR) -> str:
    """从视频或音频文件中提取 16kHz 单声道 WAV，用作克隆参考音频。"""
    stem = Path(media_path).stem
    out_wav = str(out_dir / f"{stem}_ref.wav")
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", media_path,
         "-vn",                          # 去掉视频轨
         "-ar", "16000", "-ac", "1",     # 16kHz mono
         "-t", "30",                     # 最多取前 30 秒（克隆不需要太长）
         out_wav],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            "提取音频失败: " + r.stderr.decode("utf-8", errors="replace").strip()
        )
    size = os.path.getsize(out_wav)
    if size < 8000:
        raise RuntimeError(f"提取的音频过短或为空 ({size} bytes)")
    logger.info("参考音频已保存: %s (%.1f KB)", out_wav, size / 1024)
    return out_wav


def voxcpm_synthesize(text: str, reference_wav: str) -> str:
    """VoxCPM 声音克隆 TTS：参考音频 → 克隆音色合成，返回 16kHz WAV 路径。

    注意：VoxCPM2 需要约 8GB GPU VRAM。
    输出 48kHz → ffmpeg 重采样到 16kHz 供 FlashHead 使用。
    """
    import soundfile as sf
    model = _load_voxcpm()
    logger.info("VoxCPM 合成: %d 字，参考音频: %s", len(text), reference_wav)
    wav_48k = model.generate(
        text=text,
        reference_wav_path=reference_wav,
        cfg_value=2.0,
        inference_timesteps=10,
    )
    # 先存 48kHz WAV
    fd48, path_48k = tempfile.mkstemp(suffix=".wav", prefix="cv_vox48_")
    os.close(fd48)
    sf.write(path_48k, wav_48k, 48000)
    # 重采样到 16kHz
    wav_path = tempfile.mktemp(suffix=".wav", prefix="cv_vox16_")
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", path_48k, "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True,
    )
    try:
        os.unlink(path_48k)
    except OSError:
        pass
    if r.returncode != 0:
        raise RuntimeError(
            "VoxCPM 重采样失败: " + r.stderr.decode("utf-8", errors="replace").strip()
        )
    return wav_path


def list_cloned_voices() -> list[tuple[str, str]]:
    """返回已保存的克隆声音列表，格式为 [(显示名, CLONED:path), ...]。"""
    choices = []
    for wav in sorted(CLONED_VOICES_DIR.glob("*.wav")):
        label = wav.stem.replace("_ref", "").replace("_", " ")
        choices.append((f"克隆: {label}", f"{CLONED_VOICE_PREFIX}{wav}"))
    return choices


# ─── TTS → 16kHz WAV ─────────────────────────────────────────────────────────
_SENTENCE_SPLIT_RE = None


def _split_sentences(text: str, max_chars: int = 150) -> list[str]:
    """按标点分句，每句不超过 max_chars 字符。"""
    import re
    global _SENTENCE_SPLIT_RE
    if _SENTENCE_SPLIT_RE is None:
        _SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？；!?;])\s*')
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) <= max_chars:
            buf += p
        else:
            if buf:
                sentences.append(buf)
            buf = p
    if buf:
        sentences.append(buf)
    return sentences or [text]


async def _tts_chunk_to_mp3(text: str, voice: str) -> bytes:
    """单句 TTS，返回 MP3 bytes。"""
    import edge_tts
    buf = b""
    comm = edge_tts.Communicate(text, voice=voice)
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    if not buf:
        raise RuntimeError(f"edge_tts 返回空音频（可能代理/网络问题），文本: {text[:30]!r}")
    return buf


async def _tts_to_mp3(text: str, mp3_path: str, voice: str):
    """长文本分句处理，拼接成完整 MP3。"""
    sentences = _split_sentences(text)
    logger.info("TTS: %d 句，共 %d 字", len(sentences), len(text))
    all_audio = b""
    for s in sentences:
        all_audio += await _tts_chunk_to_mp3(s, voice)
    Path(mp3_path).write_bytes(all_audio)


def text_to_wav16k(text: str, voice: str = TTS_VOICE) -> str:
    """TTS → 16kHz 单声道 WAV。

    voice 以 CLONED: 开头时走 VoxCPM 声音克隆，否则走 edge-tts。
    """
    if voice.startswith(CLONED_VOICE_PREFIX):
        ref_wav = voice[len(CLONED_VOICE_PREFIX):]
        return voxcpm_synthesize(text, ref_wav)

    fd_mp3, mp3_path = tempfile.mkstemp(suffix=".mp3", prefix="cv_tts_")
    os.close(fd_mp3)
    wav_path = tempfile.mktemp(suffix=".wav", prefix="cv_tts_")
    try:
        _run(_tts_to_mp3(text, mp3_path, voice))
        mp3_size = os.path.getsize(mp3_path)
        if mp3_size < 1000:
            raise RuntimeError(f"TTS 生成的 MP3 文件过小 ({mp3_size} bytes)，疑似网络/代理问题")
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", mp3_path,
             "-ar", str(SR_16K), "-ac", "1",
             wav_path],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg MP3→WAV 失败 (exit {r.returncode}): "
                + r.stderr.decode("utf-8", errors="replace").strip()
            )
    finally:
        try:
            os.unlink(mp3_path)
        except OSError:
            pass
    return wav_path


# ─── FlashHead 推理 ───────────────────────────────────────────────────────────
async def _run_flash_head(wav_path: str) -> tuple[np.ndarray, int]:
    """输入 16kHz WAV 路径，输出 (frames, fps)。"""
    from inference.core.types import AudioChunk

    with wave.open(wav_path, "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    async def audio_stream():
        yield AudioChunk(
            data=pcm,
            sample_rate=SR_16K,
            channels=1,
            format="pcm_s16le",
            is_final=True,
        )

    chunks = []
    async for vc in _flash_plugin.generate_stream(audio_stream()):
        if vc is not None:
            chunks.append(vc)

    if not chunks:
        raise RuntimeError("FlashHead 未产出任何帧")

    frames = np.concatenate([c.frames for c in chunks], axis=0)
    fps = int(chunks[0].fps) or 20
    return frames, fps


def synthesize_video(wav_path: str) -> str:
    """FlashHead 推理 + ffmpeg 合成 MP4，返回 MP4 路径（/tmp，Gradio 默认允许）。"""
    fd, mp4_path = tempfile.mkstemp(suffix=".mp4", prefix="cv_out_", dir="/tmp")
    os.close(fd)

    frames, fps = _run(_run_flash_head(wav_path))
    t, h, w, c = frames.shape
    assert c == 3

    with wave.open(wav_path, "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    fd_pcm, pcm_path = tempfile.mkstemp(suffix=".pcm", prefix="cv_")
    os.close(fd_pcm)
    try:
        Path(pcm_path).write_bytes(pcm)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-f", "s16le", "-ac", "1", "-ar", str(SR_16K), "-i", pcm_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            mp4_path,
        ]
        subprocess.run(cmd, input=frames.tobytes(), check=True, capture_output=True)
    finally:
        try:
            os.unlink(pcm_path)
        except OSError:
            pass

    logger.info("视频生成: %s (%d 帧, %dfps)", mp4_path, t, fps)
    return mp4_path


# ─── 字幕：WAV → SRT → 烧录进 MP4 ──────────────────────────────────────────
def _fmt_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wav_to_srt(wav_path: str, srt_path: str, offset: float = 0.0):
    """用 faster-whisper 对 WAV 识别，生成 SRT 文件。offset 用于拼接多段时间偏移。"""
    model = _load_whisper()
    segments, _ = model.transcribe(
        wav_path, language="zh", beam_size=5, word_timestamps=False, vad_filter=True,
    )
    lines = []
    idx = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(
            f"{_fmt_srt_time(seg.start + offset)} --> {_fmt_srt_time(seg.end + offset)}"
        )
        lines.append(text)
        lines.append("")
        idx += 1
    Path(srt_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("SRT 生成: %s (%d 条字幕)", srt_path, idx - 1)
    return idx - 1  # 返回字幕条数


def _burn_subtitles(mp4_path: str, srt_path: str) -> str:
    """ffmpeg subtitles filter 将 SRT 字幕烧录进视频，返回新 MP4 路径。"""
    fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="cv_sub_", dir="/tmp")
    os.close(fd)
    # ffmpeg subtitles filter 需要转义冒号
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    style = (
        "FontName=SimHei,FontSize=22,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,Alignment=2"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", mp4_path,
        "-vf", f"subtitles={safe_srt}:force_style='{style}'",
        "-c:a", "copy",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            "字幕烧录失败: " + r.stderr.decode("utf-8", errors="replace").strip()
        )
    try:
        os.unlink(mp4_path)
    except OSError:
        pass
    return out_path


# ─── 直接播报：分批处理长文本 ────────────────────────────────────────────────
def synthesize_video_direct(text: str, voice: str = TTS_VOICE, subtitle: bool = False) -> str:
    """长文本直接播报：分句分批 TTS+FlashHead，合并成一个 MP4。

    每批 ~3 句（≤450 字），避免单次 FlashHead 推理音频过长导致 OOM 或超时。
    """
    sentences = _split_sentences(text, max_chars=150)
    BATCH_SIZE = 3
    batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
    logger.info("直接播报: %d 句 → %d 批", len(sentences), len(batches))

    all_frames: list[np.ndarray] = []
    all_pcm = b""
    fps_final = 20
    batch_wav_paths: list[str] = []   # 保留用于字幕生成

    for idx, batch in enumerate(batches):
        chunk_text = "".join(batch)
        logger.info("批次 %d/%d: %s…", idx + 1, len(batches), chunk_text[:30])
        wav_path = text_to_wav16k(chunk_text, voice=voice)
        try:
            frames, fps = _run(_run_flash_head(wav_path))
            all_frames.append(frames)
            fps_final = fps
            with wave.open(wav_path, "rb") as wf:
                all_pcm += wf.readframes(wf.getnframes())
            if subtitle:
                batch_wav_paths.append(wav_path)
                wav_path = None  # 不在 finally 里删
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    if not all_frames:
        raise RuntimeError("直接播报：未产生任何视频帧")

    combined = np.concatenate(all_frames, axis=0)
    t, h, w, _ = combined.shape

    fd, mp4_path = tempfile.mkstemp(suffix=".mp4", prefix="cv_out_", dir="/tmp")
    os.close(fd)
    fd_pcm, pcm_path = tempfile.mkstemp(suffix=".pcm", prefix="cv_")
    os.close(fd_pcm)
    try:
        Path(pcm_path).write_bytes(all_pcm)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps_final), "-i", "-",
            "-f", "s16le", "-ac", "1", "-ar", str(SR_16K), "-i", pcm_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            mp4_path,
        ]
        subprocess.run(cmd, input=combined.tobytes(), check=True, capture_output=True)
    finally:
        try:
            os.unlink(pcm_path)
        except OSError:
            pass

    logger.info("直播视频生成: %s (%d 帧, %dfps)", mp4_path, t, fps_final)

    # 字幕：把每批 WAV 逐段识别，拼成带偏移的完整 SRT
    if subtitle and batch_wav_paths:
        fd_srt, srt_path = tempfile.mkstemp(suffix=".srt", prefix="cv_sub_")
        os.close(fd_srt)
        try:
            offset = 0.0
            all_lines: list[str] = []
            sub_idx = 1
            for bwav in batch_wav_paths:
                try:
                    model = _load_whisper()
                    segs, _ = model.transcribe(
                        bwav, language="zh", beam_size=5, word_timestamps=False, vad_filter=True,
                    )
                    with wave.open(bwav, "rb") as wf:
                        duration = wf.getnframes() / wf.getframerate()
                    for seg in segs:
                        txt = seg.text.strip()
                        if txt:
                            all_lines.append(str(sub_idx))
                            all_lines.append(
                                f"{_fmt_srt_time(seg.start + offset)} --> "
                                f"{_fmt_srt_time(seg.end + offset)}"
                            )
                            all_lines.append(txt)
                            all_lines.append("")
                            sub_idx += 1
                    offset += duration
                finally:
                    try:
                        os.unlink(bwav)
                    except OSError:
                        pass
            Path(srt_path).write_text("\n".join(all_lines), encoding="utf-8")
            logger.info("直播字幕: %d 条", sub_idx - 1)
            mp4_path = _burn_subtitles(mp4_path, srt_path)
        finally:
            try:
                os.unlink(srt_path)
            except OSError:
                pass

    return mp4_path


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def process(
    audio_input,       # Gradio audio: (sample_rate, np.ndarray) or None
    text_input: str,   # 直接文字输入
    history: list,     # [[user, assistant], ...]
    avatar_path: str,
    system_prompt: str,
    tts_voice: str,
    direct_mode: bool = False,  # True → 跳过 LLM，直接朗读输入文字
    subtitle: bool = False,     # True → 字幕烧录进视频
) -> tuple[list, str | None, str]:
    """完整管道：(audio|text) → [LLM] → TTS → FlashHead → MP4"""
    status = ""
    try:
        # 1. 确保 FlashHead 已初始化
        status = "⚙️ 初始化数字人模型…"
        _ensure_flash_head(avatar_path or None)

        # 2. ASR / 文字输入
        user_text = (text_input or "").strip()
        if not user_text and audio_input is not None:
            status = "🎤 语音识别…"
            sr, audio_arr = audio_input
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="cv_asr_")
            os.close(fd)
            try:
                # 转成 16-bit mono WAV（兼容 int16 / float32 输入）
                if audio_arr.ndim > 1:
                    audio_arr = audio_arr.mean(axis=1)
                if audio_arr.dtype.kind in ("i", "u"):
                    audio_arr = audio_arr.astype(np.float32) / np.iinfo(audio_arr.dtype).max
                audio_int16 = (audio_arr * 32767).clip(-32768, 32767).astype(np.int16)
                with wave.open(tmp_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(audio_int16.tobytes())
                user_text = transcribe_audio(tmp_wav)
            finally:
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass

        if not user_text:
            return history, None, "❌ 没有输入内容"

        # 3. LLM（直接播报模式下跳过）
        if direct_mode:
            ai_text = user_text
            logger.info("直接播报模式，跳过 LLM，共 %d 字", len(ai_text))
        else:
            status = "🤔 LLM 思考中…"
            messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
            for u, a in history:
                messages.append({"role": "user", "content": u})
                if a:
                    messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": user_text})
            ai_text = call_llm(messages)

        # 4+5. TTS → FlashHead
        status = "🎬 生成数字人视频…"
        if direct_mode:
            # 长文本分批处理，每批 ~3 句，避免单次推理超时
            mp4_path = synthesize_video_direct(ai_text, voice=tts_voice, subtitle=subtitle)
        else:
            status = "🔊 合成语音…"
            wav_path = text_to_wav16k(ai_text, voice=tts_voice)
            try:
                status = "🎬 生成数字人视频…"
                mp4_path = synthesize_video(wav_path)
                if subtitle:
                    status = "📝 生成字幕…"
                    fd_srt, srt_path = tempfile.mkstemp(suffix=".srt", prefix="cv_sub_")
                    os.close(fd_srt)
                    try:
                        _wav_to_srt(wav_path, srt_path)
                        mp4_path = _burn_subtitles(mp4_path, srt_path)
                    finally:
                        try:
                            os.unlink(srt_path)
                        except OSError:
                            pass
            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        history = list(history) + [[user_text, ai_text]]
        return history, mp4_path, "✅ 完成"

    except Exception as exc:
        logger.exception("pipeline error")
        return history, None, f"❌ 错误: {exc}"


# ─── Gradio 界面 ──────────────────────────────────────────────────────────────
def build_ui():
    import gradio as gr

    with gr.Blocks(title="数字人演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 数字人演示\n语音或文字输入 → AI 回复 → 实时数字人视频")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="对话记录", height=400)
                with gr.Row():
                    audio_in = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 语音输入（录音后点发送）",
                    )
                with gr.Row():
                    text_in = gr.Textbox(
                        placeholder="直接输入文字，或从下方 URL 提取摘要…",
                        label="文字输入",
                        lines=3,
                    )
                with gr.Row():
                    url_in = gr.Textbox(
                        placeholder="粘贴公众号 / 网页 URL，点「提取摘要」自动填入上方",
                        label="文章 URL",
                        lines=1,
                        scale=4,
                    )
                    url_btn = gr.Button("提取摘要", variant="secondary", scale=1)
                with gr.Row():
                    direct_mode = gr.Checkbox(label="直接播报（跳过AI，逐字朗读）", value=False)
                    subtitle_chk = gr.Checkbox(label="显示字幕", value=False)
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary", scale=2)
                    clear_btn = gr.Button("清除对话", scale=1)

            with gr.Column(scale=1):
                video_out = gr.Video(label="数字人视频", height=400, sources=[])
                status_out = gr.Textbox(label="状态", interactive=False)

        with gr.Accordion("🎙️ 声音克隆", open=False):
            gr.Markdown(
                "上传包含目标声音的视频或音频（支持 mp4/mov/mp3/wav 等），"
                "提取后作为克隆参考，可在 TTS 声音下拉框中选用。\n\n"
                "> 需要 GPU（VoxCPM2 约 8GB VRAM）。"
            )
            with gr.Row():
                clone_media_in = gr.File(
                    label="参考视频 / 音频",
                    file_types=["video", "audio"],
                )
                with gr.Column():
                    clone_name_in = gr.Textbox(
                        label="声音名称",
                        placeholder="例：张总",
                        lines=1,
                    )
                    clone_btn = gr.Button("提取并保存参考音频", variant="secondary")
                    clone_status = gr.Textbox(label="", interactive=False, lines=2)

        with gr.Accordion("⚙️ 设置", open=False):
            with gr.Row():
                with gr.Column():
                    avatar_in = gr.Image(
                        label="头像图片（上传后点「更换头像」）",
                        type="filepath",
                        height=200,
                    )
                    avatar_btn = gr.Button("更换头像", variant="secondary")
                    avatar_status = gr.Textbox(label="", interactive=False, lines=1)
                with gr.Column():
                    lms_url_in = gr.Textbox(
                        value=LM_STUDIO_URL,
                        label="LM Studio 地址",
                        lines=1,
                        placeholder="http://192.168.1.101:1234/v1",
                    )
                    lms_btn = gr.Button("应用地址", variant="secondary")
                    lms_status = gr.Textbox(label="", interactive=False, lines=1)
                    system_prompt_in = gr.Textbox(
                        value=SYSTEM_PROMPT,
                        label="系统提示词",
                        lines=4,
                    )
                    _edge_tts_choices = [
                        ("晓晓 - 女声（zh-CN）", "zh-CN-XiaoxiaoNeural"),
                        ("云希 - 男声（zh-CN）", "zh-CN-YunxiNeural"),
                        ("晓伊 - 女声（zh-CN）", "zh-CN-XiaoyiNeural"),
                        ("云健 - 男声（zh-CN）", "zh-CN-YunjianNeural"),
                        ("晓辰 - 女声（zh-CN）", "zh-CN-XiaochenNeural"),
                        ("晓涵 - 女声（zh-CN）", "zh-CN-XiaohanNeural"),
                        ("曉臻 - 女声（zh-TW）", "zh-TW-HsiaoChenNeural"),
                        ("雲哲 - 男声（zh-TW）", "zh-TW-YunJheNeural"),
                    ]
                    voice_in = gr.Dropdown(
                        choices=_edge_tts_choices + list_cloned_voices(),
                        value=TTS_VOICE,
                        label="TTS 声音",
                    )

        history_state = gr.State([])

        def on_process(audio, text, history, avatar, sys_prompt, voice, direct, sub):
            new_history, video, status = process(audio, text, history, avatar, sys_prompt, voice, direct, sub)
            return new_history, new_history, video, status, None, ""

        send_btn.click(
            fn=on_process,
            inputs=[audio_in, text_in, history_state, avatar_in, system_prompt_in, voice_in, direct_mode, subtitle_chk],
            outputs=[history_state, chatbot, video_out, status_out, audio_in, text_in],
        )
        text_in.submit(
            fn=on_process,
            inputs=[audio_in, text_in, history_state, avatar_in, system_prompt_in, voice_in, direct_mode, subtitle_chk],
            outputs=[history_state, chatbot, video_out, status_out, audio_in, text_in],
        )
        clear_btn.click(lambda: ([], [], None, ""), outputs=[history_state, chatbot, video_out, status_out])

        def on_fetch_url(url: str):
            url = (url or "").strip()
            if not url:
                return "", "❌ 请先输入 URL"
            try:
                summary, status = fetch_and_summarize(url)
                return summary, status
            except Exception as e:
                return "", f"❌ {e}"

        url_btn.click(
            fn=on_fetch_url,
            inputs=[url_in],
            outputs=[text_in, status_out],
        )

        def change_avatar(avatar_path):
            if not avatar_path:
                return "未选择图片"
            try:
                _ensure_flash_head(avatar_path)
                return "✅ 头像已更换"
            except Exception as e:
                return f"❌ {e}"

        avatar_btn.click(fn=change_avatar, inputs=[avatar_in], outputs=[avatar_status])

        def do_clone_voice(media_file, name: str):
            if not media_file:
                return "❌ 请先上传视频或音频文件", gr.update()
            name = (name or "").strip()
            if not name:
                return "❌ 请填写声音名称", gr.update()
            try:
                media_path = media_file if isinstance(media_file, str) else media_file.name
                # 使用名称作为文件名
                safe_name = name.replace(" ", "_").replace("/", "_")
                out_wav = str(CLONED_VOICES_DIR / f"{safe_name}_ref.wav")
                r = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-i", media_path,
                     "-vn", "-ar", "16000", "-ac", "1", "-t", "30",
                     out_wav],
                    capture_output=True,
                )
                if r.returncode != 0:
                    return "❌ 提取失败: " + r.stderr.decode("utf-8", errors="replace").strip(), gr.update()
                size_kb = os.path.getsize(out_wav) / 1024
                if size_kb < 8:
                    return f"❌ 提取的音频过短或为空 ({size_kb:.1f} KB)", gr.update()
                # 刷新下拉框
                new_choices = _edge_tts_choices + list_cloned_voices()
                new_val = f"{CLONED_VOICE_PREFIX}{out_wav}"
                return (
                    f"✅ 参考音频已保存 ({size_kb:.0f} KB)\n路径: {out_wav}",
                    gr.update(choices=new_choices, value=new_val),
                )
            except Exception as e:
                return f"❌ {e}", gr.update()

        clone_btn.click(
            fn=do_clone_voice,
            inputs=[clone_media_in, clone_name_in],
            outputs=[clone_status, voice_in],
        )

        def apply_lms_url(url: str):
            url = (url or "").strip()
            if not url:
                return "❌ 地址不能为空"
            try:
                from openai import OpenAI
                c = OpenAI(base_url=url, api_key="lmstudio")
                models = c.models.list()
                # 写回全局，下次 call_llm 直接复用
                global _llm_client, _llm_model
                _llm_client = c
                if models.data:
                    _llm_model = models.data[0].id
                logger.info("LM Studio 地址更新为 %s，模型: %s", url, _llm_model)
                return f"✅ 已连接: {_llm_model}"
            except Exception as e:
                return f"❌ 连接失败: {e}"

        lms_btn.click(fn=apply_lms_url, inputs=[lms_url_in], outputs=[lms_status])

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true", help="启动时预加载所有模型")
    args = parser.parse_args()

    if args.preload:
        logger.info("预加载模型…")
        _load_whisper()
        _ensure_flash_head()
        logger.info("预加载完成，启动界面")

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(OUTPUT_DIR)],
    )
