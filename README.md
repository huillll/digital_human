# 用开源工具搭建本地数字人：语音对话 + AI 回复 + 实时说话视频

> 全程本地部署，无需云服务，一张照片变成会说话的数字人。

## 效果

输入一段话（语音或文字）→ 本地大模型思考回复 → 微软 TTS 合成语音 → FlashHead 驱动照片说话 → 浏览器实时播放视频

直接播报模式：粘贴 600 字政策文件 → 数字人逐字朗读，全程不经过大模型。

---

## 技术栈一览

| 组件 | 方案 | 说明 |
|------|------|------|
| 大语言模型 | LM Studio + Qwen3.5-35B-A3B | 本机局域网 API，OpenAI 兼容 |
| 语音识别 ASR | faster-whisper large-v3 | CPU int8 推理，~1s 识别 |
| 文字转语音 TTS | edge-tts（微软 Azure 神经网络声音） | 免费在线，无需 API key |
| 数字人驱动 | FlashHead（CyberVerse） | 滑动窗口流式推理，512×512 |
| Web 界面 | Gradio 5.x | 语音输入 + 视频输出 |
| 视频合成 | ffmpeg | 帧序列 + PCM → H.264 MP4 |

---

## 架构

```
麦克风/文字
    │
    ▼
[faster-whisper]  ← ASR（语音输入时）
    │
    ▼
[LM Studio / Qwen3]  ← 直接播报模式跳过此步
    │
    ▼
[edge-tts]  → MP3 → ffmpeg → 16kHz WAV
    │
    ▼
[FlashHead]  → RGB 帧序列
    │
    ▼
[ffmpeg]  → MP4
    │
    ▼
Gradio 浏览器播放
```

---

## 核心实现

### 1. 异步/同步边界

FlashHead 的 `generate_stream()` 是 `async` 接口，而 Gradio 的回调在普通线程里运行。直接 `asyncio.run()` 会报"已有事件循环"的错误。

解决方案：启动一个专用后台事件循环，用 `run_coroutine_threadsafe` 桥接：

```python
_bg_loop = asyncio.new_event_loop()
threading.Thread(target=_bg_loop.run_forever, daemon=True).start()

def _run(coro, timeout=600.0):
    fut = asyncio.run_coroutine_threadsafe(coro, _bg_loop)
    return fut.result(timeout=timeout)
```

### 2. Qwen3 思考模式踩坑

Qwen3 MoE 开启思考模式后，500 个 token 全被推理链消耗，`content` 返回空字符串。

解决方法：
- `system` 消息**只放** `/no_think`，不放任何其他内容
- 原来的系统提示词前缀到第一条 `user` 消息里
- `max_tokens` 设到 3000（thinking 本身就要吃掉大量 token）

```python
if _is_qwen(model):
    msgs = [{"role": "system", "content": "/no_think"}] + msgs
    msgs[first_user_idx]["content"] = f"[指令] {sys_content}\n\n{original_content}"
```

### 3. edge-tts 返回空 MP3

长文本（>150字）或偶发网络抖动时，edge-tts 会返回 0 字节音频，ffmpeg 报：

```
Failed to find two consecutive MPEG audio frames
```

两层防御：
1. **分句**：按 `。！？；` 分句，每句 ≤150 字单独请求 TTS
2. **校验**：拼接后检查 MP3 文件 >1000 字节，否则直接报错

```python
async def _tts_chunk_to_mp3(text, voice):
    buf = b""
    async for chunk in edge_tts.Communicate(text, voice=voice).stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    if not buf:
        raise RuntimeError(f"edge_tts 返回空音频: {text[:30]!r}")
    return buf
```

### 4. 直接播报长文本超时

600 字文章 TTS 后约 40 秒音频，FlashHead 单次推理触发超时。

解决：分批处理，每批 3 句独立走完 TTS→FlashHead，最后把所有帧 `np.concatenate` 拼成一个视频：

```python
def synthesize_video_direct(text, voice):
    sentences = _split_sentences(text, max_chars=150)
    batches = [sentences[i:i+3] for i in range(0, len(sentences), 3)]
    all_frames, all_pcm = [], b""
    for batch in batches:
        wav = text_to_wav16k("".join(batch), voice=voice)
        frames, fps = _run(_run_flash_head(wav))
        all_frames.append(frames)
        with wave.open(wav) as wf:
            all_pcm += wf.readframes(wf.getnframes())
    combined = np.concatenate(all_frames, axis=0)
    # ffmpeg 合成最终 MP4 ...
```

### 5. Gradio 视频组件

Gradio 5 的 `gr.Video()` 默认 `sources=["upload","webcam"]`，会显示上传框而不是播放器。输出视频必须设：

```python
video_out = gr.Video(label="数字人视频", height=400, sources=[])
```

---

## 环境配置要点

**conda 环境分离**：FlashHead 依赖的 PyTorch/MediaPipe 在 `LatentSync` 环境，Gradio 在 `cyberverse` 环境。通过手动 `sys.path.insert` 混用：

```python
sys.path.insert(1, "/path/to/LatentSync/lib/python3.10/site-packages")
```

**代理绕行**：LM Studio 和 edge-tts 都需要设 NO_PROXY，否则流量被代理拦截：

```bash
export NO_PROXY="192.168.1.101,localhost,127.0.0.1"
```

**ffmpeg 路径**：系统 PATH 里没有 ffmpeg，必须用 conda 环境里的：

```bash
export PATH="/path/to/conda/envs/cyberverse/bin:$PATH"
```

---

## 启动

```bash
bash run_demo.sh
# 浏览器访问 http://<局域网IP>:7860
```

---

## 待改进

- [ ] FlashHead 推理慢（ARM CPU，每帧 ~300ms），期待 GPU 加速
- [ ] 直接播报分批之间有轻微停顿，可预生成下一批做流水线
- [ ] ASR 用 CPU int8，长语音识别慢，可换端点检测 + 流式识别

---

## 参考项目

- [CyberVerse](https://github.com/CyberVerse-AI/CyberVerse) — 实时数字人框架，FlashHead 来自这里
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 加速的 Whisper
- [edge-tts](https://github.com/rany2/edge-tts) — 微软 Azure TTS 的免费逆向库
- [LM Studio](https://lmstudio.ai) — 本地大模型管理和推理
