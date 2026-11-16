"""外部服務：GX10-B 語音辨識（asr-gateway）、千鉑 AI Studio 語音合成（CosyVoice3）、千鉑 LiteLLM（gemma4-31b）。

金鑰一律從環境／密鑰檔讀，不寫進程式碼、不送到前端：
- ASR：~/.claude/skills/video-transcribe/.api_key（或環境變數 ASR_GATEWAY_API_KEY）
- TTS：~/.claude/skill-secrets.env 的 QIANPRO_STUDIO_TOKEN（Breeze 台灣腔為主、CosyVoice3 備援）
- LLM：~/.claude/skill-secrets.env 的 FLASHBOT_LLM_BASE／FLASHBOT_LLM_KEY（暫借魔力專案的 key；
       可用環境變數 ARENA_LLM_BASE／ARENA_LLM_KEY／ARENA_LLM_MODEL 換成本專案專用的）
"""
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid

HOME = os.path.expanduser("~")
UA = {"User-Agent": "Mozilla/5.0 (OOXX-Arena)"}     # Cloudflare 會擋 Python 預設 UA（實測 403）
ASR_URL = os.environ.get("ARENA_ASR_URL", "http://gx10-b.taile62660.ts.net:9000")
TTS_URL = "https://studio.qianpro.shop"


def _secrets():
    out = {}
    try:
        for line in open(os.path.join(HOME, ".claude", "skill-secrets.env"), encoding="utf-8"):
            m = re.match(r"\s*(?:export\s+)?([A-Z0-9_]+)=(.*)", line)
            if m:
                out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    except OSError:
        pass
    return out


SECRETS = _secrets()


def _asr_key():
    k = os.environ.get("ASR_GATEWAY_API_KEY")
    if k:
        return k.strip()
    return open(os.path.join(HOME, ".claude", "skills", "video-transcribe", ".api_key"), encoding="utf-8").read().strip()


def _multipart(fields, files):
    b = "----arena" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode("utf-8")
    for k, (fn, data, ctype) in files.items():
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\nContent-Type: %s\r\n\r\n"
                 % (b, k, fn, ctype)).encode("utf-8") + data + b"\r\n"
    body += ("--%s--\r\n" % b).encode()
    return body, "multipart/form-data; boundary=" + b


# ---------------- 語音辨識 ----------------
_s2t = None


def to_trad(text):
    global _s2t
    try:
        if _s2t is None:
            import opencc
            _s2t = opencc.OpenCC("s2twp")
        return _s2t.convert(text)
    except Exception:
        return text


def transcribe(audio_bytes, suffix=".webm"):
    """瀏覽器錄音（webm/mp4/ogg）→ ffmpeg 轉 16 kHz 單聲道 wav → asr-gateway → 繁體文字。"""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in" + suffix)
        dst = os.path.join(td, "in.wav")
        open(src, "wb").write(audio_bytes)
        r = subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-f", "wav", dst],
                           capture_output=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError("錄音轉檔失敗：%s" % r.stderr[-300:].decode("utf-8", "ignore"))
        wav = open(dst, "rb").read()
    body, ctype = _multipart({"language": "zh"}, {"file": ("voice.wav", wav, "audio/wav")})
    req = urllib.request.Request(ASR_URL + "/v1/transcribe", data=body,
                                 headers={"Content-Type": ctype, "X-API-Key": _asr_key(), **UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            text = d.get("text") or "".join(s.get("text", "") for s in d.get("segments", []))
            return to_trad(text.strip())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:                  # 每分鐘 10 次、同時 1 個
                time.sleep(float(e.headers.get("Retry-After") or 3))
                continue
            raise RuntimeError("語音辨識失敗 HTTP %d：%s" % (e.code, e.read()[:200].decode("utf-8", "ignore")))


# ---------------- 語音合成 ----------------
# 使用者要求：送進 TTS 前轉簡體（發音較準）、O 念「圈」X 念「叉」、要台灣腔。
# 台灣腔用 Breeze TTS 2（管理員 token、研究／非商業授權）：voice_ref/ref.wav 是用「聲音設計」做的
# 台灣年輕女聲參考音（voice_ref/ref.json 有描述與逐字稿），每句用 clone 模式念，音色才一致。
# Breeze 失敗（服務沒開等）時自動退回 CosyVoice3。ARENA_TTS_ENGINE=cosyvoice 可強制用 CosyVoice3。
VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_ref")
_t2s = None


def _num(n):
    d = "零一二三四五六七八九"
    if n < 10:
        return d[n]
    return ("" if n < 20 else d[n // 10]) + "十" + ("" if n % 10 == 0 else d[n % 10])


def speech_text(text):
    """給 TTS 念的文字：OOXX／O／X 換成圈叉，再轉簡體。"""
    global _t2s
    t = re.sub(r"(?i)ooxx", "圈圈叉叉", text)
    t = re.sub(r"(?<![A-Za-z])[Oo](?![A-Za-z])", "圈", t)
    t = re.sub(r"(?<![A-Za-z])[Xx](?![A-Za-z])", "叉", t)
    t = re.sub(r"(圈|叉) +", lambda m: m.group(1), t)
    t = re.sub(r" *(\d+) *", lambda m: m.group(1), t)
    t = re.sub(r"\d{1,2}", lambda m: _num(int(m.group(0))), t)
    try:
        if _t2s is None:
            import opencc
            _t2s = opencc.OpenCC("t2s")
        t = _t2s.convert(t).replace("搆", "够")
    except Exception:
        pass
    return t


def _token():
    token = os.environ.get("QIANPRO_STUDIO_TOKEN") or SECRETS.get("QIANPRO_STUDIO_TOKEN")
    if not token:
        raise RuntimeError("沒有 QIANPRO_STUDIO_TOKEN")
    return token


def _studio_post(path, fields, files, token):
    body, ctype = _multipart(fields, files)
    req = urllib.request.Request(TTS_URL + path, data=body,
                                 headers={"Content-Type": ctype, "Authorization": "Bearer " + token, **UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    if not d.get("ok"):
        raise RuntimeError("語音合成失敗：%s" % d.get("error"))
    req = urllib.request.Request(TTS_URL + "/asset/" + d["id"], headers={"Authorization": "Bearer " + token, **UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


_ref = None


def tts_breeze(text, token):
    global _ref
    if _ref is None:
        meta = json.load(open(os.path.join(VOICE_DIR, "ref.json"), encoding="utf-8"))
        _ref = (open(os.path.join(VOICE_DIR, "ref.wav"), "rb").read(), meta["text"])
    wav, ref_text = _ref
    return _studio_post("/api/breeze/tts", {"mode": "clone", "seed": "1", "ref_text": ref_text, "text": text[:600]},
                        {"ref_audio": ("ref.wav", wav, "audio/wav")}, token)


def tts_cosy(text, token, voice="female", style=""):
    return _studio_post("/api/voice/tts", {"text": text[:600], "cv3_voice": voice, "style": style, "dialect": ""}, {}, token)


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_cache")


def _cache_path(text):
    import hashlib
    eng = os.environ.get("ARENA_TTS_ENGINE", "breeze")
    return os.path.join(CACHE_DIR, hashlib.sha1((eng + "|" + speech_text(text)).encode("utf-8")).hexdigest()[:16] + ".wav")


def tts_cached(text):
    """同一句話合成過就直接讀存檔（固定的提示句幾乎零延遲）；沒有回 None。"""
    try:
        with open(_cache_path(text), "rb") as f:
            return f.read()
    except OSError:
        return None


def tts(text, voice="female", style=""):
    """回 wav bytes。預設 Breeze 台灣腔，失敗退回 CosyVoice3。合成結果存到 voice_cache/，同一句下次直接用。"""
    wav = tts_cached(text)
    if wav:
        return wav
    token = _token()
    t = speech_text(text)
    wav = None
    if os.environ.get("ARENA_TTS_ENGINE", "breeze") == "breeze":
        try:
            wav = tts_breeze(t, token)
        except Exception as e:
            print("[TTS] Breeze 失敗，改用 CosyVoice3：%s" % str(e)[:120], flush=True)
    if wav is None:
        wav = tts_cosy(t, token, voice, style)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(text), "wb") as f:
            f.write(wav)
    except OSError:
        pass
    return wav


# ---------------- 語言模型 ----------------
def llm(messages, max_tokens=300, temperature=0.4):
    base = os.environ.get("ARENA_LLM_BASE") or SECRETS.get("FLASHBOT_LLM_BASE", "https://ai.qianpro.shop")
    key = os.environ.get("ARENA_LLM_KEY") or SECRETS.get("FLASHBOT_LLM_KEY")
    model = os.environ.get("ARENA_LLM_MODEL", "gemma4-31b")
    if not key:
        raise RuntimeError("沒有 LLM 金鑰")
    base = base.rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer " + key, **UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"]


if __name__ == "__main__":
    import sys
    t = time.time()
    print("LLM：", llm([{"role": "user", "content": "用一句繁體中文跟我打招呼"}], max_tokens=60), "%.1fs" % (time.time() - t))
    t = time.time()
    wav = tts("你好，我是井字棋機器人，準備好了嗎？")
    print("TTS：%d bytes %.1fs" % (len(wav), time.time() - t))
    open(os.path.join(os.path.dirname(__file__), "tts_test.wav"), "wb").write(wav)
    t = time.time()
    print("ASR（用剛合成的語音測）：", transcribe(wav, ".wav"), "%.1fs" % (time.time() - t))
