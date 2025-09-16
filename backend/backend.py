"""
Nexus Core — Colab Relay Backend (Unbounded, cloudflared)
This script installs required dependencies, launches a Flask app
providing multiple endpoints, and exposes it via Cloudflare tunnel.

Set environment variable HF_TOKEN with your Hugging Face token before running.
"""

import os, subprocess, sys, json, time, base64, threading, asyncio, re, signal

# ------------------- INSTALLS -----------------------
print("Installing dependencies…")

def _install_deps():
    pkgs = [
        "flask",
        "flask-cors",
        "gevent",
        "sentence-transformers",
        "py-mini-racer",
        "duckduckgo-search",
        "huggingface_hub>=0.22",
        "playwright",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
    subprocess.run(["playwright", "install", "--with-deps"], check=True)

try:
    _install_deps()
    print("✅ Python deps installed")
except Exception as e:
    print("⚠️ Dependency installation failed:", e)

# After installs, import heavy packages
from flask import Flask, request, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from py_mini_racer import py_mini_racer
from sentence_transformers import SentenceTransformer, CrossEncoder
from duckduckgo_search import DDGS
from huggingface_hub import HfApi, ModelFilter
from playwright.async_api import async_playwright
import requests

# ---------------------- CONFIG ----------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
PORT = int(os.environ.get("PORT", "8081"))
ENABLE_SAFETY_HOOKS = os.environ.get("ENABLE_SAFETY_HOOKS", "False").lower() == "true"

# Ensure cloudflared binary is available

def ensure_cloudflared():
    try:
        out = subprocess.check_output(["cloudflared", "--version"], text=True)
        print("cloudflared present:", out.splitlines()[0])
        return
    except Exception:
        pass
    print("Installing cloudflared binary…")
    subprocess.run([
        "wget",
        "-q",
        "-O",
        "/tmp/cloudflared.deb",
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
    ], check=True)
    subprocess.run(["dpkg", "-i", "/tmp/cloudflared.deb"], check=False)
    try:
        out = subprocess.check_output(["cloudflared", "--version"], text=True)
        print("✅ cloudflared installed:", out.splitlines()[0])
    except Exception as e:
        print("❌ cloudflared install failed:", e)

ensure_cloudflared()

# ------------------- GLOBALS ------------------------
os.environ["HF_TOKEN"] = HF_TOKEN if HF_TOKEN else ""

def _apply_safety(text: str) -> str:
    if not ENABLE_SAFETY_HOOKS:
        return text
    try:
        import safety_hooks
        return safety_hooks.apply(text)
    except Exception:
        return text

embed_model = SentenceTransformer("WhereIsAI/UAE-Large-V1")
rank_model  = CrossEncoder("BAAI/bge-reranker-base")
js_ctx      = py_mini_racer.MiniRacer()
api         = HfApi()
PUBLIC_URL  = None

REG_PATH = "/content/nexus_models.json"
DEFAULT_REGISTRY = {
    "reason_pool": [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    "code_pool": [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "bigcode/starcoder2-15b",
    ],
    "math_pool": [
        "deepseek-ai/deepseek-math-7b",
        "Qwen/Qwen2.5-Math-7B-Instruct",
    ],
}

def _hf_headers():
    tok = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {tok}"} if tok else {}

def _load_registry():
    if os.path.exists(REG_PATH):
        try:
            with open(REG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    with open(REG_PATH, "w") as f:
        json.dump(DEFAULT_REGISTRY, f, indent=2)
    return json.loads(json.dumps(DEFAULT_REGISTRY))

def _save_registry(reg):
    with open(REG_PATH, "w") as f:
        json.dump(reg, f, indent=2)

MODEL_REGISTRY = _load_registry()

def _probe_chat_model(model_id: str, timeout=40):
    t0 = time.time()
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers={**_hf_headers(), "content-type":"application/json"},
            json={"inputs":"Say OK.","parameters":{"max_new_tokens":12,"temperature":0.1,"return_full_text":False}},
            timeout=timeout
        )
        if r.status_code != 200:
            return False, None, f"HTTP {r.status_code}: {r.text[:200]}"
        j = r.json()
        text = (j[0].get("generated_text") if isinstance(j, list) else j.get("generated_text")) or ""
        lat = int((time.time() - t0)*1000)
        return (("OK" in text or len(text.strip())>0), lat, None)
    except Exception as e:
        return False, None, str(e)

# ------------------ APP -----------------------------
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "url": PUBLIC_URL})

@app.route("/relay-url", methods=["GET"])
def relay_url():
    return jsonify({"url": PUBLIC_URL})

@app.route("/capabilities", methods=["GET"])
def capabilities():
    return jsonify({"chat": bool(os.environ.get("HF_TOKEN")),
                    "embed": True, "rank": True, "exec": True, "browser": True, "search": True})

@app.route("/chat", methods=["POST"])
def chat():
    if not os.environ.get("HF_TOKEN"):
        return jsonify({"error":"Hugging Face token not set"}), 500
    p = request.json or {}
    model_id = p.get("model") or "mistralai/Mixtral-8x7B-Instruct-v0.1"
    prompt   = p.get("prompt", "")
    params   = p.get("params") or {}
    t0 = time.time()
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers={**_hf_headers(), "content-type":"application/json"},
            json={"inputs": prompt, "parameters": {
                "max_new_tokens": int(params.get("max_new_tokens", 2048)),
                "temperature": float(params.get("temperature", 0.2)),
                "return_full_text": False
            }},
            timeout=120
        )
        r.raise_for_status()
        j = r.json()
        text = (j[0].get("generated_text") if isinstance(j,list) else j.get("generated_text")) or ""
        return jsonify({"text": _apply_safety(text), "latency_ms": int((time.time()-t0)*1000), "model": model_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/embed", methods=["POST"])
def embed():
    t0 = time.time()
    p = request.json or {}
    try:
        vecs = embed_model.encode(p.get("texts") or [], normalize_embeddings=True).tolist()
        return jsonify({"vectors": vecs, "latency_ms": int((time.time()-t0)*1000)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rank", methods=["POST"])
def rank():
    t0 = time.time()
    p = request.json or {}
    q = p.get("query","" ); c = p.get("candidates") or []
    try:
        pairs = [[q, s] for s in c]
        scores = rank_model.predict(pairs, convert_to_numpy=True).tolist()
        return jsonify({"scores": scores, "latency_ms": int((time.time()-t0)*1000)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/exec", methods=["POST"])
def exec_js():
    t0 = time.time()
    p = request.json or {}
    code = p.get("code","" )
    try:
        res = js_ctx.eval(code)
        return jsonify({"success": True, "result": res, "latency_ms": int((time.time()-t0)*1000)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/search", methods=["POST"])
def search():
    t0 = time.time()
    p = request.json or {}
    query = p.get("query") or ""
    max_results = int(p.get("max_results", 10))
    region = p.get("region","wt-wt"); safesearch = p.get("safesearch","Off")
    results = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(query, region=region, safesearch=safesearch, max_results=max_results):
                results.append({"title": r.get("title",""),
                                "url": r.get("href") or r.get("url") or "",
                                "snippet": r.get("body") or r.get("snippet") or ""})
        return jsonify({"results": results, "latency_ms": int((time.time()-t0)*1000)})
    except Exception as e:
        return jsonify({"error": str(e), "results": []}), 500

@app.route("/browser/run-adv", methods=["POST"])
def browser_run_adv():
    p = request.json or {}
    async def run():
        t0 = time.time()
        out = {"ok": True, "html": "", "screenshots": [], "log": []}
        url = p.get("url") or "https://example.com"
        actions = p.get("actions") or []
        timeout_ms = int(p.get("timeout_ms", 0))  # 0 = unlimited
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=(0 if timeout_ms==0 else timeout_ms))
                out["log"].append({"event":"goto","url":url})
                for step in actions:
                    typ = step.get("type")
                    if typ == "goto":
                        u = step.get("url") or url
                        await page.goto(u, timeout=(0 if timeout_ms==0 else timeout_ms)); out["log"].append({"event":"goto","url":u})
                    elif typ == "click":
                        sel = step.get("selector"); await page.click(sel, timeout=(0 if timeout_ms==0 else timeout_ms)); out["log"].append({"event":"click","selector":sel})
                    elif typ == "fill":
                        sel = step.get("selector"); val = step.get("value","" ); await page.fill(sel, val, timeout=(0 if timeout_ms==0 else timeout_ms)); out["log"].append({"event":"fill","selector":sel})
                    elif typ == "wait_for":
                        sel = step.get("selector"); await page.wait_for_selector(sel, timeout=(0 if timeout_ms==0 else timeout_ms)); out["log"].append({"event":"wait_for","selector":sel})
                    elif typ == "extract":
                        sel = step.get("selector"); attr = step.get("attr","text"); el = await page.query_selector(sel)
                        val = ""
                        if el:
                            if attr=="text": val = await el.inner_text()
                            elif attr=="html": val = await el.inner_html()
                            else: val = await el.get_attribute(attr) or ""
                        out["log"].append({"event":"extract","selector":sel,"attr":attr,"value":(val or "")[:2000]})
                out["html"] = await page.content()
                if p.get("screenshots", True):
                    shot = await page.screenshot(full_page=True)
                    out["screenshots"].append(base64.b64encode(shot).decode())
                await browser.close()
        except Exception as e:
            out["ok"] = False; out["error"] = str(e)
        out["latency_ms"] = int((time.time()-t0)*1000)
        return jsonify(out)
    return asyncio.run(run())

# ---------- Model registry ----------
@app.route("/models/list", methods=["GET"])
def models_list():
    return jsonify(MODEL_REGISTRY)

@app.route("/models/add", methods=["POST"])
def models_add():
    p = request.json or {}
    role = p.get("role"); model = p.get("model"); skip_probe = bool(p.get("skip_probe", False))
    if role not in ("reason","code","math") or not model:
        return jsonify({"error":"role must be reason|code|math and model required"}), 400
    key = {"reason":"reason_pool","code":"code_pool","math":"math_pool"}[role]
    if model not in MODEL_REGISTRY[key]:
        if not skip_probe:
            ok, lat, err = _probe_chat_model(model)
            if not ok: return jsonify({"error": f"model probe failed: {err or 'no text'}"}), 400
        MODEL_REGISTRY[key].append(model); _save_registry(MODEL_REGISTRY)
    return jsonify({"ok": True, "registry": MODEL_REGISTRY})

@app.route("/models/remove", methods=["POST"])
def models_remove():
    p = request.json or {}
    role = p.get("role"); model = p.get("model")
    if role not in ("reason","code","math") or not model:
        return jsonify({"error":"role must be reason|code|math and model required"}), 400
    key = {"reason":"reason_pool","code":"code_pool","math":"math_pool"}[role]
    if model in MODEL_REGISTRY[key]:
        MODEL_REGISTRY[key].remove(model); _save_registry(MODEL_REGISTRY)
    return jsonify({"ok": True, "registry": MODEL_REGISTRY})

@app.route("/models/test", methods=["POST"])
def models_test():
    p = request.json or {}
    model = p.get("model")
    if not model: return jsonify({"error":"model required"}), 400
    ok, lat, err = _probe_chat_model(model)
    return jsonify({"model": model, "ok": ok, "latency_ms": lat, "error": err})

@app.route("/models/search", methods=["POST"])
def models_search():
    p = request.json or {}
    query = p.get("query","" ); limit = int(p.get("limit", 15))
    role = p.get("role","reason")
    try:
        results = api.list_models(search=query, sort="downloads", direction=-1, limit=limit,
                                  filter=ModelFilter(task="text-generation"))
    except Exception:
        results = []
    out = []
    for m in results:
        if getattr(m, "private", False) or getattr(m, "gated", False):
            continue
        mid = m.modelId
        ok, lat, err = _probe_chat_model(mid, timeout=25)
        out.append({"model": mid, "downloads": getattr(m,"downloads",None),
                    "likes": getattr(m,"likes",None), "ok": ok, "latency_ms": lat, "error": err})
    ok_only = [r for r in out if r["ok"]]
    return jsonify({"query": query, "role": role, "candidates": out, "ok_candidates": ok_only})

# ---------------- SERVER + TUNNEL -------------------

def run_app():
    print(f"Starting Flask server on :{PORT}…")
    http_server = WSGIServer(('', PORT), app)
    http_server.serve_forever()

def start_cloudflared(port=PORT, max_wait=60):
    cmd = ["cloudflared","tunnel","--url",f"http://127.0.0.1:{port}","--no-autoupdate","--loglevel","info"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    url = None
    start = time.time()
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    while True:
        line = proc.stdout.readline()
        if line:
            m = pattern.search(line)
            if m:
                url = m.group(0)
                break
        if proc.poll() is not None:
            raise RuntimeError("cloudflared exited early")
        if time.time() - start > max_wait:
            raise TimeoutError("Timed out waiting for cloudflared URL")
    return proc, url


def keep_alive(interval=600):
    while True:
        try:
            if PUBLIC_URL:
                requests.get(f"{PUBLIC_URL}/health", timeout=10)
        except Exception:
            pass
        time.sleep(interval)



if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_app, daemon=True)
    flask_thread.start()
    try:
        CF_PROC, PUBLIC_URL = start_cloudflared(PORT)

        threading.Thread(target=keep_alive, daemon=True).start()


        print("\n" + "="*56)
        print("✅ NEXUS CORE BACKEND LIVE")
        print(f"🔗 Public URL: {PUBLIC_URL}")
        print("📋 Paste into frontend → Relay URL")
        print("="*56 + "\n")
    except Exception as e:
        print("❌ Tunnel failed:", e)
        CF_PROC = None
    try:
        flask_thread.join()
    except KeyboardInterrupt:
        if CF_PROC:
            try:
                CF_PROC.send_signal(signal.SIGINT)
            except Exception:
                pass
