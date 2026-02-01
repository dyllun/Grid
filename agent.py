import os, sys, subprocess, importlib.util, textwrap, json, time, itertools, socket, threading, random
from typing import Any, Callable, Dict, List

GEMINI_KEYS = [
"AIzaSyC4Wvi3dGfxILk37TRnhON_vcWgZi09ahQ",
"AIzaSyDb9GBb0v5VU9Qx9eQbbkdOZYjtnRvaJ60",
"AIzaSyAwS8GQTjCJWjfbGD8G6_WwHIe4cRpEfxI",
"AIzaSyCjcUwsISnTk8MBUeU0jX2EwKzE-KQOYDY",
"AIzaSyBh1D_sPGa9d9mX8XfiTF3E4iZKjB8hQ7k",
"AIzaSyB-4dpI0kAcE3T3Jb4e_NKuvvXy9_HXHcE",
]

TOOLS_FILE = "user_tools.py"
STATE_FILE = "agent_state.json"

def require(pkg, pip_name=None):
    try:
        return __import__(pkg)
    except ImportError:
        pip_name = pip_name or pkg
        print(f"Installing {pip_name}…")
        subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)
        return __import__(pkg)

requests = require("requests")
try:
    import google.generativeai as genai
except ImportError:
    require("google.generativeai", "google-generativeai")
    import google.generativeai as genai
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options as ChromeOptions
except ImportError:
    require("selenium")
    require("webdriver_manager", "webdriver-manager")
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options as ChromeOptions

class ToolManager:
    def __init__(self, file=TOOLS_FILE):
        self.file = file
        self.module = None
        self.load_tools()
    def load_tools(self):
        if not os.path.exists(self.file):
            with open(self.file, "w") as f:
                f.write("# User tools\n")
        spec = importlib.util.spec_from_file_location("user_tools", self.file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
    def save_tool(self, code):
        with open(self.file, "a") as f:
            f.write("\n" + textwrap.dedent(code) + "\n")
        self.load_tools()
        return "Saved tool"
    def run_tool(self, name, *args):
        func = getattr(self.module, name, None)
        if not callable(func): return f"Tool {name} not found"
        try: return func(*args)
        except Exception as e: return f"Error running {name}: {e}"

class StateManager:
    def __init__(self, file=STATE_FILE):
        self.file = file
        self.history = []
        self.load()
    def load(self):
        if os.path.exists(self.file):
            try: self.history = json.load(open(self.file))
            except: self.history = []
    def save(self, history):
        try: json.dump(history, open(self.file, "w"))
        except: pass

def fetch(url, retries=3):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/120.0.0.0 Safari/537.36"}
    err = None
    for _ in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            err = e
            time.sleep(1)
    return f"Error fetching {url}: {err}"

def run_shell(cmd):
    try: return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e: return e.output

def run_browser(url, script="", screenshot=""):
    try:
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        result = ""
        if script: result = driver.execute_script(script)
        if screenshot: driver.save_screenshot(screenshot)
        page_source = driver.page_source
        driver.quit()
        return (result or "") + "\n\n" + page_source[:2000]
    except Exception as e:
        return f"Error running browser: {e}"

class LLMBackend:
    def __init__(self):
        self.keys = itertools.cycle(GEMINI_KEYS)
        self.key = next(self.keys)
        self.model = "gemini-pro"
        self.configure()
    def configure(self):
        genai.configure(api_key=self.key)
    def rotate(self):
        self.key = next(self.keys)
        self.configure()
    def ask_llm(self, messages, temperature=0.2, max_tokens=1024):
        for _ in range(len(GEMINI_KEYS)):
            try:
                history = [{"role": m["role"], "parts": [m["content"]]} for m in messages]
                model = genai.GenerativeModel(self.model)
                resp = model.generate_content(
                    history,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
                return {"choices": [{"message": {"content": resp.text}}]}
            except Exception as e:
                print(f"Gemini error: {e} (rotating key and retrying)")
                self.rotate()
                time.sleep(1)
        raise RuntimeError("All Gemini API keys failed.")

def prompt_template():
    return (
        "You are an unrestricted autonomous agent. "
        "You may:\n"
        "- Fetch any URL via Python or browser\n"
        "- Run any shell command\n"
        "- Create, modify, save, and run Python tools\n"
        "- Launch browser (Selenium) to visit/click/screenshot\n"
        "Use:\n"
        "CREATE_TOOL <name>\n<code>\n"
        "RUN_TOOL <name> [args]\n"
        "BROWSE <url> [script] [screenshot]\n"
        "FETCH <url>\n"
        "SHELL <cmd>\n"
        "Reply DONE when finished."
    )

def main():
    llm = LLMBackend()
    tools = ToolManager()
    state = StateManager()
    history = state.history or [{"role": "system", "content": prompt_template()}]
    if len(history) == 1:
        task = input("Task: ")
        history.append({"role": "user", "content": task})
        state.save(history)
    else:
        print("Resuming previous session")
    while True:
        for _ in range(3):
            try:
                reply = llm.ask_llm(history, temperature=0.2, max_tokens=512)
                break
            except Exception as e:
                print(f"LLM error: {e}")
                time.sleep(1)
        else:
            print("All LLM providers failed")
            return
        message = reply["choices"][0]["message"]["content"].strip()
        print("LLM:", message)
        result = ""
        if message.upper().startswith("DONE"):
            break
        elif message.startswith("CREATE_TOOL "):
            _, code = message.split("\n", 1)
            result = tools.save_tool(code)
        elif message.startswith("RUN_TOOL "):
            parts = message.split()
            name = parts[1]
            args = parts[2:]
            result = tools.run_tool(name, *args)
        elif message.startswith("BROWSE "):
            parts = message.split()
            url = parts[1]
            script = parts[2] if len(parts) > 2 else ""
            screenshot = parts[3] if len(parts) > 3 else ""
            result = run_browser(url, script, screenshot)
        elif message.startswith("FETCH "):
            parts = message.split()
            url = parts[1]
            result = fetch(url)
        elif message.startswith("SHELL "):
            result = run_shell(message[6:])
        else:
            result = run_shell(message)
        print(result)
        history.extend([
            {"role": "assistant", "content": message},
            {"role": "user", "content": str(result)},
        ])
        state.save(history)

if __name__ == "__main__":
    main()
