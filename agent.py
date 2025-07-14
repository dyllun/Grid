import os, sys, subprocess, importlib.util, textwrap, json, time, socket, threading, random
from typing import Any, Callable, Dict, List

TOOLS_FILE = "user_tools.py"
STATE_FILE = "agent_state.json"

def require(pkg, pip_name=None, optional=False):
    """Import a package, installing it if missing.

    If installation or import fails and ``optional`` is True, ``None`` is
    returned instead of raising an exception."""
    try:
        return __import__(pkg)
    except ImportError:
        pip_name = pip_name or pkg
        try:
            print(f"Installing {pip_name}…")
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)
        except Exception as e:
            print(f"Failed to install {pip_name}: {e}")
            if optional:
                return None
            raise
        try:
            return __import__(pkg)
        except Exception as e:
            print(f"Failed to import {pkg}: {e}")
            if optional:
                return None
            raise

requests = require("requests")

torch = require("torch")
transformers = require("transformers")
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options as ChromeOptions
except ImportError:
    require("selenium")
    require("webdriver-manager")
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
        if not callable(func):
            return f"Tool {name} not found"
        try:
            return func(*args)
        except Exception as e:
            return f"Error running {name}: {e}"

class StateManager:
    def __init__(self, file=STATE_FILE):
        self.file = file
        self.history = []
        self.load()
    def load(self):
        if os.path.exists(self.file):
            try:
                self.history = json.load(open(self.file))
            except:
                self.history = []
    def save(self, history):
        try:
            json.dump(history, open(self.file, "w"))
        except:
            pass

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
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return e.output

def run_browser(url, script="", screenshot=""):
    try:
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        result = ""
        if script:
            result = driver.execute_script(script)
        if screenshot:
            driver.save_screenshot(screenshot)
        page_source = driver.page_source
        driver.quit()
        return (result or "") + "\n\n" + page_source[:2000]
    except Exception as e:
        return f"Error running browser: {e}"

class LLMBackend:
    """Backend that relies solely on an open source model via transformers."""

    def __init__(self, model_name: str = None):
        self.local_model_name = model_name or os.environ.get("LOCAL_MODEL", "distilgpt2")
        self.pipeline = None

        if transformers is None or torch is None:
            raise RuntimeError("Local backend requires transformers and torch")

        self.configure_local()

    # Local model helpers
    def configure_local(self):
        if transformers is None or torch is None:
            raise RuntimeError("Local backend requires transformers and torch")

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.local_model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(self.local_model_name)
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = transformers.pipeline(
                "text-generation", model=model, tokenizer=tokenizer, device=device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load local model {self.local_model_name}: {e}")

    def ask_llm(self, messages, temperature=0.2, max_tokens=1024):
        if self.pipeline is None:
            raise RuntimeError("Local model pipeline not configured")

        prompt = "\n".join(m["content"] for m in messages)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )
        text = outputs[0]["generated_text"][len(prompt) :].strip()
        return {"choices": [{"message": {"content": text}}]}

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
