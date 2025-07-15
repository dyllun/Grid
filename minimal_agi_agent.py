import os
import openai
import time
import json

import subprocess
# Basic memory to store conversation history
class Memory:
    def __init__(self):
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})

    def context(self):
        return self.messages

# Basic tool to execute shell commands (unsafe; for demo only)
class ShellTool:
    def run(self, command):
        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True, timeout=30)
            return output.strip()
        except Exception as e:
            return f"Error: {e}"

# Simple AGI-like agent using OpenAI's ChatCompletion API
class Agent:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.memory = Memory()
        self.tool = ShellTool()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def ask(self, prompt):
        self.memory.add("user", prompt)
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.memory.context(),
        )
        reply = completion.choices[0].message["content"]
        self.memory.add("assistant", reply)
        return reply

    def interactive_loop(self):
        print("Interactive AGI agent. Type 'exit' to quit.")
        while True:
            inp = input("You: ")
            if inp.strip().lower() == "exit":
                break
            if inp.startswith("!shell "):
                cmd = inp[len("!shell "):]
                result = self.tool.run(cmd)
                print(result)
                continue
            response = self.ask(inp)
            print(response)

if __name__ == "__main__":
    agent = Agent()
    if not openai.api_key:
        print("OPENAI_API_KEY environment variable not set.")
    agent.interactive_loop()
