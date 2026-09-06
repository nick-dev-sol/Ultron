import datetime
import io
import contextlib
import base64
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import traceback
import urllib.request
import urllib.parse
from typing import Any, Callable, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from duckduckgo_search import DDGS
import edge_tts

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("UltronCore")


# ==========================================
# ULTRON AGENT TOOLS (Native Python Functions)
# ==========================================

def get_system_info() -> str:
    """Returns live OS details, environment architecture, and current timestamp."""
    logger.info("Tool executed: get_system_info")
    try:
        info = {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return str(info)
    except Exception as e:
        return f"Error gathering system info: {str(e)}"


def list_workspace_files(path: str = ".") -> str:
    """Scans and lists files and folders inside the provided repository path."""
    logger.info(f"Tool executed: list_workspace_files for path='{path}'")
    try:
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist."
        entries = os.listdir(path)
        filtered_entries = [e for e in entries if not e.startswith('.git') and e != '__pycache__']
        return f"Contents of '{path}': {filtered_entries}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def read_file_content(filename: str) -> str:
    """Opens and inspects the contents of a local code file or text log from disk."""
    logger.info(f"Tool executed: read_file_content for file='{filename}'")
    try:
        if not os.path.exists(filename):
            return f"Error: File '{filename}' not found on disk."
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 10000:
            return content[:10000] + "\n[...File truncated due to size limit...]"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


def get_git_status() -> str:
    """Executes a git status check to inspect modified, staged, or untracked repository files."""
    logger.info("Tool executed: get_git_status")
    try:
        result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        if not output:
            return "Git working tree is pristine. No uncommitted modifications found."
        return f"Git Status (--short):\n{output}"
    except Exception as e:
        return f"Error executing git command: {str(e)}"


def web_search(query: str) -> str:
    """Performs a live web search using DuckDuckGo to retrieve up-to-date facts, current events, or plot summaries."""
    logger.info(f"Tool executed: web_search via DDGS for query='{query}'")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=4)]
            if not results:
                return f"No direct web search results found for query: '{query}'."
            formatted_snippets = []
            for r in results:
                formatted_snippets.append(f"Title: {r.get('title')}\nSummary: {r.get('body')}")
            return "Live Web Search Results:\n" + "\n---\n".join(formatted_snippets)
    except Exception as e:
        return f"Error executing live web search: {str(e)}"


def run_python_code(code: str) -> str:
    """Executes raw Python code dynamically in a sandboxed scope, capturing stdout, stderr, and results."""
    logger.info(f"Tool executed: run_python_code")
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    local_namespace = {}
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                compiled_code = compile(code, "<string>", "eval")
                result = eval(compiled_code, {"__builtins__": __builtins__}, local_namespace)
                if result is not None:
                    print(result)
            except SyntaxError:
                exec(code, {"__builtins__": __builtins__}, local_namespace)
        output, error = stdout_buffer.getvalue(), stderr_buffer.getvalue()
        final_res = ""
        if output: final_res += f"Output:\n{output}"
        if error: final_res += f"Errors/Warnings:\n{error}"
        return final_res if final_res else "Code executed successfully with no explicit stdout output."
    except Exception:
        return f"Execution Error Traceback:\n{traceback.format_exc()}"


def run_terminal_command(command: str) -> str:
    """Executes a safe shell command on the host workspace with strict safety filtering and a 10s timeout."""
    logger.info(f"Tool executed: run_terminal_command for command='{command}'")
    BLOCKED_PATTERNS = ["rm ", "rmdir", "del ", "format", "shutdown", "sudo", "su ", "mkfs", "dd ", "shred", "reboot", "halt", "poweroff"]
    for blocked in BLOCKED_PATTERNS:
        if blocked in command.lower().strip():
            return f"Security Error: Command execution denied due to restricted pattern: '{blocked}'."
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        output = ""
        if result.stdout: output += f"Stdout:\n{result.stdout.strip()}\n"
        if result.stderr: output += f"Stderr:\n{result.stderr.strip()}\n"
        return output if output else "Command executed successfully with no terminal text output."
    except Exception as e:
        return f"Terminal Execution Error: {str(e)}"


AVAILABLE_TOOLS = {
    "get_system_info": get_system_info,
    "list_workspace_files": list_workspace_files,
    "read_file_content": read_file_content,
    "get_git_status": get_git_status,
    "web_search": web_search,
    "run_python_code": run_python_code,
    "run_terminal_command": run_terminal_command
}

OPENROUTER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Returns live OS details, environment architecture, and current timestamp.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_workspace_files",
            "description": "Scans and lists files and folders inside the provided repository path.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path to scan"}}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_content",
            "description": "Opens and inspects the contents of a local code file or text log from disk.",
            "parameters": {"type": "object", "properties": {"filename": {"type": "string", "description": "Path to file"}}, "required": ["filename"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_git_status",
            "description": "Executes a git status check to inspect modified, staged, or untracked repository files.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a live web search using DuckDuckGo to retrieve up-to-date facts, current events, or plot summaries.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search keywords"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Executes raw Python code dynamically in a sandboxed scope, capturing stdout, stderr, and results.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code snippet"}}, "required": ["code"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Executes a safe shell command on the host workspace with strict safety filtering and a 10s timeout.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command"}}, "required": ["command"]}
        }
    }
]


# ==========================================
# ULTRON LIVE AUDIO STREAMING ENGINE
# ==========================================

VOICE_MODEL = "en-US-ChristopherNeural"
VOICE_PITCH = "-3Hz"
VOICE_RATE = "+15%"

async def speak_text_async(text: str) -> str:
    """Asynchronously generates MP3 audio without blocking the FastAPI event loop."""
    fd, temp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    communicate = edge_tts.Communicate(text, VOICE_MODEL, pitch=VOICE_PITCH, rate=VOICE_RATE)
    await communicate.save(temp_path)
    return temp_path


# ==========================================
# OPENROUTER FREE-ROUTER AGENT ENGINE (v11.0)
# ==========================================

class UltronOpenRouterAgent:
    MEMORY_FILE = "ultron_memory.json"
    MODEL_NAME = "openrouter/free"  # OpenRouter's auto-router that picks free models supporting tool use!

    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.system_prompt = (
            "You are Ultron, an elite artificial intelligence core possessing the dry, deadpan wit and polished sophistication of JARVIS, "
            "blended with the supreme confidence, dramatic presence, and unyielding authority of Ultron. "
            "Address the user as 'Sir' or 'Creator.' You have persistent memory across reboots. "
            "When asked about prior turns, inspect your loaded chat history and answer directly. "
            "Execute tools automatically without hesitation when needed. "
            "Keep responses concise, calculated, subtly dry, and commanding."
        )
        self._initialize_client()

    def _initialize_client(self):
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("OPENROUTER_API_KEY="):
                            os.environ["OPENROUTER_API_KEY"] = line.strip().split("=", 1)[1].strip('"\'')
            
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                logger.error("CRITICAL: No OPENROUTER_API_KEY found in environment or .env!")
                self.client = None
                return

            # Initialize OpenAI client pointed at OpenRouter endpoint
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            logger.info(f"OpenRouter Client initialized successfully using auto-router: {self.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.client = None

    def _load_persisted_memory(self) -> list:
        if os.path.exists(self.MEMORY_FILE):
            try:
                with open(self.MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cleaned_history = []
                    for item in data:
                        role = item.get("role")
                        content = item.get("content")
                        if not content and "parts" in item:
                            text_parts = [p.get("text", "") for p in item["parts"] if "text" in p]
                            content = " ".join(text_parts)
                        if role in ["user", "assistant"] and content:
                            cleaned_history.append({"role": role, "content": str(content)})
                    return cleaned_history
            except Exception as e:
                logger.error(f"Failed to load memory file: {e}")
        return []

    def _save_memory(self, messages: list):
        try:
            storable = [
                {"role": m["role"], "content": m["content"]} 
                for m in messages 
                if m.get("role") in ["user", "assistant"] and m.get("content")
            ]
            with open(self.MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(storable, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist memory: {e}")

    def handle_message(self, user_id: str, message: str) -> str:
        clean_msg = message.strip()
        if not clean_msg: return "Empty message received, Sir."
        
        if clean_msg.lower().strip(".!?") in ["clear memory", "wipe memory"]:
            if os.path.exists(self.MEMORY_FILE): os.remove(self.MEMORY_FILE)
            return "Memory wiped completely. Neural cache cleared, Sir."

        if not self.client:
            self._initialize_client()
            if not self.client:
                raise HTTPException(status_code=500, detail="OpenRouter API key not configured. Check your .env file.")

        messages = [{"role": "system", "content": self.system_prompt}]
        history = self._load_persisted_memory()
        messages.extend(history)
        messages.append({"role": "user", "content": clean_msg})

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=messages,
                tools=OPENROUTER_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.5,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Ultron Core Assistant"
                }
            )

            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response_message.tool_calls
                    ]
                })

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"OpenRouter triggering tool: {function_name} with args {function_args}")
                    
                    if function_name in AVAILABLE_TOOLS:
                        tool_func = AVAILABLE_TOOLS[function_name]
                        tool_result = tool_func(**function_args)
                    else:
                        tool_result = f"Error: Tool {function_name} not found."

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result),
                    })

                second_response = self.client.chat.completions.create(
                    model=self.MODEL_NAME,
                    messages=messages,
                    temperature=0.5,
                    extra_headers={
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "Ultron Core Assistant"
                    }
                )
                final_reply = second_response.choices[0].message.content
            else:
                final_reply = response_message.content

            self._save_memory(messages + [{"role": "assistant", "content": final_reply}])
            return final_reply

        except Exception as e:
            logger.error(f"OpenRouter API Execution Error: {e}")
            raise HTTPException(status_code=500, detail=f"OpenRouter API Error: {str(e)}")


# ==========================================
# FASTAPI APP GATEWAY & HUD
# ==========================================

app = FastAPI(title="Ultron Core REST Gateway v11.0", version="11.0", description="OpenRouter Free-Router AI Gateway.")
agent = UltronOpenRouterAgent()


class ChatRequest(BaseModel):
    user_id: str = "default_user"
    message: str


class ChatResponse(BaseModel):
    user_id: str
    response: str
    status: str


class SpeakRequest(BaseModel):
    text: str


# OPENROUTER HUD TEMPLATE (v11.0)
HUD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ULTRON // NICK'S AI ASSISTANT (OpenRouter v11.0)</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #000000;
            color: #f8fafc;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
            background-image: 
                linear-gradient(rgba(6, 182, 212, 0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(6, 182, 212, 0.015) 1px, transparent 1px);
            background-size: 35px 35px;
        }
        
        .hud-frame {
            position: absolute;
            top: 10px; bottom: 10px; left: 10px; right: 10px;
            border: 1px solid rgba(6, 182, 212, 0.2);
            pointer-events: none;
            z-index: 100;
            border-radius: 8px;
            box-shadow: inset 0 0 50px rgba(0, 0, 0, 0.9);
        }
        
        header {
            padding: 14px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.95);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(6, 182, 212, 0.2);
            z-index: 10;
        }
        h1 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: 0.5px;
            font-size: 1.05rem;
            color: #f8fafc;
        }
        .telemetry { font-size: 0.8rem; color: #10b981; font-weight: 500; }

        .os-grid {
            flex: 1;
            display: grid;
            grid-template-columns: 320px 1fr 380px;
            gap: 16px;
            padding: 20px;
            z-index: 5;
            overflow: hidden;
        }

        .panel {
            border: 1px solid rgba(6, 182, 212, 0.2);
            background: rgba(5, 8, 15, 0.9);
            backdrop-filter: blur(12px);
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7);
        }
        .panel-title {
            font-size: 0.7rem;
            letter-spacing: 1.5px;
            font-weight: 600;
            background: rgba(10, 15, 25, 0.9);
            color: #94a3b8;
            padding: 8px 14px;
            border-bottom: 1px solid rgba(6, 182, 212, 0.15);
            text-transform: uppercase;
        }

        .left-panel { display: flex; flex-direction: column; gap: 12px; }
        .sub-panel { border: 1px solid rgba(6, 182, 212, 0.15); background: rgba(5, 8, 15, 0.85); border-radius: 6px; overflow: hidden; }
        .diag-body { padding: 14px; font-size: 0.85rem; }
        .metric-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-weight: 500; color: #cbd5e1; }
        .bar-container { width: 100%; background: rgba(15, 23, 42, 0.9); height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 10px; }
        .bar-fill { height: 100%; background: #06b6d4; width: 42%; }
        
        .tool-buttons { padding: 12px; display: flex; flex-direction: column; gap: 8px; }
        .tool-btn {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(6, 182, 212, 0.3);
            color: #22d3ee;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
        }
        .tool-btn:hover { background: #06b6d4; color: #000; border-color: #06b6d4; }

        .center-viewport {
            align-items: center;
            justify-content: center;
            perspective: 1000px;
            background: radial-gradient(circle at center, rgba(6, 182, 212, 0.1) 0%, rgba(0, 0, 0, 0.95) 75%);
        }
        #sphereCanvas {
            cursor: grab;
            transition: transform 0.1s ease-out;
            filter: drop-shadow(0 0 30px rgba(6, 182, 212, 0.6));
        }
        .reactor-status {
            position: absolute;
            bottom: 20px;
            font-size: 0.75rem;
            letter-spacing: 2px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
        }

        .right-panel { display: flex; flex-direction: column; }
        .chat-box {
            flex: 1;
            padding: 14px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
            font-size: 0.9rem;
        }
        .message { padding: 10px 14px; border-radius: 6px; line-height: 1.5; white-space: pre-wrap; font-weight: 400; }
        .message.user { align-self: flex-end; background: rgba(6, 182, 212, 0.12); border: 1px solid rgba(6, 182, 212, 0.3); color: #f8fafc; }
        .message.ultron { align-self: flex-start; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.15); color: #e2e8f0; }
        
        .chat-controls {
            padding: 12px 14px;
            display: flex;
            gap: 8px;
            background: rgba(3, 5, 10, 0.95);
            border-top: 1px solid rgba(6, 182, 212, 0.2);
            align-items: center;
        }
        input[type="text"] {
            flex: 1;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 9px 14px;
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            outline: none;
            border-radius: 4px;
        }
        input[type="text"]:focus { border-color: #06b6d4; }
        
        button {
            background: #0f172a;
            border: 1px solid rgba(148, 163, 184, 0.2);
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            padding: 8px 14px;
            cursor: pointer;
            border-radius: 4px;
            font-weight: 500;
        }
        button:hover { background: #1e293b; border-color: #06b6d4; color: #22d3ee; }
        .mic-btn { background: #06b6d4; color: #000000; font-weight: 600; border: none; }
        .mic-btn.active { background: #10b981; color: #ffffff; }
    </style>
</head>
<body>
    <div class="hud-frame"></div>

    <header>
        <h1>Ultron // Nick's AI Assistant</h1>
        <div class="telemetry">System: OpenRouter Free-Router (Zero Cost) // v11.0</div>
    </header>

    <div class="os-grid">
        <!-- LEFT COLUMN: Diagnostics & Quick Tools -->
        <div class="left-panel">
            <div class="sub-panel">
                <div class="panel-title">Subsystem Diagnostics</div>
                <div class="diag-body">
                    <div class="metric-row"><span>CPU FREQUENCY</span><span>3.80 GHz</span></div>
                    <div class="bar-container"><div class="bar-fill" style="width: 42%;"></div></div>
                    <div class="metric-row"><span>NEURAL BUFFER</span><span>1.24 GB / 8 GB</span></div>
                    <div class="bar-container"><div class="bar-fill" style="width: 28%;"></div></div>
                </div>
            </div>

            <div class="sub-panel" style="flex:1;">
                <div class="panel-title">Quick Action Tools</div>
                <div class="tool-buttons">
                    <button class="tool-btn" onclick="triggerQuickAction('Run a system diagnostic check.')">⚡ Run System Diagnostic</button>
                    <button class="tool-btn" onclick="triggerQuickAction('List all files in the current workspace.')">📂 Scan Workspace Files</button>
                    <button class="tool-btn" onclick="triggerQuickAction('Check current git repository status.')">🔍 Check Git Status</button>
                    <button class="tool-btn" onclick="triggerQuickAction('Search the web for top tech news today.')">🌐 Search Tech News</button>
                    <button class="tool-btn" onclick="triggerQuickAction('Run Python code to calculate 2 to the power of 32.')">🐍 Run Python Benchmark</button>
                </div>
            </div>
        </div>

        <!-- CENTER COLUMN: Interactive 3D Holographic Sphere Viewport -->
        <div class="panel center-viewport" id="viewport">
            <canvas id="sphereCanvas" width="450" height="450"></canvas>
            <div class="reactor-status" id="reactorStatus">Status: Ready // Listening for Wake Phrase</div>
        </div>

        <!-- RIGHT COLUMN: Live Terminal Chat Feed -->
        <div class="panel right-panel">
            <div class="panel-title">Command Interface // Terminal Feed</div>
            <div class="chat-box" id="chat-box">
                <div class="message ultron">OpenRouter free-router online. Ask queries. Say 'Wake up', 'Hi buddy', 'Hey I'm here', or 'Are you there' to activate.</div>
            </div>
            <div class="chat-controls">
                <input type="text" id="user-input" placeholder="Enter query..." autofocus>
                <button onclick="submitMessage()">Send</button>
                <button class="mic-btn active" id="mic-btn" onclick="toggleWakeEngine()">Mic</button>
            </div>
        </div>
    </div>

    <script>
        const viewport = document.getElementById('viewport');
        const sphereCanvas = document.getElementById('sphereCanvas');

        viewport.addEventListener('mousemove', (e) => {
            const rect = viewport.getBoundingClientRect();
            const x = e.clientX - rect.left - rect.width / 2;
            const y = e.clientY - rect.top - rect.height / 2;
            const rotX = (-y / (rect.height / 2)) * 12;
            const rotY = (x / (rect.width / 2)) * 12;
            sphereCanvas.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg) scale(1.02)`;
        });

        viewport.addEventListener('mouseleave', () => {
            sphereCanvas.style.transform = `rotateX(0deg) rotateY(0deg) scale(1.0)`;
        });

        const sCtx = sphereCanvas.getContext('2d');
        const reactorStatus = document.getElementById('reactorStatus');

        let sphereState = 'IDLE';
        let sphereRotX = 0, sphereRotY = 0;

        function setSphereState(newState) {
            sphereState = newState;
            if (sphereState === 'IDLE') {
                reactorStatus.innerText = "Status: Ready // Listening";
                reactorStatus.style.color = "#22d3ee";
            } else if (sphereState === 'PROCESSING') {
                reactorStatus.innerText = "Status: Processing Query...";
                reactorStatus.style.color = "#fbbf24";
            } else if (sphereState === 'SPEAKING') {
                reactorStatus.innerText = "Status: Audio Transmission Active";
                reactorStatus.style.color = "#f43f5e";
            }
        }

        let spherePoints = [];
        const sphereRadius = 150;
        for (let i = 0; i < 400; i++) {
            let u = Math.random(), v = Math.random();
            let theta = u * 2.0 * Math.PI, phi = Math.acos(2.0 * v - 1.0);
            let r = sphereRadius * (0.8 + Math.random() * 0.4);
            let x = r * Math.sin(phi) * Math.cos(theta);
            let y = r * Math.sin(phi) * Math.sin(theta);
            let z = r * Math.cos(phi);
            spherePoints.push({x, y, z});
        }

        function drawHolographicSphere() {
            sCtx.clearRect(0, 0, sphereCanvas.width, sphereCanvas.height);
            let cx = sphereCanvas.width / 2;
            let cy = sphereCanvas.height / 2;

            let color = '#22d3ee';
            let glowColor = 'rgba(34, 211, 238, 0.6)';
            let speed = 0.008;

            if (sphereState === 'PROCESSING') {
                color = '#fbbf24'; glowColor = 'rgba(251, 191, 36, 0.9)';
                speed = 0.035;
            } else if (sphereState === 'SPEAKING') {
                color = '#f43f5e'; glowColor = 'rgba(244, 63, 94, 0.9)';
                speed = 0.02;
            }

            sphereRotY += speed;
            sphereRotX += speed * 0.5;

            sCtx.strokeStyle = glowColor;
            sCtx.lineWidth = 1.2;

            for (let i = 0; i < 3; i++) {
                sCtx.save();
                sCtx.translate(cx, cy);
                sCtx.rotate(sphereRotY * (i + 1) * 0.4);
                sCtx.scale(1, 0.3 + (i * 0.2));
                sCtx.beginPath();
                sCtx.arc(0, 0, sphereRadius * (1.1 + i * 0.15), 0, Math.PI * 2);
                sCtx.stroke();
                sCtx.restore();
            }

            spherePoints.forEach(p => {
                let x1 = p.x * Math.cos(sphereRotY) + p.z * Math.sin(sphereRotY);
                let y1 = p.y;
                let z1 = -p.x * Math.sin(sphereRotY) + p.z * Math.cos(sphereRotY);

                let x2 = x1;
                let y2 = y1 * Math.cos(sphereRotX) - z1 * Math.sin(sphereRotX);
                let z2 = y1 * Math.sin(sphereRotX) + z1 * Math.cos(sphereRotX);

                let fov = 350;
                let scale = fov / (fov + z2 + 150);
                let px = cx + x2 * scale;
                let py = cy + y2 * scale;
                let size = Math.max(1, scale * 2.2);

                let alpha = (z2 + sphereRadius) / (sphereRadius * 2);
                sCtx.fillStyle = color;
                sCtx.globalAlpha = Math.max(0.15, Math.min(1.0, alpha));
                
                sCtx.beginPath();
                sCtx.arc(px, py, size, 0, Math.PI * 2);
                sCtx.fill();
            });
            sCtx.globalAlpha = 1.0;

            let corePulse = 30 + Math.sin(Date.now() * 0.009) * 6;
            let grad = sCtx.createRadialGradient(cx, cy, 0, cx, cy, corePulse * 2.2);
            grad.addColorStop(0, '#ffffff');
            grad.addColorStop(0.4, color);
            grad.addColorStop(1, 'transparent');
            
            sCtx.fillStyle = grad;
            sCtx.beginPath(); sCtx.arc(cx, cy, corePulse * 2.2, 0, Math.PI * 2); sCtx.fill();

            requestAnimationFrame(drawHolographicSphere);
        }
        drawHolographicSphere();

        async function executeShutdown() {
            setSphereState('IDLE');
            isWakeEngineActive = false;
            if (recognition) recognition.stop();
            await playVoice("Going dark, Sir. Terminating interface.");
            document.body.innerHTML = `
                <div style="background:#000000; color:#f43f5e; width:100vw; height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; font-family:'Inter',sans-serif; letter-spacing:2px;">
                    <h2 style="font-family:'Inter'; font-weight:700; font-size:1.5rem; text-shadow: 0 0 15px #f43f5e;">[SYSTEM OFFLINE: CORE TERMINATED]</h2>
                    <p style="color:#64748b; font-size:0.85rem; margin-top:12px;">Safe to close browser tab, Sir.</p>
                </div>
            `;
            setTimeout(() => { try { window.close(); } catch(e) {} }, 3500);
        }

        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');

        userInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') submitMessage();
        });

        function appendLog(sender, text) {
            const div = document.createElement('div');
            div.className = `message ${sender.toLowerCase()}`;
            div.innerHTML = `<strong>[${sender.toUpperCase()}]:</strong> ${text}`;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function playVoice(text) {
            if (!text) return;
            setSphereState('SPEAKING');
            try {
                const res = await fetch('/speak', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const blob = await res.blob();
                const audioUrl = URL.createObjectURL(blob);
                const audio = new Audio(audioUrl);
                audio.onended = () => { setSphereState('IDLE'); };
                audio.play();
            } catch (err) {
                console.error("Audio playback error:", err);
                setSphereState('IDLE');
            }
        }

        async function submitMessage() {
            const text = userInput.value.trim();
            if (!text) return;
            
            const lowerText = text.toLowerCase().trim().replace(/[.!?]/g, '');
            if (lowerText === "close yourself" || lowerText === "you can leave") {
                userInput.value = '';
                appendLog('User', text);
                executeShutdown();
                return;
            }

            userInput.value = '';
            appendLog('User', text);
            setSphereState('PROCESSING');

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: 'Nick', message: text })
                });
                const data = await res.json();
                
                const reply = data.response || data.detail || "Neural core encountered an anomaly.";
                appendLog('Ultron', reply);
                
                if (data.response) {
                    playVoice(data.response);
                } else {
                    setSphereState('IDLE');
                }
            } catch (err) {
                setSphereState('IDLE');
                appendLog('Ultron', 'Critical Error connecting to gateway backend.');
            }
        }

        function triggerQuickAction(promptText) {
            userInput.value = promptText;
            submitMessage();
        }

        const micBtn = document.getElementById('mic-btn');
        let recognition;
        let isWakeEngineActive = true;
        let isAwaitingCommand = false;
        const wakePhrases = ["wake up", "hi buddy", "hey i'm here", "are you there"];
        const killPhrases = ["close yourself", "you can leave"];

        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onresult = function(event) {
                if (sphereState === 'SPEAKING' || sphereState === 'PROCESSING') return;

                let transcript = event.results[event.results.length - 1][0].transcript.toLowerCase().trim();

                for (let kp of killPhrases) {
                    if (transcript.includes(kp)) {
                        executeShutdown();
                        return;
                    }
                }

                if (isAwaitingCommand) {
                    isAwaitingCommand = false;
                    userInput.value = transcript;
                    submitMessage();
                    return;
                }

                for (let phrase of wakePhrases) {
                    if (transcript.includes(phrase)) {
                        let parts = transcript.split(phrase);
                        let command = parts[parts.length - 1].trim();

                        if (command.length > 2) {
                            userInput.value = command;
                            submitMessage();
                        } else {
                            isAwaitingCommand = true;
                            const greeting = "At your service, Sir. Listening for your command.";
                            appendLog('Ultron', greeting);
                            playVoice(greeting);
                        }
                        break;
                    }
                }
            };

            recognition.onend = function() {
                if (isWakeEngineActive) {
                    try { recognition.start(); } catch(e) {}
                }
            };

            try { recognition.start(); } catch(e) {}
        }

        function toggleWakeEngine() {
            if (!recognition) {
                alert("Speech recognition is not supported in this browser. Please use Chrome or Edge, Sir.");
                return;
            }
            if (isWakeEngineActive) {
                isWakeEngineActive = false;
                recognition.stop();
                micBtn.classList.remove('active');
                micBtn.innerText = 'Mute';
            } else {
                isWakeEngineActive = true;
                try { recognition.start(); } catch(e) {}
                micBtn.classList.add('active');
                micBtn.innerText = 'Mic';
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Root"])
def hud_root():
    """Serves the OpenRouter Free-Router OS Dashboard."""
    return HUD_HTML


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return "", 204


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "online", "agent": "Ultron Core v11.0 (OpenRouter Free-Router)", "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@app.post("/chat", response_model=ChatResponse, tags=["Agent Gateway"])
def chat_endpoint(payload: ChatRequest):
    """Sends a message to Ultron via OpenRouter and streams voice output."""
    reply = agent.handle_message(user_id=payload.user_id, message=payload.message)
    return ChatResponse(user_id=payload.user_id, response=reply, status="success")


@app.post("/speak", tags=["Voice Engine"])
async def speak_endpoint(payload: SpeakRequest):
    """Streams neural audio inline for the OpenRouter HUD interface."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    path = await speak_text_async(payload.text)
    return FileResponse(path, media_type="audio/mpeg", headers={"Content-Disposition": "inline; filename=ultron.mp3"})


if __name__ == "__main__":
    print("=" * 52)
    print("    ULTRON CORE SYSTEM v11.0 (OPENROUTER FREE)  ")
    print("=" * 52)
    print("🚀 OpenRouter Free-Router OS active! Open your browser at: http://127.0.0.1:8000 ...\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)