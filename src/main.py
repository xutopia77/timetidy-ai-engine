"""
AI Engine — stdin/stdout JSON-line protocol.

Protocol (one JSON object per line):
  Request:  {"id": "...", "cmd": "generate_tags", "data": {...}}
  Response: {"id": "...", "cmd": "generate_tags", "code": 0, "status": "success",
             "data": {...}, "data_dbg": {...}}

All cmd values are unique (flat namespace). Routing happens here in handle_request().
Engines receive {cmd, ...data_fields} — cmd is the wire command, used directly for
engine-internal dispatch inside engine.process().
"""

import sys
import json
import signal
import traceback
import os
import time
import threading
import queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from utils.logger import log, set_log_level, set_log_file
from engines.search_engine_faiss import SearchEngine
from engines.index_init_engine import IndexInitEngine
from engines.hierarchical_tag_engine import HierarchicalTagEngine
from engines.similarity_engine import SimilarityEngine
import config


class AIEngine:
    def __init__(self):
        self.engines: dict[str, object] = {}
        self.initialized = False
        self.request_queue = queue.Queue()
        self.running = True
        self.stdout_lock = threading.Lock()
        self.last_ping_time = time.time()
        self.ping_timeout = 120
        self.ping_check_interval = 5

    # ---- init ----

    def init(self, config_data: dict = None) -> bool:
        """Initialize AI Engine with optional runtime config."""
        if config_data:
            log.info(f"Loading runtime config: {config_data}")
            config.RUNTIME_CONFIG.load_from_dict(config_data)

            if config.RUNTIME_CONFIG.log_level:
                log_level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
                set_log_level(log_level_map.get(config.RUNTIME_CONFIG.log_level, 20))

            if config.RUNTIME_CONFIG.log_file:
                set_log_file(config.RUNTIME_CONFIG.log_file)

        if not config.RUNTIME_CONFIG.is_configured():
            missing = config.RUNTIME_CONFIG.get_missing_config()
            log.warning(f"Configuration incomplete. Missing: {', '.join(missing)}")
            return False

        if not config.RUNTIME_CONFIG.auto_detect_model():
            log.error("No model detected! Please place models in ai_engine/models/H, models/M, or models/L")
            return False

        log.info(f"Auto-detected model: tier={config.get_model_tier()}, name={config.get_clip_model_name()}")
        log.info("Initializing AI Engine...")

        self.engines["search"] = SearchEngine()
        self.engines["index_init"] = IndexInitEngine()
        self.engines["tag_detect"] = HierarchicalTagEngine()
        self.engines["find_similar"] = SimilarityEngine()

        self.initialized = True
        log.info(f"AI Engine initialized. Available engines: {list(self.engines.keys())}")
        return True

    # ---- wire helpers ----

    def send_response(self, response: dict):
        """Thread-safe JSON-line response to stdout."""
        with self.stdout_lock:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    @staticmethod
    def _ok(id_: str, cmd: str, data: dict = None, data_dbg: dict = None) -> dict:
        resp: dict = {"id": id_, "cmd": cmd, "code": 0, "status": "success"}
        if data is not None:
            resp["data"] = data
        if data_dbg:
            resp["data_dbg"] = data_dbg
        return resp

    @staticmethod
    def _err(id_: str, cmd: str, message: str) -> dict:
        return {"id": id_, "cmd": cmd, "code": 1, "status": message}

    # ---- ping (handled inline in stdin reader) ----

    def handle_ping(self, request: dict):
        self.last_ping_time = time.time()
        self.send_response(self._ok(request.get("id", ""), "ping", {"message": "pong"}))

    def _ping_check_loop(self):
        """Background thread: exit if Go process stops sending pings."""
        while self.running:
            time.sleep(self.ping_check_interval)
            elapsed = time.time() - self.last_ping_time
            if elapsed > self.ping_timeout:
                log.warning(f"Ping timeout: {elapsed:.1f}s > {self.ping_timeout}s, exiting...")
                os._exit(0)

    def start_ping_check(self):
        t = threading.Thread(target=self._ping_check_loop, daemon=True)
        t.start()
        log.info(f"Ping check started (timeout={self.ping_timeout}s, interval={self.ping_check_interval}s)")

    # ---- request routing ----

    def _ensure_ready(self, req_id: str, cmd: str) -> dict | None:
        """Check config/init; returns error dict or None if ready."""
        if not config.RUNTIME_CONFIG.is_configured():
            missing = config.RUNTIME_CONFIG.get_missing_config()
            msg = f"Configuration incomplete. Missing: {', '.join(missing)}. Call cfg_set first."
            log.error(msg)
            return self._err(req_id, cmd, msg)
        if not self.initialized:
            log.info("AI Engine not initialized, initializing...")
            if not self.init():
                missing = config.RUNTIME_CONFIG.get_missing_config()
                return self._err(req_id, cmd, f"Configuration incomplete. Missing: {', '.join(missing)}")
        return None

    def handle_request(self, request: dict) -> dict:
        cmd = request.get("cmd", "")
        req_id = request.get("id", "")
        params = request.get("data", {}) or {}
        params["cmd"] = cmd  # engine.process() dispatches on params["cmd"]

        log.info(f"request: cmd={cmd}, id={req_id}")

        # ── cfg_set (no engine needed, no config check) ──
        if cmd == "cfg_set":
            ok = self.init(params)
            if ok:
                return self._ok(req_id, cmd, {"message": "AI Engine configured"})
            missing = config.RUNTIME_CONFIG.get_missing_config()
            return self._err(req_id, cmd, f"Configuration incomplete. Missing: {', '.join(missing)}")

        # ── guard: config + init ──
        err = self._ensure_ready(req_id, cmd)
        if err:
            return err

        # ── dispatch ──
        try:
            if cmd in ("generate_tags", "get_tag_config", "get_default_tag_config",
                       "save_tag_config", "reset_tag_config"):
                result = self.engines["tag_detect"].process(params)

            elif cmd in ("search_by_text", "encode_image", "encode_text"):
                result = self.engines["search"].process(params)

            elif cmd in ("find_similar", "detect_similar_groups"):
                result = self.engines["find_similar"].process(params)

            elif cmd in ("index_start", "index_stop", "index_get_progress", "index_get_status"):
                result = self.engines["index_init"].process(params)

            else:
                return self._err(req_id, cmd, f"Unknown cmd: {cmd}")

            data = result.get("data", result)
            data_dbg = result.get("data_dbg")
            log.info(f"response: cmd={cmd}, id={req_id}, status=success")
            return self._ok(req_id, cmd, data, data_dbg)

        except Exception as e:
            traceback.print_exc()
            return self._err(req_id, cmd, str(e))

    # ---- worker thread ----

    def process_request_worker(self):
        while self.running:
            try:
                request = self.request_queue.get(timeout=1)
                if request is None:
                    break
                response = self.handle_request(request)
                self.send_response(response)
            except queue.Empty:
                continue
            except Exception as e:
                traceback.print_exc()
                self.send_response({"id": "", "code": 1, "status": str(e)})


# ---- entry point ----

def main():
    engine = AIEngine()

    worker = threading.Thread(target=engine.process_request_worker, daemon=True)
    worker.start()

    engine.start_ping_check()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            if request.get("cmd") == "ping":
                engine.handle_ping(request)
            else:
                engine.request_queue.put(request)
        except json.JSONDecodeError as e:
            engine.send_response({"id": "", "code": 1, "status": f"Invalid JSON: {e}"})
        except Exception as e:
            traceback.print_exc()
            engine.send_response({"id": "", "code": 1, "status": str(e)})


if __name__ == "__main__":
    main()
