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
        self.engines = {}
        self.initialized = False
        self.request_queue = queue.Queue()
        self.running = True
        self.stdout_lock = threading.Lock()
        self.last_ping_time = time.time()
        self.ping_timeout = 120  # Exit if no ping for 120s (Go process may have crashed)
        self.ping_check_interval = 5

    def init(self, config_data: dict = None):
        """Initialize AI Engine with optional runtime config"""
        # Load runtime config if provided
        if config_data:
            log.info(f"Loading runtime config: {config_data}")
            config.RUNTIME_CONFIG.load_from_dict(config_data)

            # Apply log settings
            if config.RUNTIME_CONFIG.log_level:
                log_level_map = {
                    "DEBUG": 10,
                    "INFO": 20,
                    "WARNING": 30,
                    "ERROR": 40,
                }
                set_log_level(log_level_map.get(config.RUNTIME_CONFIG.log_level, 20))

            if config.RUNTIME_CONFIG.log_file:
                set_log_file(config.RUNTIME_CONFIG.log_file)

        # Check if configuration is complete
        if not config.RUNTIME_CONFIG.is_configured():
            missing = config.RUNTIME_CONFIG.get_missing_config()
            log.warning(f"Configuration incomplete. Missing: {', '.join(missing)}")
            return False

        # Auto-detect model tier (H > M > L priority)
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
        log.info(f"AI Engine initialized successfully. Available actions: {list(self.engines.keys())}")
        return True

    def send_response(self, response: dict):
        """Thread-safe response sending"""
        with self.stdout_lock:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    def handle_ping(self, request: dict):
        """Handle ping request directly in stdin reader thread for immediate response"""
        self.last_ping_time = time.time()
        response = {
            "id": request.get("id", ""),
            "action": "ping",
            "code": 0,
            "status": "success",
            "data": {"message": "pong"},
        }
        self.send_response(response)

    def _ping_check_loop(self):
        """Background thread to check ping timeout, exit if Go process may have crashed"""
        while self.running:
            time.sleep(self.ping_check_interval)
            elapsed = time.time() - self.last_ping_time
            if elapsed > self.ping_timeout:
                log.warning(f"Ping timeout: {elapsed:.1f}s > {self.ping_timeout}s, Go process may have crashed, exiting...")
                os._exit(0)

    def start_ping_check(self):
        """Start ping check thread"""
        t = threading.Thread(target=self._ping_check_loop, daemon=True)
        t.start()
        log.info(f"Ping check thread started (timeout={self.ping_timeout}s, interval={self.ping_check_interval}s)")

    def handle_request(self, request: dict) -> dict:
        action = request.get("action", "")

        # Handle cfg_set command (no config check needed)
        if action == "cfg_set":
            log.info(f"request-data: {request}")
            config_data = request.get("data", {})
            success = self.init(config_data)
            if success:
                return {
                    "id": request.get("id", ""),
                    "action": action,
                    "code": 0,
                    "status": "success",
                    "data": {"message": "AI Engine configured"},
                }
            else:
                missing = config.RUNTIME_CONFIG.get_missing_config()
                return {
                    "id": request.get("id", ""),
                    "action": action,
                    "code": 1,
                    "status": f"Configuration incomplete. Missing: {', '.join(missing)}",
                }

        log.info(f"request-data: {request}")
        # Check if configuration is complete for other commands
        if not config.RUNTIME_CONFIG.is_configured():
            missing = config.RUNTIME_CONFIG.get_missing_config()
            error_msg = f"Configuration incomplete. Missing: {', '.join(missing)}. Please call cfg_set first."
            log.error(error_msg)
            return {
                "id": request.get("id", ""),
                "action": action,
                "code": 1,
                "status": error_msg,
            }

        # Check if initialized
        if not self.initialized:
            log.info("AI Engine not initialized, initializing...")
            success = self.init()
            if not success:
                missing = config.RUNTIME_CONFIG.get_missing_config()
                return {
                    "id": request.get("id", ""),
                    "action": action,
                    "code": 1,
                    "status": f"Configuration incomplete. Missing: {', '.join(missing)}",
                }

        engine = self.engines.get(action)

        if engine is None:
            return {
                "id": request.get("id", ""),
                "action": action,
                "code": 1,
                "status": f"Unknown action: {action}",
            }

        try:
            result = engine.process(request.get("data", {}))

            # Extract data and data_dbg from result
            data = result.get("data", result)
            data_dbg = result.get("data_dbg", {})

            respDat = {
                "id": request.get("id", ""),
                "action": action,
                "code": 0,
                "status": "success",
                "data": data,
            }

            # Add data_dbg if present
            if data_dbg:
                respDat["data_dbg"] = data_dbg

            log.info(f"response-data: {respDat}")
            return respDat
        except Exception as e:
            traceback.print_exc()
            return {
                "id": request.get("id", ""),
                "action": action,
                "code": 1,
                "status": str(e),
            }

    def process_request_worker(self):
        """Worker thread to process non-ping requests"""
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
                error_resp = {
                    "id": "",
                    "code": 1,
                    "status": str(e),
                }
                self.send_response(error_resp)


def main():
    engine = AIEngine()

    # Start worker thread for request processing
    worker = threading.Thread(target=engine.process_request_worker, daemon=True)
    worker.start()

    # Start ping check thread (exit if Go process crashes)
    engine.start_ping_check()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Main thread reads stdin and dispatches
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            # Ping requests are handled directly for immediate response
            if request.get("action") == "ping":
                engine.handle_ping(request)
            else:
                # Other requests are queued for processing
                engine.request_queue.put(request)
        except json.JSONDecodeError as e:
            error_resp = {
                "id": "",
                "code": 1,
                "status": f"Invalid JSON: {e}",
            }
            engine.send_response(error_resp)
        except Exception as e:
            traceback.print_exc()
            error_resp = {
                "id": "",
                "code": 1,
                "status": str(e),
            }
            engine.send_response(error_resp)


if __name__ == "__main__":
    main()
