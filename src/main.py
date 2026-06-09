import sys
import json
import signal
import traceback
import os
import time
import threading

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
        self.last_ping_time = time.time()  # Track last ping time
        self.ping_timeout = 30  # 30 seconds timeout
        self.ping_check_interval = 5  # Check every 5 seconds
        self.ping_check_thread = None
        self.running = True  # Flag to control thread

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
        
        log.info("Initializing AI Engine...")
        
        self.engines["search"] = SearchEngine()
        self.engines["index_init"] = IndexInitEngine()
        self.engines["tag_detect"] = HierarchicalTagEngine()
        self.engines["find_similar"] = SimilarityEngine()
        
        self.initialized = True
        log.info(f"AI Engine initialized successfully. Available actions: {list(self.engines.keys())}")
        return True

    def check_ping_timeout(self) -> bool:
        """Check if ping timeout occurred
        
        Returns:
            True if timeout occurred, False otherwise
        """
        elapsed = time.time() - self.last_ping_time
        if elapsed > self.ping_timeout:
            log.warning(f"Ping timeout detected: {elapsed:.1f}s > {self.ping_timeout}s, exiting...")
            return True
        return False

    def update_ping_time(self):
        """Update last ping time"""
        self.last_ping_time = time.time()

    def _ping_check_loop(self):
        """Background thread to check ping timeout"""
        while self.running:
            time.sleep(self.ping_check_interval)
            if self.check_ping_timeout():
                # Timeout occurred, exit process
                os._exit(0)  # Force exit

    def start_ping_check(self):
        """Start ping check thread"""
        if self.ping_check_thread is None:
            self.ping_check_thread = threading.Thread(
                target=self._ping_check_loop,
                daemon=True  # Daemon thread, will exit when main thread exits
            )
            self.ping_check_thread.start()
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
        
        # Handle ping command (no config check needed)
        if action == "ping":
            self.update_ping_time()  # Update last ping time
            return {
                "id": request.get("id", ""),
                "action": action,
                "code": 0,
                "status": "success",
                "data": {"message": "pong"},
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


def main():
    engine = AIEngine()
    # Don't initialize at startup - wait for cfg_set
    
    # Start ping check thread
    engine.start_ping_check()
    
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = engine.handle_request(request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            error_resp = {
                "id": "",
                "code": 1,
                "status": f"Invalid JSON: {e}",
            }
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            traceback.print_exc()
            error_resp = {
                "id": "",
                "code": 1,
                "status": str(e),
            }
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
