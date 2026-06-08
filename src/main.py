import sys
import json
import signal
import traceback
import os

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

    def handle_request(self, request: dict) -> dict:
        log.info(f"request-data: {request}")
        action = request.get("action", "")
        
        # Handle cfg_set command (no config check needed)
        if action == "cfg_set":
            config_data = request.get("data", {})
            success = self.init(config_data)
            if success:
                return {
                    "id": request.get("id", ""),
                    "code": 0,
                    "status": "success",
                    "data": {"message": "AI Engine configured"},
                }
            else:
                missing = config.RUNTIME_CONFIG.get_missing_config()
                return {
                    "id": request.get("id", ""),
                    "code": 1,
                    "status": f"Configuration incomplete. Missing: {', '.join(missing)}",
                }
        
        # Handle ping command (no config check needed)
        if action == "ping":
            return {
                "id": request.get("id", ""),
                "code": 0,
                "status": "success",
                "data": {"message": "pong"},
            }
        
        # Check if configuration is complete for other commands
        if not config.RUNTIME_CONFIG.is_configured():
            missing = config.RUNTIME_CONFIG.get_missing_config()
            error_msg = f"Configuration incomplete. Missing: {', '.join(missing)}. Please call cfg_set first."
            log.error(error_msg)
            return {
                "id": request.get("id", ""),
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
                    "code": 1,
                    "status": f"Configuration incomplete. Missing: {', '.join(missing)}",
                }
        
        engine = self.engines.get(action)

        if engine is None:
            return {
                "id": request.get("id", ""),
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
                "code": 1,
                "status": str(e),
            }


def main():
    engine = AIEngine()
    # Don't initialize at startup - wait for cfg_set
    
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
