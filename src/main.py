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

from utils.logger import log
from engines.search_engine_faiss import SearchEngine
from engines.index_init_engine import IndexInitEngine
from engines.sync_engine import SyncEngine
from engines.hierarchical_tag_engine import HierarchicalTagEngine
from engines.similarity_engine import SimilarityEngine


class AIEngine:
    def __init__(self):
        self.engines = {}

    def init(self):
        log.info("Initializing AI Engine...")
        
        self.engines["search"] = SearchEngine()
        self.engines["index_init"] = IndexInitEngine()
        self.engines["sync"] = SyncEngine()
        self.engines["tag_detect"] = HierarchicalTagEngine()
        self.engines["find_similar"] = SimilarityEngine()
        self.engines["ping"] = PingEngine()
        
        log.info(f"AI Engine initialized successfully Available actions: {list(self.engines.keys())}")

    def handle_request(self, request: dict) -> dict:
        log.info(f"request-data: {request}")
        action = request.get("action", "")
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


class PingEngine:
    def process(self, params: dict) -> dict:
        return {"message": "pong"}


def main():
    engine = AIEngine()
    engine.init()

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
