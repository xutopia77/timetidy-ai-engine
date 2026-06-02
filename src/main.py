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

from engines.search_engine_faiss import SearchEngine
from engines.index_init_engine import IndexInitEngine
from engines.sync_engine import SyncEngine
from engines.hierarchical_tag_engine import HierarchicalTagEngine
from engines.similarity_engine import SimilarityEngine


class AIEngine:
    def __init__(self):
        self.engines = {}

    def init(self):
        print("Initializing AI Engine...", file=sys.stderr)
        
        self.engines["search"] = SearchEngine()
        self.engines["index_init"] = IndexInitEngine()
        self.engines["sync"] = SyncEngine()
        self.engines["tag_detect"] = HierarchicalTagEngine()
        self.engines["find_similar"] = SimilarityEngine()
        self.engines["ping"] = PingEngine()
        
        print("AI Engine initialized successfully", file=sys.stderr)
        print(f"  - Available actions: {list(self.engines.keys())}", file=sys.stderr)

    def handle_request(self, request: dict) -> dict:
        action = request.get("action", "")
        engine = self.engines.get(action)

        if engine is None:
            return {
                "id": request.get("id", ""),
                "success": False,
                "error": f"Unknown action: {action}",
            }

        try:
            result = engine.process(request.get("params", {}))
            return {
                "id": request.get("id", ""),
                "success": True,
                "data": result,
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "id": request.get("id", ""),
                "success": False,
                "error": str(e),
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
                "success": False,
                "error": f"Invalid JSON: {e}",
            }
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            traceback.print_exc()
            error_resp = {
                "id": "",
                "success": False,
                "error": str(e),
            }
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
