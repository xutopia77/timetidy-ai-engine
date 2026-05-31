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

from engines.tag_engine import TagEngine


class AIEngine:
    def __init__(self):
        self.engines = {}

    def init(self):
        self.engines["tag_detect"] = TagEngine()
        self.engines["ping"] = PingEngine()

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
