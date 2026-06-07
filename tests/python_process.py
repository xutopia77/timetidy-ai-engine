"""
Python Process Simulator for testing AI Engine stdio communication

This script simulates the Go PythonProcess module:
- Starts Python AI subprocess
- Sends JSON requests via stdin
- Receives JSON responses via stdout
- Receives log messages via stderr
- Manages request-response matching
"""

import sys
import json
import subprocess
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Any
import queue


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add src directory to Python path for utils module
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.logger import log


class PythonProcess:
    def __init__(self, python_path: str = "python", script_path: str = None):
        self.python_path = python_path
        self.script_path = script_path
        self.process: Optional[subprocess.Popen] = None
        self.pending: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()
        self.running = False
        self.response_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
    
    def start(self) -> bool:
        if self.script_path is None:
            project_root = Path(__file__).parent.parent
            self.script_path = str(project_root / "src" / "main.py")
        
        if not Path(self.script_path).exists():
            log.error(f"AI script not found: {self.script_path}")
            return False
        
        try:
            self.process = subprocess.Popen(
                [self.python_path, self.script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1
            )
            
            self.running = True
            
            self.response_thread = threading.Thread(target=self._read_responses, daemon=True)
            self.response_thread.start()
            
            self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self.stderr_thread.start()
            
            time.sleep(1)
            
            if self.process.poll() is not None:
                log.error(f"Python process exited immediately with code {self.process.returncode}")
                self.running = False
                return False
            
            log.info(f"Python process started successfully (PID: {self.process.pid})")
            return True
            
        except Exception as e:
            log.error(f"Error starting Python process: {e}")
            self.running = False
            return False
    
    def _read_responses(self):
        try:
            for line in self.process.stdout:
                if not line:
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    response = json.loads(line)
                    request_id = response.get("id", "")
                    
                    self.lock.acquire()
                    try:
                        if request_id in self.pending:
                            self.pending[request_id].put(response)
                    finally:
                        self.lock.release()
                    
                except json.JSONDecodeError as e:
                    log.error(f"Error decoding response: {e}")
                    log.error(f"Raw line: {line}")
                    
        except Exception as e:
            log.error(f"Error reading responses: {e}")
        finally:
            self.running = False
    
    def _read_stderr(self):
        try:
            for line in self.process.stderr:
                if line:
                    log.info(f"[ai-log] {line.strip()}")
        except Exception as e:
            log.error(f"Error reading stderr: {e}")
    
    def send_request(self, action: str, params: Dict[str, Any], timeout: float = 120.0) -> Optional[Dict]:
        if not self.running or self.process.poll() is not None:
            log.error("Python process is not running")
            return None
        
        request_id = str(int(time.time() * 1000000))
        
        request = {
            "id": request_id,
            "action": action,
            "data": params
        }
        
        response_queue = queue.Queue()
        
        self.lock.acquire()
        try:
            self.pending[request_id] = response_queue
        finally:
            self.lock.release()
        
        try:
            request_json = json.dumps(request)
            self.process.stdin.write(request_json + "\n")
            self.process.stdin.flush()
        except Exception as e:
            self.lock.acquire()
            try:
                if request_id in self.pending:
                    del self.pending[request_id]
            finally:
                self.lock.release()
            log.error(f"Error sending request: {e}")
            return None
        
        try:
            response = response_queue.get(timeout=timeout)
            
            self.lock.acquire()
            try:
                if request_id in self.pending:
                    del self.pending[request_id]
            finally:
                self.lock.release()
            
            return response
            
        except queue.Empty:
            self.lock.acquire()
            try:
                if request_id in self.pending:
                    del self.pending[request_id]
            finally:
                self.lock.release()
            log.error(f"Request timeout after {timeout}s")
            return None
    
    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                log.error(f"Error stopping process: {e}")
        
        self.running = False
    
    def is_running(self) -> bool:
        return self.running and self.process and self.process.poll() is None


def test_communication():
    log.info("\n" + "="*80)
    log.info("AI Engine stdio Communication Test")
    log.info("="*80)
    
    process = PythonProcess()
    
    if not process.start():
        log.error("Failed to start Python process")
        return False
    
    log.info("\nTesting ping...")
    response = process.send_request("ping", {})
    
    if response and response.get("code") == 0:
        log.info("✓ Ping successful")
        log.info(f"  Response: {response}")
    else:
        log.error("✗ Ping failed")
        log.error(f"  Response: {response}")
        process.stop()
        return False
    
    log.info("\nCommunication test completed successfully!")
    
    process.stop()
    
    return True


if __name__ == "__main__":
    test_communication()