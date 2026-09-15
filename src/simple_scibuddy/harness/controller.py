"""Public-only RPC shim copied into the controller container."""

import json
import runpy
import sys
import traceback


class API:
    def call(self, op, **payload):
        print(json.dumps({"op": op, **payload}), flush=True)
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("Broker disconnected")
        return json.loads(line)

    def generate(self, messages):
        return self.call("generate", messages=messages)

    def execute(self, code):
        return self.call("execute", code=code)

    def submit(self, text):
        return self.call("submit", text=text)


if __name__ == "__main__":
    try:
        task = json.loads(sys.stdin.readline())
        runpy.run_path("/tmp/harness.py")["run"](task, API())
        print(json.dumps({"op": "end"}), flush=True)
    except BaseException:
        print(json.dumps({"op": "error", "error": traceback.format_exc()}), flush=True)
