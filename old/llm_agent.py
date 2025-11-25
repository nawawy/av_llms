import json
import time
from llm_runner import AsyncLLM, build_detector_prompt, parse_json_from_text

class LLMAgent:
    def __init__(self, async_llm):
        self.async_llm = async_llm

    def build_controller_prompt(self, history_snapshot):
        # short structured prompt for control; history_snapshot is a dict
        inp = json.dumps(history_snapshot, separators=(',',':'))
        prompt = f"You are a driving controller. Given JSON state: {inp} Respond with JSON: {\"throttle\":float,\"brake\":float,\"steer\":float}" 
        return prompt

    def decide_control(self, history_snapshot, max_wait=0.3):
        prompt = self.build_controller_prompt(history_snapshot)
        tid = self.async_llm.submit(prompt)
        t0 = time.time()
        while time.time() - t0 < max_wait:
            res = self.async_llm.poll_result()
            if res is not None and res.get('task_id') == tid:
                parsed = parse_json_from_text(res.get('text'))
                return parsed, res.get('elapsed')
        return None, None