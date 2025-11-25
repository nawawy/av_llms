import time
import json
import threading
from queue import Queue, Empty
from config import USE_LOCAL_LLM, LOCAL_LLM_MODEL, LOCAL_LLM_MAX_TOKENS, LOCAL_LLM_TEMPERATURE, USE_OPENAI_FALLBACK, OPENAI_API_KEY, OPENAI_MODEL_NAME, LLM_MAX_ALLOWABLE_LATENCY
from openai import OpenAI


# Try to import llama_cpp
# try:
#     from llama_cpp import Llama
# except Exception:
#     Llama = None

try:
    import openai
    client = OpenAI(api_key="")

except Exception:
    openai = None

class AsyncLLM:
    """Runs LLM inference in a background thread and provides latest result with timestamps."""
    def __init__(self):
        self._task_queue = Queue()
        self._result_queue = Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop = threading.Event()
        self._thread.start()
        self._model = None
        if USE_LOCAL_LLM and Llama is not None:
            try:
                self._model = Llama(model_path=LOCAL_LLM_MODEL)
                print('Local LLM loaded in worker')
            except Exception as e:
                print('Failed to init local LLM:', e)
                self._model = None

    def _worker(self):
        while not self._stop.is_set():
            try:
                task = self._task_queue.get(timeout=0.1)
            except Empty:
                continue
            prompt, task_id = task
            start = time.time()
            response_text = None
            if self._model is not None:
                try:
                    out = self._model.create(prompt=prompt, max_tokens=LOCAL_LLM_MAX_TOKENS, temperature=LOCAL_LLM_TEMPERATURE)
                    response_text = out.get('choices', [{}])[0].get('text', '')
                except Exception as e:
                    print('Local LLM inference error:', e)
                    response_text = None
            if response_text is None and USE_OPENAI_FALLBACK and openai is not None and OPENAI_API_KEY:
                try:
                    openai.api_key = OPENAI_API_KEY
                    resp = openai.ChatCompletion.create(model=OPENAI_MODEL_NAME, messages=[{'role':'user','content':prompt}], max_tokens=200, temperature=0.0)
                    response_text = resp['choices'][0]['message']['content']
                except Exception as e:
                    print('OpenAI fallback failed:', e)
                    response_text = None
            elapsed = time.time() - start
            self._result_queue.put({'task_id': task_id, 'text': response_text, 'elapsed': elapsed, 'timestamp': time.time()})

    def submit(self, prompt, task_id=None):
        if task_id is None:
            task_id = int(time.time()*1000)
        self._task_queue.put((prompt, task_id))
        return task_id

    def poll_result(self, timeout=0.0):
        # non-blocking poll latest
        try:
            return self._result_queue.get(block=False)
        except Exception:
            return None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

# helper prompt builder for detector

def build_detector_prompt(descriptor):
    # Keep it short: JSON input + explicit JSON output request
    inp = json.dumps(descriptor, separators=(',',':'))
    prompt = f"You are a driving hazard detector. Input JSON: {inp}\nRespond only with JSON: {\"critical\":bool,\"severity\":0.0,\"recommended_action\":\"BRAKE|FOLLOW|MAINTAIN|EVADE_LEFT|EVADE_RIGHT\"}"
    return prompt

# parse helper

def parse_json_from_text(text):
    if not text:
        return None
    s = text.strip()
    # try direct parse
    try:
        return json.loads(s)
    except Exception:
        # try to find a JSON substring
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end>start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
    return None

# Synchronous wrapper that waits up to max_latency, else returns None

def detect_with_llm(async_llm, descriptor, max_latency=LLM_MAX_ALLOWABLE_LATENCY):
    prompt = build_detector_prompt(descriptor)
    tid = async_llm.submit(prompt)
    t0 = time.time()
    # wait loop until result or timeout
    while time.time() - t0 < max_latency:
        res = async_llm.poll_result()
        if res is not None and res.get('task_id') == tid:
            parsed = parse_json_from_text(res.get('text'))
            return parsed, res.get('elapsed')
    # timed out
    return None, None