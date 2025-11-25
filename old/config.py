from pathlib import Path


# CARLA
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
LIDAR_CHANNELS = 32
LIDAR_RANGE = 50.0
LIDAR_ROTATION_FREQ = 10
LIDAR_POINTS_PER_SEC = 56000


# Detector thresholds (rule)
TTC_CRITICAL_THRESHOLD = 1.5
TTC_CAUTION_THRESHOLD = 4.0
DIST_CRITICAL = 6.0
DIST_CAUTION = 15.0
BEARING_THRESHOLD = 20.0
MIN_CLOSING_SPEED = 0.5


# FSM / confirmation
CONFIRMATION_FRAMES = 2
SEVERITY_CLEAR_THRESHOLD = 0.25


# LLM settings
USE_LOCAL_LLM = True
LOCAL_LLM_MODEL = str(Path('models/mistral-7b-q4.bin').absolute())
LOCAL_LLM_MAX_TOKENS = 64
LOCAL_LLM_TEMPERATURE = 0.0
USE_OPENAI_FALLBACK = False
OPENAI_API_KEY = ''
OPENAI_MODEL_NAME = 'gpt-3.5-turbo'


# Runtime
LOOP_SLEEP = 0.05 # seconds -> ~20 Hz
LLM_MAX_ALLOWABLE_LATENCY = 0.25 # seconds: if exceeded, fallback to rule/FSM


# Logging
LOG_PATH = Path('logs')
LOG_PATH.mkdir(exist_ok=True)