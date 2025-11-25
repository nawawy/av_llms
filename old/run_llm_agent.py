import time
from carla_client import CarlaClient
from perception import cluster_point_cloud, extract_object_info
from rule_detector import rule_based_detector
from fsm import DrivingFSM
from llm_runner import AsyncLLM, detect_with_llm
from llm_agent import LLMAgent
from config import LOOP_SLEEP, LLM_MAX_ALLOWABLE_LATENCY


def run_once(duration=30.0):
    client = CarlaClient()
    try:
        vehicle = client.spawn_ego()
        client.attach_lidar()
        fsm = DrivingFSM()
        async_llm = AsyncLLM()
        agent = LLMAgent(async_llm)

        last_safe_control = {'throttle':0.4,'brake':0.0,'steer':0.0}
        t0 = time.time()
        while time.time() - t0 < duration:
            data = client.get_latest_lidar()
            if data is None:
                time.sleep(0.01)
                continue
            pts = data['points']
            clusters = cluster_point_cloud(pts)
            objects = [extract_object_info(c) for c in clusters]
            ego_speed = client.get_ego_speed()
            descriptor = {'ego_speed':ego_speed,'objects':objects}

            # 1) Fast rule check (safety net) - do it every frame
            safety_hazard = rule_based_detector(ego_speed, objects)
            if safety_hazard['critical']:
                # immediate override by FSM safety
                fsm.update(safety_hazard)
                control = fsm.get_controls()
                client.apply_control(control['throttle'],control['brake'],control['steer'])
                last_safe_control = control
                time.sleep(LOOP_SLEEP)
                continue

            # 2) Try LLM detector (async) with short wait
            llm_hazard, elapsed = detect_with_llm(async_llm, descriptor, max_latency=LLM_MAX_ALLOWABLE_LATENCY)

            if llm_hazard is None:
                # LLM timed out or returned nothing - fall back to last safe control
                client.apply_control(last_safe_control['throttle'], last_safe_control['brake'], last_safe_control['steer'])
                time.sleep(LOOP_SLEEP)
                continue

            # 3) If LLM marks critical -> use FSM to execute (there is no separate LLM controller in this mode)
            if llm_hazard.get('critical'):
                fsm.update(llm_hazard)
                control = fsm.get_controls()
                client.apply_control(control['throttle'],control['brake'],control['steer'])
                last_safe_control = control
                time.sleep(LOOP_SLEEP)
                continue

            # 4) Else ask LLM agent for a control decision (prompt-based agent)
            agent_control, agent_elapsed = agent.decide_control(descriptor, max_wait=0.3)
            if agent_control is None:
                client.apply_control(last_safe_control['throttle'], last_safe_control['brake'], last_safe_control['steer'])
            else:
                # normalize control values
                throttle = float(agent_control.get('throttle', last_safe_control['throttle']))
                brake = float(agent_control.get('brake', last_safe_control['brake']))
                steer = float(agent_control.get('steer', last_safe_control['steer']))
                client.apply_control(throttle, brake, steer)
                last_safe_control = {'throttle':throttle,'brake':brake,'steer':steer}

            time.sleep(LOOP_SLEEP)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy()

if __name__ == '__main__':
    run_once(60.0)