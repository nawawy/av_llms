from enum import Enum
from config import CONFIRMATION_FRAMES, SEVERITY_CLEAR_THRESHOLD

class DriveState(Enum):
    CRUISE = 0
    FOLLOW = 1
    BRAKE = 2
    EVADE_LEFT = 3
    EVADE_RIGHT = 4
    STOPPED = 5

class DrivingFSM:
    def __init__(self):
        self.state = DriveState.CRUISE
        self.confirmation_count = 0

    def update(self, hazard_info):
        critical = hazard_info.get('critical', False)
        action = hazard_info.get('recommended_action', 'MAINTAIN')
        severity = hazard_info.get('severity', 0.0)

        if critical:
            self.confirmation_count += 1
        else:
            self.confirmation_count = 0

        if self.state == DriveState.CRUISE:
            if self.confirmation_count >= CONFIRMATION_FRAMES:
                if action == 'BRAKE':
                    self.state = DriveState.BRAKE
                elif action == 'FOLLOW':
                    self.state = DriveState.FOLLOW
                elif action == 'EVADE_LEFT':
                    self.state = DriveState.EVADE_LEFT
                elif action == 'EVADE_RIGHT':
                    self.state = DriveState.EVADE_RIGHT

        elif self.state == DriveState.FOLLOW:
            if critical:
                self.state = DriveState.BRAKE
            elif action == 'MAINTAIN' and severity < SEVERITY_CLEAR_THRESHOLD:
                self.state = DriveState.CRUISE

        elif self.state == DriveState.BRAKE:
            if severity < SEVERITY_CLEAR_THRESHOLD and not critical:
                self.state = DriveState.CRUISE

        elif self.state in (DriveState.EVADE_LEFT, DriveState.EVADE_RIGHT):
            # after evasive maneuver, if severity low, return to cruise
            if severity < SEVERITY_CLEAR_THRESHOLD:
                self.state = DriveState.CRUISE

        elif self.state == DriveState.STOPPED:
            if not critical:
                self.state = DriveState.CRUISE

        return self.state

    def get_controls(self):
        # These control values are placeholders; tune for your vehicle dynamics
        if self.state == DriveState.CRUISE:
            return {'throttle': 0.5, 'brake': 0.0, 'steer': 0.0}
        elif self.state == DriveState.FOLLOW:
            return {'throttle': 0.3, 'brake': 0.0, 'steer': 0.0}
        elif self.state == DriveState.BRAKE:
            return {'throttle': 0.0, 'brake': 0.8, 'steer': 0.0}
        elif self.state == DriveState.EVADE_LEFT:
            return {'throttle': 0.0, 'brake': 0.4, 'steer': -0.6}
        elif self.state == DriveState.EVADE_RIGHT:
            return {'throttle': 0.0, 'brake': 0.4, 'steer': 0.6}
        elif self.state == DriveState.STOPPED:
            return {'throttle': 0.0, 'brake': 1.0, 'steer': 0.0}