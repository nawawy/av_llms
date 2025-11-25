import math
from config import TTC_CRITICAL_THRESHOLD, TTC_CAUTION_THRESHOLD, DIST_CRITICAL, DIST_CAUTION, BEARING_THRESHOLD, MIN_CLOSING_SPEED


def compute_severity(distance_m, closing_speed, ttc):
    # distance score
    dist_term = max(0.0, (DIST_CAUTION - distance_m) / (DIST_CAUTION))
    # ttc term: map smaller ttc to larger score; clamp
    if ttc == float('inf'):
        ttc_term = 0.0
    else:
        ttc_term = max(0.0, 1.0 - (ttc / (TTC_CAUTION_THRESHOLD)))
    # speed term
    speed_term = min(1.0, closing_speed / 10.0)
    return 0.4 * dist_term + 0.4 * ttc_term + 0.2 * speed_term


def rule_based_detector(ego_speed, objects):
    critical = False
    max_severity = 0.0
    recommended_action = 'MAINTAIN'

    for obj in objects:
        d = obj['distance_m']
        rel_speed = obj.get('relative_speed_mps', 0.0)
        bearing = obj.get('bearing_deg', 0.0)

        # closing speed positive if closing
        closing_speed = max(0.0, rel_speed)
        if closing_speed > MIN_CLOSING_SPEED:
            ttc = d / closing_speed
        else:
            ttc = float('inf')

        severity = compute_severity(d, closing_speed, ttc)
        max_severity = max(max_severity, severity)

        if (ttc < TTC_CRITICAL_THRESHOLD) or (d < DIST_CRITICAL and closing_speed > MIN_CLOSING_SPEED):
            critical = True
            if abs(bearing) < BEARING_THRESHOLD:
                recommended_action = 'BRAKE'
            else:
                recommended_action = 'EVADE_LEFT' if bearing > 0 else 'EVADE_RIGHT'
        elif (ttc < TTC_CAUTION_THRESHOLD) or (d < DIST_CAUTION and abs(bearing) < BEARING_THRESHOLD):
            # caution
            if recommended_action != 'BRAKE':
                recommended_action = 'FOLLOW'

    return {
        'critical': critical,
        'severity': float(max_severity),
        'recommended_action': recommended_action
    }