import time
from datetime import datetime


# How many consecutive failed checks are required
# before a real violation event starts.
VIOLATION_START_THRESHOLD = 3

# How many consecutive good checks are required
# before an active violation ends.
VIOLATION_END_THRESHOLD = 2


violation_events = {}


def _get_key(person_id, violation_type):
    return (person_id, violation_type)


def update_violation(person_id, violation_type, is_violation):
    """
    Updates the violation state for one person and one violation type.

    Returns:
        event_started  -> True only when a NEW violation starts
        event          -> current event information
    """

    key = _get_key(person_id, violation_type)

    if key not in violation_events:
        violation_events[key] = {
            "active": False,
            "count": 0,
            "bad_checks": 0,
            "good_checks": 0,
            "first_detected": None,
            "last_detected": None,
            "person_id": person_id,
            "violation": violation_type,
            "photo_path": None
        }

    event = violation_events[key]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    event_started = False

    # -----------------------------------------
    # VIOLATION DETECTED
    # -----------------------------------------
    if is_violation:

        event["bad_checks"] += 1
        event["good_checks"] = 0

        # Start a NEW violation event
        if not event["active"]:

            if event["bad_checks"] >= VIOLATION_START_THRESHOLD:

                event["active"] = True
                event["count"] = 1
                event["first_detected"] = now
                event["last_detected"] = now
                event["photo_path"] = None

                event_started = True

        else:
            # Existing violation continues
            event["count"] += 1
            event["last_detected"] = now

    # -----------------------------------------
    # NO VIOLATION
    # -----------------------------------------
    else:

        event["good_checks"] += 1
        event["bad_checks"] = 0

        # End the active violation
        if event["active"]:

            if event["good_checks"] >= VIOLATION_END_THRESHOLD:

                event["active"] = False
                event["good_checks"] = 0
                event["bad_checks"] = 0

    return event_started, event


def get_active_events():
    """
    Returns all currently active violation events.
    """

    return [
        event
        for event in violation_events.values()
        if event["active"]
    ]


def get_all_events():
    """
    Returns all violation events stored in memory.
    """

    return list(violation_events.values())


def reset_events():
    """
    Clears all events.
    Useful during testing.
    """

    violation_events.clear()