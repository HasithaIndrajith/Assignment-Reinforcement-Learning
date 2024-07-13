import random

def validate_coordinates(s, x, y):
    if (x, y) == (2, 2) or x < 1 or x > 4 or y < 1 or y > 3:
        return s
    else:
        return (x, y)

def NextState(s, a):
    x, y = s
    action_state_transitions = {
        "Move Up": (
            validate_coordinates(s, x, y + 1),
            validate_coordinates(s, x + 1, y),
            validate_coordinates(s, x - 1, y)
        ),
        "Move Right": (
            validate_coordinates(s, x + 1, y),
            validate_coordinates(s, x, y + 1),
            validate_coordinates(s, x, y - 1)
        ),
        "Move Down": (
            validate_coordinates(s, x, y - 1),
            validate_coordinates(s, x + 1, y),
            validate_coordinates(s, x - 1, y)
        ),
        "Move Left": (
            validate_coordinates(s, x - 1, y),
            validate_coordinates(s, x, y + 1),
            validate_coordinates(s, x, y - 1)
        )
    }
    return random.choices(action_state_transitions[a], weights=[0.8, 0.1, 0.1], k=1)[0]

def next_state_to_label(s, next_s):
    x1, y1 = s
    x2, y2 = next_s

    if x2 == x1 + 1:
        return 'right'
    elif x2 == x1 - 1:
        return 'left'
    elif y2 == y1 + 1:
        return 'upper'
    elif y2 == y1 - 1:
        return 'lower'
    elif s == next_s:
        return 'itself'
    else:
        raise ValueError("Invalid next state: {} to {}".format(s, next_s))

test_cases = [
    ((1, 1), "Move Right"),
    ((1, 1), "Move Up"),
    ((3, 2), "Move Down"),
    ((3, 2), "Move Left"),
    ((3, 3), "Move Left")
]

for s, a in test_cases:
    frequency = {'itself': 0, 'upper': 0, 'lower': 0, 'right': 0, 'left': 0}
    for _ in range(100):
        next_s = NextState(s, a)
        frequency[next_state_to_label(s, next_s)] += 1
    print((s, a), frequency)
