import time
from showdown.battle import Battle

TIME_TOLLERANCE = 10 # 10seconds of tollerance

def is_time_over(battle: Battle) -> bool:
    if not hasattr(battle, "start_time"):
        return False

    if battle.time_remaining is None:
        return False

    #print(f"time remaining: {time.time() - battle.start_time - battle.time_remaining}")

    return time.time() - battle.start_time > battle.time_remaining - TIME_TOLLERANCE