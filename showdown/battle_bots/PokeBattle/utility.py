from showdown.battle import Battler


def game_over(challenger: Battler, opponent: Battler) -> bool:
    """Checks if battle is over given two opponents"""
    user_pokemon_alive = challenger.active.is_alive() if challenger.active is not None else False
    opponent_pokemon_alive = opponent.active.is_alive() if opponent.active is not None else False

    # Checks if there are available Pokémon to switch
    reserve_pokemon_alive = not challenger.get_switches() == []
    reserve_opponent_alive = not opponent.get_switches() == []

    # if no Pokémon is alive battle is over
    return not (user_pokemon_alive or reserve_pokemon_alive) or not (
            opponent_pokemon_alive or reserve_opponent_alive)


def adjust_time(elapsed: int, limit: int) -> int:
    """Returns the new upper limit for the timer"""
    if elapsed >= 120:
        # the agent is categorized as inactive and the timer is the lowest possible
        return 30

    if elapsed >= 60:
        # the lowest timer Pokémon showdown allows is 30s
        if limit <= 30:
            return 30
        # every turn that takes more than a minute sets the limit to a minute less than the original timer
        return limit - 60
    # each turn that takes more than 30s sets the timer to 30s less than the original timer
    if elapsed >= 30:
        return limit - 30

    # in general the timer continues unless is 0 or less
    new_timer = limit - elapsed
    return new_timer if new_timer > 0 else 30
