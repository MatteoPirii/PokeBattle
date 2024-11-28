import time

from showdown.battle import Battler

TIME_TOLLERANCE = 10  # 10 seconds of tolerance


def is_time_over(start: float, time_remaining: float | None) -> bool:
    """Checks if timer of a battle is over"""
    if time_remaining is None:
        return False

    effective_timer = time_remaining - TIME_TOLLERANCE
    elapsed_time = time.time() - start

    print(f"Tempo trascorso: {elapsed_time:.2f}s, Timer effettivo con tolleranza: {effective_timer:.2f}s")

    return elapsed_time > effective_timer


def game_over(challenger: Battler, opponent: Battler) -> bool:
    """Checks if battle is over given two opponents"""
    user_pokemon_alive = challenger.active.is_alive() if challenger.active is not None else False
    opponent_pokemon_alive = opponent.active.is_alive() if opponent.active is not None else False

    # Cheks if there are available Pokémon to switch
    reserve_pokemon_alive = not challenger.get_switches() == []
    reserve_opponent_alive = not opponent.get_switches() == []

    # if no Pokémon is alive battle is over
    return not (user_pokemon_alive or reserve_pokemon_alive) or not (
            opponent_pokemon_alive or reserve_opponent_alive)
