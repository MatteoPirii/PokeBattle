import constants
import time

from showdown.battle import Battler, Pokemon, Move

TIME_TOLLERANCE = 10 # 10seconds of tollerance

def is_time_over(start: float, time_remaining: float | None) -> bool:
    "Checks if timer of a battle is over"
    if time_remaining is None:
        return False

    return time.time() - start > time_remaining - TIME_TOLLERANCE

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

def options_categorization(options: list[str]) -> tuple[list[str], list[str]]:
    """
    Separates moves and switches given a list of all possible options
    Returns (moves, switches)
    """
    moves: list[str] = []
    switches: list[str] = []
    for option in options:
        if option.startswith(constants.SWITCH_STRING + " "):
            switches.append(option)
        else:
            moves.append(option)

    return moves, switches

def get_attack_stat(pokemon: Pokemon, move: Move) -> float:
    if not isinstance(move, Move):
        raise TypeError(f"Expected move to be an instance of Move, got {type(move).__name__}")

    return pokemon.stats[constants.ATTACK] if move.category == constants.PHYSICAL else pokemon.stats[
        constants.SPECIAL_ATTACK]


def get_defense_stat(pokemon: Pokemon, move: Move) -> float:
    return pokemon.stats[constants.DEFENSE] if move.category == constants.PHYSICAL else pokemon.stats[
        constants.SPECIAL_DEFENSE]
