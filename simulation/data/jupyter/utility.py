import simulation.data.jupyter.constants as constants
import random


def game_over(challenger, opponent) -> bool:
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

def calculate_type_multiplier(move_type: str, defender_types: list[str]) -> float:
    """Calculates damage multiplier considering defender's type(s)"""
    multiplier = 1.0

    effectiveness_values = constants.TYPE_EFFECTIVENESS.get(move_type, {})

    for defender_type in defender_types:
        try:
            current_effectiveness = effectiveness_values.get(defender_type, 1)
            multiplier *= current_effectiveness

        except KeyError:
            print(f"Warning: Effectiveness for {move_type} against {defender_type} not found.")

    return max(multiplier, 0)

def apply_damage_conditions(defender, move, attacker = None) -> float:
    """
    Apply damage reduction modifiers, including Reflect, Light Screen, and specific abilities.
    """
    modifier = 1.0
    side_conditions = getattr(defender, 'side_conditions', {})

    # Check if abilities are ignored by the attacker
    def is_ability_ignored():
        return attacker and attacker.ability and attacker.ability.lower() in ["mold_breaker", "teravolt", "turboblaze",
                                                                              "mycelium_might"]

    # Constants for easier handling of conditions
    ABILITY_IMMUNITIES = {
        "ground": ["levitate", "earth_eater"],
        "water": ["storm_drain", "water_absorb", "desolate_land"],
        "electric": ["lightning_rod", "volt_absorb"],
        "fire": ["flash_fire"],
        "ice": ["dry_skin"],
        "grass": ["sap_sipper"],
    }

    # Ability-based immunities (using a dictionary lookup)
    if not is_ability_ignored() and defender.ability:
        for move_type, abilities in ABILITY_IMMUNITIES.items():
            if move.type == move_type and defender.ability.lower() in abilities:
                return 0.0  # No damage

    # Critical hits ignore reductions
    if getattr(move, 'critical_hit', False):
        return 1.0  # Ignore all reductions

    # Field-based defensive conditions
    if side_conditions.get(constants.REFLECT) and move.category == constants.PHYSICAL:
        modifier *= 0.5
    if side_conditions.get(constants.LIGHT_SCREEN) and move.category == constants.SPECIAL:
        modifier *= 0.5
    if side_conditions.get(constants.AURORA_VEIL):
        modifier *= 0.67

    # Abilities with damage reduction and increment
    ability_conditions = {
        "multiscale": (defender.hp == defender.max_hp, 0.5),
        "filter": (calculate_type_multiplier(move.type, defender.types) > 1, 0.75),
        "solid_rock": (calculate_type_multiplier(move.type, defender.types) > 1, 0.75),
        "prism_armor": (calculate_type_multiplier(move.type, defender.types) > 1, 0.75),
        "ice_scales": (move.category == constants.SPECIAL, 0.5),
        "fluffy": (move.category == constants.PHYSICAL and move.type != "fire", 0.5),
        "thick_fat": (move.type in ["fire", "ice"], 0.5),
        "fur_coat": (move.category == constants.PHYSICAL, 0.5),
        "ice_face": (move.category == constants.PHYSICAL and move.type == "ice", 0.5),
        "heatproof": (move.type == "fire", 0.5),
        "water_bubble": (move.type == "fire", 0.5),
        "thermal_exchange": (move.type == "fire", 0.5),
        "water_veil": (move.type == "fire", 0.5),
        "water_compaction": (move.type == "water", 0.5),
        "dry_skin": (move.type == "fire", 1.5),
        "fluffy": (move.type == "fire", 1.5),
        "friend_guard": (getattr(defender, 'partner_ability', '').lower() == "friend_guard", 0.75)
    }

    if defender.ability:
        for ab, (condition, modifier_value) in ability_conditions.items():
            if defender.ability.lower() == ab and condition:
                modifier *= modifier_value

        # Special abilities that variably affect damage
        if defender.ability.lower() == "wonder_guard":
            if calculate_type_multiplier(move.type, defender.types) <= 1:
                return 0.0  # No damage from non-super-effective moves

    return modifier

def calculate_damage(attacker, defender, move) -> int:
    """Calculate damage inflicted by move with additional modifier logic."""

    # Base damage calculation
    level = attacker.level
    attack_stat = get_attack_stat(attacker, move)
    defense_stat = get_defense_stat(defender, move)
    # Ensure defense_stat is not zero to avoid division errors
    defense_stat = max(defense_stat, 1)

    damage = (((2 * level / 5 + 2) * move.basePower * (attack_stat / defense_stat)) / 50 + 2)

    # STAB (Same-Type Attack Bonus)
    stab = 1.5 if move.type in attacker.types else 1.0
    damage *= stab

    # Add type effectiveness
    type_multiplier = calculate_type_multiplier(move.type, defender.types)
    damage *= type_multiplier

    # Apply external conditions
    damage *= apply_damage_conditions(defender, move)

    # Apply random variance (from 0.85 to 1.0)
    damage *= random.uniform(0.85, 1.0)

    return max(1, int(damage))

def get_attack_stat(pokemon, move) -> float:
    return pokemon.stats[constants.ATTACK] if move.category == constants.PHYSICAL else pokemon.stats[
        constants.SPECIAL_ATTACK]

def avg(el) -> float:
    return (el[0] + el[1]) / 2

def scale_range(value: float, source_range: list[float | int], destination_range = [-100, 100]) -> float:
    "Scales the value into a different range"
    assert len(source_range) == 2, f"source range should by an array of two elements, it is {source_range}"

    value_diff = value - source_range[0]
    destination_diff = destination_range[1] - destination_range[0]
    source_diff = source_range[1] - source_range[0]
    if source_diff == 0:
        return source_range[0]
    scale = value_diff * destination_diff / source_diff
    return destination_range[0] + scale

def get_defense_stat(pokemon, move) -> float:
    return pokemon.stats[constants.DEFENSE] if move.category == constants.PHYSICAL else pokemon.stats[
        constants.SPECIAL_DEFENSE]