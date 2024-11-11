import json
import math
import random
import time
from copy import deepcopy
# from typing import override

import constants
from data import pokedex
from showdown.battle import Battle, Pokemon, Move
from showdown.battle_bots.helpers import format_decision
from showdown.engine import helpers

# JSON file containing all moves
with open('data/moves.json', 'r') as f:
    all_move_json = json.load(f)


class BattleBot(Battle):

    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)
        self.consecutive_switch_penalty = 10
        self.debug = False

    def find_best_move(self):
        """Finds best move or best switch using Minmax"""
        best_move = None
        max_value = float('-inf')

        # Check if the Pokémon is alive or inactive
        if not self.user.active.is_alive():
            print("Error: active Pokémon is invalid or exausted.") if self.debug else None
            # Returns first available switch
            switches = [f"{constants.SWITCH_STRING} {name}" for name in self.user.get_switches()]
            if switches:
                selected_switch = format_decision(self, switches[0])
                print(f"Selected switch: {selected_switch}") if self.debug else None
                return selected_switch
            return ["no valid move or switch"]

        # Get all available moves and switches
        user_options, _ = self.get_all_options()
        print(f"Available moves: {user_options}") if self.debug else None
        moves, switches = BattleBot.options_categorization(user_options)

        start_time = time.time()  # timer start
        time_limit = 2 * 60 + 30  # 2 minutes 30 seconds time limit

        # If we're forced to switch or there are no available moves the first switch is returned
        if self.force_switch or not moves:
            if not switches:
                print("Error: no available switch.") if self.debug else None
                return ["no valid move or switch"]

            switch = self.find_best_switch()
            if switch is None:
                switch = self.get_pkmn_by_switch(switches[0])
            selected_switch = format_decision(self, f"{constants.SWITCH_STRING} {switch.name}")
            print(f"Selected switch: {selected_switch}") if self.debug else None
            return selected_switch

        # Prioritize type advantage moves
        opponent_types = self.opponent.active.types
        prioritized_moves = [move for move in moves if is_type_advantage_move(move, opponent_types)]
        if prioritized_moves:
            moves = prioritized_moves

        # Execute MinMax for each option
        for move in moves:
            if time.time() - start_time > time_limit:
                print("Time expired, returning best found move") if self.debug else None
                break

            saved_state = deepcopy(self)  # Saving the battle state
            self.apply_move(move)  # Choice simulation
            move_value = self.minimax(is_maximizing=True, alpha=float('-inf'), beta=float('inf'))

            self.restore_state(saved_state)  # Battle state recovery

            if move_value > max_value:
                max_value = move_value
                best_move = move

        # Select highest damage move
        if best_move:
            selected_move = format_decision(self, best_move)
            print(f"Best found move: {selected_move}") if self.debug else None
            return selected_move  # returns formatted decision

            # In case there are no valid moves Pokémon get switched (dead code)
            switch = self.find_best_switch()
            assert switch is not None
            selected_switch = format_decision(self, f"{constants.SWITCH_STRING} {switch.name}")
            print(f"Selected switch: {selected_switch}") if self.debug else None
            return selected_switch

    @staticmethod
    def options_categorization(options: list[str]) -> tuple[list[str], list[str]]:
        # Separate moves and switches
        moves: list[str] = []
        switches: list[str] = []
        for option in options:
            if option.startswith(constants.SWITCH_STRING + " "):
                switches.append(option)
            else:
                moves.append(option)

        return moves, switches

    def apply_move(self, move_name: str) -> None:
        """Apply simulated move or switch considering type advantage"""
        move_part = move_name.split()

        if constants.SWITCH_STRING in move_name and len(move_part) > 1:
            self.user.active = Pokemon.from_switch_string(move_part[1])
        else:
            # Damage, status effect and stats changes simulation
            move = Move(move_name)

            if move:
                # Accuracy based move success rate calculation
                if random.randint(1, 100) <= move.accuracy:
                    assert self.user.active is not None
                    assert self.opponent.active is not None
                    print(f"{self.user.active.name} use {move.name}") if self.debug else None

                    # Damage calculation considering types
                    type_multiplier = calculate_type_multiplier(move.type, self.opponent.active.types)

                    damage = calculate_damage(self.user.active, self.opponent.active, move)
                    damage *= type_multiplier
                    damage *= (self.user.active.level / self.opponent.active.level)

                    self.opponent.active.hp -= math.floor(damage)
                    print(f"{move.name} inflicted {damage:.2f} hp of damage to {self.opponent.active.name} with an "
                          f"efficacy multiplier of {type_multiplier}.") if self.debug else None

                    if move.status is not None:
                        # The move has no secondary effects
                        self.opponent.active.status = move.status
                        print(f"{self.opponent.active.name} has been {move.status}!") if self.debug else None
                else:
                    assert self.user.active is not None
                    print(f"{self.user.active.name} missed the move {move.name}.") if self.debug else None
            else:
                print(f"Error: move {move_name} not found") if self.debug else None

    def restore_state(self, saved_state):
        """Restores battle state after single move simulation"""
        self.__dict__.update(saved_state.__dict__)

    def minimax(self, is_maximizing: bool, alpha: float, beta: float, max_depth: int = 7) -> float:
        """Minimax algorithm with Alpha-Beta cutting-out."""
        # End conditions: max-depth reached or match ended
        if max_depth == 0 or self.game_over():
            return self.evaluate_state()

        assert self.user.active is not None

        if not self.user.active.is_alive():
            print("Error: No pokemon alive") if self.debug else None
            # Switching Pokémon
            switch = self.find_best_switch()
            if switch:
                self.apply_move(f"Switch {switch}") if self.debug else None
            return float('-inf')  # In case of no valid option the minimum evaluation is returned

        if is_maximizing:
            return self.max_eval(alpha, beta, max_depth)
        else:
            return self.min_eval(alpha, beta, max_depth)

    def max_eval(self, alpha: float, beta: float, max_depth: int) -> float:
        ineffective_moves = True  # Move ineffectiveness flag

        max_eval = float('-inf')
        user_options, _ = self.get_all_options()

        for move in user_options:
            saved_state = deepcopy(self)  # Savestate before moving
            self.apply_move(move)

            # Move effectiveness check
            move_type = all_move_json.get(move.lower(), {}).get('type', None)
            if move_type:
                assert self.opponent.active is not None, "opponent pokemon shouldn't be None"
                type_multiplier = calculate_type_multiplier(move_type, self.opponent.active.types)
                print(type_multiplier) if self.debug else None
                if type_multiplier > 1:
                    ineffective_moves = False  # Effective move found

            eval = self.minimax(False, alpha, beta, max_depth - 1)  # Opponent turn
            self.restore_state(saved_state)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Alpha-Beta pruning

        if ineffective_moves and max_depth == 7:
            print("Every move is ineffective, looking for a switch") if self.debug else None
            switch = self.find_best_switch()
            if switch is None:
                return max_eval
            if switch:
                print(f"Switching with {switch}.") if self.debug else None
                self.apply_move(f"switch {switch}")

        return max_eval

    def min_eval(self, alpha: float, beta: float, max_depth: int) -> float | int:
        min_eval = float('inf')
        _, opponent_options = self.get_all_options()

        for move in opponent_options:
            saved_state = deepcopy(self)  # Save battle state before moving
            self.apply_move(move)
            eval = self.minimax(True, alpha, beta, max_depth - 1)  # Bot turn, maximizing
            self.restore_state(saved_state)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha-Beta pruning
        return min_eval

    def evaluate_state(self):
        """Battle state evaluation"""
        score = 0

        # Check if active Pokémon are alive
        assert self.user.active is not None
        assert self.opponent.active is not None

        if not (self.user.active.is_alive() or self.opponent.active.is_alive()):
            print(f"Error: Pokemon not alive. score {score}.") if self.debug else None
            return score

        if hasattr(self.user, 'consecutive_switches') and self.user.consecutive_switches > 1:
            score -= self.user.consecutive_switches * self.consecutive_switch_penalty

        # 1. Scores by hp difference and level
        score += (self.user.active.hp - self.opponent.active.hp) + (
                self.user.active.level - self.opponent.active.level) * 10

        # 2. Bonus/penalty for alive reserve
        score += sum(10 for p in self.user.reserve if p.hp > 0)
        score -= sum(10 for p in self.opponent.reserve if p.hp > 0)

        # 3. Bonus/penalty for type advantage/disadvantage
        if is_type_disadvantageous(self.user.active, self.opponent.active):
            score -= -40
        else:
            score += 50

        # 4 Bonus for weather conditions
        if self.weather == 'sunnyday' and 'Fire' in self.user.active.types:
            score += 20
        elif self.weather == 'raindance' and 'Water' in self.user.active.types:
            score += 20

        # 5. Penalty for status conditions
        if self.user.active.status == constants.PARALYZED:
            score -= 20
        elif self.user.active.status == constants.POISON:
            score -= 15
        elif self.user.active.status == 'badly poisoned':
            score -= 25
        elif self.user.active.status == constants.BURN:
            score -= 20
        elif self.user.active.status == constants.SLEEP:
            score -= 30
        elif self.user.active.status == constants.FROZEN:
            score -= 40

            # Integrate worst-case opponent move analysis
            opponent_moves = self.opponent.active.moves
            if opponent_moves:
                worst_opponent_score = min(
                    self.evaluate_move_risk(move, self.user.active) for move in opponent_moves
                )
                score += worst_opponent_score  # Adjust score with worst-case scenario

        return score

    @staticmethod
    def evaluate_move_risk(move, user_pokemon):
        """Evaluate the risk posed by an opponent's move based on potential damage."""
        type_multiplier = calculate_type_multiplier(move.type, user_pokemon.types)
        damage_potential = move.basePower * type_multiplier
        accuracy_factor = move.accuracy / 100 if isinstance(move.accuracy, int) else 1  # Account for move accuracy
        risk_score = -(damage_potential * accuracy_factor)
        return risk_score  # Negative for riskier moves

    def find_best_switch(self) -> Pokemon | None:
        """Finds best switch in reserve (defence-wise)"""
        best_switch = None
        max_resistance = float('-inf')

        for switch in self.user.get_switches():
            pokemon_to_switch = self.get_pkmn_by_switch(switch)

            # Confirm the Pokémon belongs to the user's team
            if pokemon_to_switch not in self.user.reserve:
                print(f"Error: {pokemon_to_switch.name} is not in the user's reserve.") if self.debug else None
                continue

            resistance = 1
            for opponent_pokemon_type in self.opponent.active.types:
                for pokemon_to_switch_type in pokemon_to_switch.types:
                    resistance *= constants.TYPE_EFFECTIVENESS[pokemon_to_switch_type][opponent_pokemon_type]

            if resistance > max_resistance:
                max_resistance = resistance
                best_switch = pokemon_to_switch

        print(f"best switch: {best_switch}") if self.debug else None

        return best_switch

    def get_pokemon_by_name(self, name: str) -> Pokemon | None:
        """Returns the Pokémon with the name took from user reserve."""

        # Remove switch prefix if present
        if name.startswith(f"{constants.SWITCH_STRING} "):
            name = name.split(" ", 1)[1]

        normalized_name: str = helpers.normalize_name(name)
        for pokemon in self.user.reserve:
            if pokemon.name.lower() == normalized_name:
                return pokemon
        return None

    def get_pkmn_by_switch(self, switch: str) -> Pokemon:
        """Returns Pokémon from a switching string"""
        name = switch.split(' ')[1]
        pkmn = self.get_pokemon_by_name(name)
        assert pkmn is not None
        return pkmn

    def game_over(self) -> bool:
        """Checks if battle is over"""
        user_pokemon_alive = self.user.active.is_alive() if self.user.active is not None else False
        opponent_pokemon_alive = self.opponent.active.is_alive() if self.opponent.active is not None else False

        # Cheks if there are available Pokémon to switch
        reserve_pokemon_alive = not self.user.get_switches() == []
        reserve_opponent_alive = not self.opponent.get_switches() == []

        # if no Pokémon is alive battle is over
        return not (user_pokemon_alive or reserve_pokemon_alive) or not (
                opponent_pokemon_alive or reserve_opponent_alive)


def is_type_disadvantageous(user: Pokemon, opponent: Pokemon) -> bool:
    """Checks if Pokémon type is disvantageus or not"""
    user_pokemon_types = user.types
    opponent_pokemon_types = opponent.types

    # Advantage and disadvantage evaluations
    advantage_count = 0
    disadvantage_count = 0
    total_multiplier = 1

    for user_type in user_pokemon_types:
        for opponent_type in opponent_pokemon_types:

            multiplier = constants.TYPE_EFFECTIVENESS[opponent_type].get(user_type, 1)

            if multiplier > 1:
                advantage_count += 1
            elif multiplier < 1:
                disadvantage_count += 1

            total_multiplier *= multiplier

    return disadvantage_count > advantage_count or total_multiplier < 1


def calculate_type_multiplier(move_type: str, defender_types: list[str]) -> float:
    """Calculates damage multiplier considering defender's type(s)"""
    multiplier = 1.0

    for defender_type in defender_types:
        try:
            current_effectiveness = constants.TYPE_EFFECTIVENESS[move_type][defender_type]
            multiplier *= current_effectiveness

            # Apply special abilities that modify type effectiveness
            if current_effectiveness > 1 and defender_type in ["filter", "solid_rock"]:
                multiplier *= 0.75  # Reduces super-effective damage
            if defender_type == "wonder_guard" and current_effectiveness <= 1:
                multiplier = 0  # Immune to non-super-effective hits
        except KeyError:
            print(f"Warning: Effectiveness for {move_type} against {defender_type} not found.")

    return max(multiplier, 0)


def calculate_damage(attacker: Pokemon, defender: Pokemon, move: Move) -> int:
    """Calculate damage inflicted by move with additional modifier logic."""

    # Base damage calculation
    level = attacker.level
    attack_stat = get_attack_stat(attacker, move)
    defense_stat = get_defense_stat(defender, move)
    damage = (((2 * level / 5 + 2) * move.basePower * (attack_stat / defense_stat)) / 50 + 2)

    # STAB (Same-Type Attack Bonus)
    if move.type in attacker.types:
        damage *= stab_modifier(attacker, move)

    # Add type effectiveness
    type_multiplier = calculate_type_multiplier(move.type, defender.types)
    damage *= type_multiplier

    # Apply external conditions
    damage *= apply_damage_conditions(defender, move)

    # Apply random variance (from 0.85 to 1.0)
    damage *= random.uniform(0.85, 1.0)

    return int(damage)


def stab_modifier(attacking_pokemon, attacking_move):
    """Calculates the STAB (Same-Type Attack Bonus) multiplier. The damage is increased by 50% if the move type
    matches the type of the Pokémon."""
    if attacking_move.type in attacking_pokemon.types:
        # Check if the Pokémon is Terastallized and its Terastal type matches the move type
        if (
                attacking_pokemon.terastallized and
                attacking_pokemon.types[0] in pokedex[attacking_pokemon.id].types
        ):
            return 2  # Enhanced STAB bonus for Terastalization
        else:
            return 1.5  # Standard STAB bonus

    elif attacking_pokemon.terastallized and attacking_move.type in pokedex[attacking_pokemon.id].types:
        return 1.5  # STAB bonus when the Terastal type matches the move type

    return 1  # No STAB bonus


def apply_damage_conditions(defender: Pokemon, move: Move) -> float:
    """Apply damage reduction modifiers like Reflect, Light Screen, and Multiscale."""
    modifier = 1.0
    side_conditions = getattr(defender, 'side_conditions', {})

    # Check defensive conditions
    if side_conditions.get(constants.REFLECT) and move.category == constants.PHYSICAL:
        modifier *= 0.5
    if side_conditions.get(constants.LIGHT_SCREEN) and move.category == constants.SPECIAL:
        modifier *= 0.5
    if side_conditions.get(constants.AURORA_VEIL):
        modifier *= 0.5
    if defender.ability == "multiscale" and defender.hp == defender.max_hp:
        modifier *= 0.5
    if defender.ability in ["filter", "solid_rock"] and calculate_type_multiplier(move.type, defender.types) > 1:
        modifier *= 0.75
    if getattr(defender, 'partner_ability', '') == "friend_guard":
        modifier *= 0.75

    return modifier


def is_type_advantage_move(move_name: str, opponent_types: list[str]) -> bool:
    """Checks if a move has a type advantage against the opponent's active Pokémon."""
    move = Move(move_name)
    type_multiplier = calculate_type_multiplier(move.type, opponent_types)
    return type_multiplier > 1


def get_attack_stat(pokemon: Pokemon, move: Move) -> float:
    return pokemon.stats[constants.ATTACK] if move.category == constants.PHYSICAL else pokemon.stats[
        constants.SPECIAL_ATTACK]


def get_defense_stat(pokemon: Pokemon, move: Move) -> float:
    return pokemon.stats[constants.DEFENSE] if move.category == constants.PHYSICAL else pokemon.stats[
        constants.SPECIAL_DEFENSE]
