import json
import math
import random
import time
from copy import deepcopy
from typing import override

import constants
from showdown.battle import Battle, Pokemon, Move
from showdown.battle_bots.helpers import format_decision

# JSON file containing all moves
with open('data/moves.json', 'r') as f:
    all_move_json = json.load(f)


class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    @override
    def find_best_move(self):
        """
        Finds best move or best switch using Minmax with decision formatting.
        """
        best_move = None
        max_value = float('-inf')

        # Check if active Pokemon is alive
        if not self.user.active:
            print("Error: active Pokémon is invalid or exausted.")
            # Returns first available switch
            switches = [f"{constants.SWITCH_STRING} {name}" for name in self.user.get_switches()]
            if switches:
                selected_switch = format_decision(self, switches[0])
                print(f"Selected switch: {selected_switch}")
                return selected_switch
            return ["no valid move or switch"]

        # Get all available moves and switches
        user_options, _ = self.get_all_options()
        print(f"Available moves: {user_options}")
        moves, switches = BattleBot.options_categorization(user_options)

        start_time = time.time()  # timer start
        time_limit = 2 * 60 + 30  # 2 minutes 30 seconds time limit

        # If we're forced to switch or there are no available moves the first switch is returned
        if self.force_switch or not moves:
            if not switches:
                print("Error: no available switch.")
                return ["no valid move or switch"]
            
            switch = self.find_best_switch()
            if switch is None:
                switch = self.get_pkmn_by_switch(switches[0])
            selected_switch = format_decision(self, f"{constants.SWITCH_STRING} {switch.name}")
            print(f"Selected switch: {selected_switch}")
            return selected_switch

        # Execute MinMax for each option
        for move in moves:
            if time.time() - start_time > time_limit:
                print("Time expired, returning best found move")
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
            print(f"Best found mmove: {selected_move}")
            return selected_move # returns formatted decision

        # In case there are no valid moves pokemons get switched
        switch = self.find_best_switch()
        assert switch is not None
        selected_switch = format_decision(self, f"{constants.SWITCH_STRING} {switch.name}")
        print(f"Switch selezionato: {selected_switch}")
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

        return (moves, switches)

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
                    print(f"{self.user.active.name} use {move.name}")

                    # Damage calculation considering types
                    type_multiplier = calculate_type_multiplier(move.type, self.opponent.active.types)

                    damage = calculate_damage(self.user.active, self.opponent.active, move)
                    damage *= type_multiplier

                    self.opponent.active.hp -= math.floor(damage)
                    print(
                        f"{move.name} inflicted {damage} hp of damage to {self.opponent.active.name} with an efficacy multiplier of {type_multiplier}.")

                    if move.status is not None:
                        # The move has no secondary effects
                        self.opponent.active.status = move.status
                        print(f"{self.opponent.active.name} has been {move.status}!")
                else:
                    assert self.user.active is not None
                    print(f"{self.user.active.name} missed the move {move.name}.")
            else:
                print(f"Error: move {move_name} not found")

    def restore_state(self, saved_state):
        """Restores battle state after single move simulation"""
        self.__dict__.update(saved_state.__dict__)

    def minimax(self, is_maximizing: bool, alpha: float, beta: float, max_depth: int = 3) -> float:
        """Minimax algorithm with Alpha-Beta cutting-out."""
        # End conditions: max-depth reached or match ended
        if max_depth == 0 or self.game_over():
            return self.evaluate_state()

        assert self.user.active is not None

        if not self.user.active.is_alive():
            print("Error: No pokemon alive")
            # Switching pokemon
            switch = self.find_best_switch()
            if switch:
                self.apply_move(f"Switch {switch}")
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
                print(type_multiplier)
                if type_multiplier > 1:
                    ineffective_moves = False  # Effective move found

            eval = self.minimax(False, alpha, beta, max_depth - 1)  # Opponent turn
            self.restore_state(saved_state)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Alpha-Beta pruning

        if ineffective_moves and max_depth == 3:
            print("Every move is ineffective, looking for a switch")
            switch = self.find_best_switch()
            if switch:
                print(f"Switching with {switch}.")
                self.apply_move(f"switch {switch}")

        return max_eval

    def min_eval(self, alpha:float, beta:float, max_depth:int) -> float | int:
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

        # Check if active Pokemons are alive
        assert self.user.active is not None
        assert self.opponent.active is not None

        if not (self.user.active.is_alive() or self.opponent.active.is_alive()):
            print(f"Error: Pokemon not alive. score {score}.")
            return score

        # Score calculation based of HP of active Pokemons
        score += self.user.active.hp
        score -= self.opponent.active.hp

        # Bonus/penalty for each attitional pokemon in each team
        score += len([pokemon for pokemon in self.user.reserve if pokemon.hp > 0])
        score -= len([pokemon for pokemon in self.opponent.reserve if pokemon.hp > 0])

        # Penalty for disvantageous pokemon type
        if is_type_disadvantageous(self.user.active, self.opponent.active):
            score -= 50

        return score

    def find_best_switch(self) -> Pokemon | None:
        """Finds best switch in reserve (defence-wise)"""
        best_switch = None
        max_resistance = float('-inf')

        for switch in self.user.get_switches():
            pokemon_to_switch = self.get_pkmn_by_switch(switch)

            # Evaluate reserve pokemon types resistance
            if self.opponent.active is None:
                print("opponent pokemon is None")
                continue

            resistance = 1
            for opponent_pokemon_type in self.opponent.active.types:
                for pokemon_to_switch_type in pokemon_to_switch.types:
                    resistance *= constants.TYPE_EFFECTIVENESS[opponent_pokemon_type][pokemon_to_switch_type]

            if resistance > max_resistance:
                max_resistance = resistance
                best_switch = pokemon_to_switch

        
        print(f"best switch: {best_switch}")

        return best_switch

    def get_pokemon_by_name(self, name: str) -> Pokemon | None:
        """
        Returns the pokemon with the name took from user reserve.
        """
        for pokemon in self.user.reserve:
            if pokemon.name.lower() == name.lower():
                return pokemon
        return None

    def get_pkmn_by_switch(self, switch: str) -> Pokemon:
        """Returns pokemon from a switching string"""
        name = switch.split(' ')[1]
        pkmn = self.get_pokemon_by_name(name)
        assert pkmn is not None
        return pkmn


    def game_over(self) -> bool:
        """Checks if battle is over"""
        user_pokemon_alive = self.user.active.is_alive() if self.user.active is not None else False
        opponent_pokemon_alive = self.opponent.active.is_alive() if self.opponent.active is not None else False

        # Cheks if there are available pokemons to switch
        reserve_pokemon_alive = not self.user.get_switches() == []
        reserve_opponent_alive = not self.opponent.get_switches() == []

        # if no pokemon is alive battle is over
        return not (user_pokemon_alive or reserve_pokemon_alive or opponent_pokemon_alive or reserve_opponent_alive)


def is_type_disadvantageous(user: Pokemon, opponent: Pokemon) -> bool:
    """Checka if Pokemon type se il tipo del Pokémon usato dal bot è adatto allo scontro contro i due tipi avversari"""
    user_pokemon_types = user.types
    opponent_pokemon_types = opponent.types

    total_disadvantage = 1
    for user_pokemon_type in user_pokemon_types:
        for opponent_pokemon_type in opponent_pokemon_types:
            disadvantage = constants.TYPE_EFFECTIVENESS[opponent_pokemon_type][user_pokemon_type]
            total_disadvantage *= disadvantage

    return total_disadvantage < 1


def calculate_type_multiplier(move_type: str, defender_types: list[str]) -> float:
    """Calculates damage multiplier considering defender's type(s)"""
    multiplier = 1.0

    for defender_type in defender_types:
        multiplier *= constants.TYPE_EFFECTIVENESS[move_type][
            defender_type]

    return multiplier


def calculate_damage(attacker: Pokemon, defender: Pokemon, move: Move) -> float:
    """Calculated damage inflicted by the move"""

    attack_stat = get_attack_stat(attacker, move)
    defense_stat = get_defense_stat(defender, move)

    # Damage calculus from "Gen V onward": https://bulbapedia.bulbagarden.net/wiki/Damage
    level_dmg = 2 * attacker.level / 5 + 2
    power_dmg = level_dmg * move.basePower * (attack_stat / defense_stat) / 50
    burn = 1.5 if defender.status == constants.BURN else 1
    damage = power_dmg * burn + 2

    # Viene aggiunto un moltiplicatore di danno per il tipo
    type_multiplier = calculate_type_multiplier(move.type, defender.types)
    return damage * type_multiplier

def get_attack_stat(pokemon: Pokemon, move: Move) -> float:
    return pokemon.stats[constants.ATTACK] if move.category == constants.PHYSICAL else pokemon.stats[constants.SPECIAL_ATTACK]

def get_defense_stat(pokemon: Pokemon, move: Move) -> float:
    return pokemon.stats[constants.DEFENSE] if move.category == constants.PHYSICAL else pokemon.stats[constants.SPECIAL_DEFENSE]
