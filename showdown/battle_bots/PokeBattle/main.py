import constants
import json
import math
import random
import time

from copy import deepcopy
from data import pokedex
from showdown.battle import Battle, Pokemon, Move
from showdown.battle_bots.PokeBattle import utility, state_eval
from showdown.battle_bots.helpers import format_decision
from showdown.engine import helpers

# Maximum depth of exploration for minimax
MAX_DEPTH = 12

# JSON file containing all moves
with open('data/moves.json', 'r') as f:
    all_moves = json.load(f)


class BattleBot(Battle):

    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)
        self.debug = False
        self.start_time: float = 0

    def find_best_move(self) -> list[str]:
        """Finds best move or best switch using Minimax"""
        best_move = None
        self.start_time = time.time()  # timer start

        # Check if the Pokémon is alive or inactive
        if not self.user.active.is_alive() or self.force_switch:
            print("Error: active Pokémon is invalid or exhausted.") if self.debug else None
            switch = self.switch_pokemon()
            print(f"Returning {switch}")
            if switch is not None:
                return switch
            # if forced to switch and no switch available backup plan
            return format_decision(self, constants.DO_NOTHING_MOVE)

        # Execute MinMax for each option
        score, best_move = self.minimax()

        # Select highest damage move
        if best_move is not constants.DO_NOTHING_MOVE:
            selected_move = format_decision(self, best_move)
            print(f"Best found move: {selected_move} with score {score}") #if self.debug else None
            return selected_move  # returns formatted decision

        # If no optimal switches of moves
        options, _ = self.get_all_options()
        chosen = random.choice(options)
        print(f"Choose {chosen} randolmy")
        return format_decision(self, chosen)

    def apply_move(self, move_name: str) -> None:
        """Apply simulated move or switch considering type advantage"""
        move_part = move_name.split()

        if constants.SWITCH_STRING in move_name and len(move_part) > 1:
            self.user.active = Pokemon.from_switch_string(move_part[1])
            return

        # Damage, status effect and stats changes simulation
        move = Move(move_name)

        # Accuracy based move success rate calculation
        if random.randint(1, 100) > move.accuracy:
            if self.debug:
                print(f"{self.user.active.name} missed the move {move.name}.")
            return

        # Damage calculation considering types
        type_multiplier = calculate_type_multiplier(move.type, self.opponent.active.types)

        damage = calculate_damage(self.user.active, self.opponent.active, move)
        damage *= type_multiplier
        damage *= (self.user.active.level / self.opponent.active.level)

        self.opponent.active.hp -= math.floor(damage)
        print(f"{move.name} inflicted {damage:.2f} hp of damage to {self.opponent.active.name} with an "
              f"efficacy multiplier of {type_multiplier}.") if self.debug else None

        # The move has no secondary effects
        if move.status is not None:
            self.opponent.active.status = move.status
            print(f"{self.opponent.active.name} has been {move.status}!") if self.debug else None
        else:
            print(f"{self.user.active.name} missed the move {move.name}.") if self.debug else None

    def restore_state(self, saved_state):
        """Restores battle state after single move simulation"""
        self.__dict__.update(saved_state.__dict__)

    def minimax(self) -> tuple[float, str]:
        """Minimax algorithm with Alpha-Beta cutting-out."""

        alpha = float('-inf')
        beta = float('inf')
        max_depth = MAX_DEPTH
        if len(self.user.reserve) <= 2:
            max_depth = 7  # Dynamic depth adjustment

        return self.max_eval(alpha, beta, max_depth)

    def max_eval(self, alpha: float, beta: float, max_depth: int) -> tuple[float, str]:
        max_eval = float('-inf')
        best_action = constants.DO_NOTHING_MOVE
        user_options, _ = self.get_all_options()

        # Separate moves from switches
        moves, switches = utility.options_categorization(user_options)
        # Sort moves by importance
        moves.sort(key=lambda move: self.evaluate_move(move), reverse=True)

        # Evaluate moves
        for option in moves + switches:
            saved_state = deepcopy(self)  # Save the battle state before simulating the move

            if self.is_terminal(max_depth):
                return max_eval, best_action

            self.apply_move(option)
            eval, _ = self.min_eval(alpha, beta, max_depth - 1)
            assert not eval == float('inf')
            self.restore_state(saved_state)  # Restore the battle state

            if eval > max_eval:
                max_eval = eval
                best_action = option
                alpha = max(alpha, eval)

            if max_eval >= beta:
                return max_eval, best_action  # Alpha-Beta pruning

        """
        # Evaluate switches without saving/restoring state, as it isn't necessary
        for switch in switches:
            if self.is_terminal(max_depth):
                return max_eval, best_action
            eval = self.evaluate_switch(switch)  # Evaluate based on the current state

            if eval > max_eval:
                max_eval = eval
                best_action = f"{constants.SWITCH_STRING} {switch}"
                alpha = max(alpha, eval)

            if max_eval >= beta:
                return max_eval, best_action  # Alpha-Beta pruning
        """

        return max_eval, best_action

    def min_eval(self, alpha: float, beta: float, max_depth: int) -> tuple[float, str]:
        min_eval = float('inf')
        best_action = constants.DO_NOTHING_MOVE
        _, opponent_options = self.get_all_options()

        # If opponent options are unknown there's nothing to estimate about him
        if opponent_options == [] or opponent_options == [constants.DO_NOTHING_MOVE]:
            return self.evaluate_state(), best_action

        for move in opponent_options:
            saved_state = deepcopy(self)  # Save battle state before moving
            if self.is_terminal(max_depth):
                return self.evaluate_state(), best_action

            self.apply_move(move)
            eval, _ = self.max_eval(alpha, beta, max_depth - 1)  # Bot turn, maximizing
            self.restore_state(saved_state)

            if eval < min_eval:
                min_eval = eval
                best_action = move
                beta = min(beta, eval)


            if min_eval <= alpha:
                assert not min_eval == float('inf'), f"Infinite score inside loop, checked: {opponent_options}"

                return min_eval, best_action

        assert not min_eval == float('inf'), f"Infinite score outside of loop, checked: {opponent_options}"
        return min_eval, best_action

    def is_terminal(self, max_depth: int) -> bool:
        "Checks weather or not the game is in a terminal state"
        if utility.is_time_over(self.start_time, self.time_remaining):
            print("Time expired, returning best found move") if self.debug else None
            return True

        # End conditions: max-depth reached or match ended
        if max_depth == 0 or utility.game_over(self.user, self.opponent):
            return True

        return False

    def evaluate_state(self) -> float:
        """Battle state evaluation"""
        score = 0

        # Check if active Pokémon are alive
        if not (self.user.active.is_alive() or self.opponent.active.is_alive()):
            print(f"Error: Pokemon not alive. score {score}.") if self.debug else None
            return score

        # 1. Scores by hp difference and level
        user_hp_advantage = self.user.active.hp - self.opponent.active.hp
        level_advantage = (self.user.active.level - self.opponent.active.level) * 10
        score += user_hp_advantage + level_advantage

        # 2. Bonus/penalty for alive reserve
        user_reserve_score = sum(10 for p in self.user.reserve if p.hp > 0)
        opponent_reserve_score = sum(10 for p in self.opponent.reserve if p.hp > 0)
        score += user_reserve_score - opponent_reserve_score

        # 3. Bonus/penalty for type advantage/disadvantage
        if is_type_disadvantageous(self.user.active, self.opponent.active):
            score -= 40
        else:
            score += 50

        # 4 Bonus for weather conditions
        score += state_eval.weather_condition(self.user.active, self.weather)
        # Consider weather-based penalties for the opponent
        score -= state_eval.weather_condition(self.opponent.active, self.weather)

        # 5. Penalty for status conditions
        status_penalty = {
            constants.PARALYZED: 20,
            constants.POISON: 15,
            'badly poisoned': 25,
            constants.BURN: 20,
            constants.SLEEP: 30,
            constants.FROZEN: 40
        }
        score -= status_penalty.get(self.user.active.status, 0)

        # 6. Bonus for moves that have set up effects (e.g., Swords Dance, Calm Mind)
        for move in self.user.active.moves:
            if move.name in ['Swords Dance', 'Calm Mind', 'Nasty Plot']:
                score += 30  # Arbitrary bonus for setup moves

        # Integrate worst-case opponent move analysis
        opponent_moves = self.opponent.active.moves
        if opponent_moves:
            worst_opponent_score = min(
                self.evaluate_move_risk(move, self.user.active) for move in opponent_moves
            )
            score += worst_opponent_score  # Adjust score with worst-case scenario

        return score

    @staticmethod
    def evaluate_move_risk(move: Move, pokemon: Pokemon) -> float:
        """Evaluate the risk posed by an opponent's move based on potential damage."""
        type_multiplier = calculate_type_multiplier(move.type, pokemon.types)
        damage_potential = move.basePower * type_multiplier
        accuracy_factor = move.accuracy / 100 if isinstance(move.accuracy, int) else 1  # Account for move accuracy
        risk_score = -(damage_potential * accuracy_factor)
        return risk_score  # Negative for riskier moves

    def find_best_switch(self) -> Pokemon | None:
        """
        Finds the best Pokémon in the team to make the switch
        """
        best_pokemon = None
        max_score = float('-inf')
        best_pokemon_candidates = [None]

        for switch in self.user.get_switches():
            pokemon_to_switch = self.get_pokemon_by_name(switch)

            if pokemon_to_switch is None:
                if self.debug:
                    print(f"Error: Pokémon {switch} not found in the user's reserve.")
                continue

            # Evaluate the resistance of the reserve Pokémon against the opponent's type
            resistance = 0
            resistance += sum(constants.TYPE_EFFECTIVENESS.get(switch_type, {}).get(opponent_type, 1)
                              for opponent_type in self.opponent.active.types
                              for switch_type in pokemon_to_switch.types)

            # Calculate the move score
            best_move_score = self.pokemon_score_moves(pokemon_to_switch)

            # Combine resistance and move score for overall evaluation
            total_score = resistance + best_move_score

            if total_score > max_score:
                max_score = total_score
                best_pokemon_candidates = [pokemon_to_switch] # Reset the list of best candidates
            elif total_score == max_score:
                best_pokemon_candidates.append(pokemon_to_switch) # Add to the list of best candidates

        # Choose the best Pokémon from the candidates
        if best_pokemon_candidates:
            #if we have more than one candidate choose an heuristic to select the best pokemon
            #for the moment we choose randomly
            best_pokemon = random.choice(best_pokemon_candidates)
            if self.debug:
                print(f"Best switch: {best_pokemon.name} with a total score of {max_score}")
        elif self.debug:
            print("No suitable Pokémon found.")

        return best_pokemon

    def pokemon_score_moves(self, pokemon: Pokemon) -> float:
        """Find if the moves of the Pokémon are good or not"""
        max_score = float('-inf')
        total_score = 0

        for move in pokemon.moves:
            move_score = self.evaluate_move(move)
            # Assuming accuracy is between 0 and 100 (adjust the move score based on the accuracy of the move)
            accuracy_multiplier = move.accuracy / 100 if isinstance(move.accuracy, int) else 1
            total_score = move_score * accuracy_multiplier

            if total_score > max_score:
                max_score = total_score

        print(f"Total score for {self.user.name}: {total_score}") if self.debug else None
        return total_score

    def evaluate_move(self, move_name: str) -> float:
        """Evaluate the move based on the type effectiveness against the opponent's Pokémon."""
        if isinstance(move_name, Move):
            move_name = move_name.name  # Usa il nome se è un oggetto Move

        move = Move(move_name)  # Create an instance of Move using the move name as a string
        if not move:
            print(f"Error: Move {move_name} not found.") if self.debug else None
            return float('-inf')

        # Calculate the type multiplier
        type_multiplier = calculate_type_multiplier(move.type, self.opponent.active.types)
        # Calculate the potential damage inflicted
        damage = calculate_damage(self.user.active, self.opponent.active, move) * type_multiplier

        # Consider the opponent's Pokémon level
        damage *= (self.user.active.level / self.opponent.active.level)

        print(f"Move {move.name} inflicts {damage:.2f} damage to {self.opponent.active.name}.") if self.debug else None
        return damage
    
    """def normalize_score(self, score: float, a = -1000, b = 1000) -> float:
        #Normalize score betweeen -1000 and 1000
        
        #calculate minimum and maximum score
        min_score = self.calc_score(level = 1, basePower = 20, attack_stat = 5, defense_stat = 700)
        max_score = self.calc_score(level = 100, basePower = 250, attack_stat = 700, defense_stat = 5)
        
        return a + ((score - min_score) * (b - a)) / (max_score - min_score)
    
    def calc_score (self, level, basePower, attack_stat, defense_stat) -> float:
        score = (((2 * level / 5 + 2) * basePower * (attack_stat / defense_stat)) / 50 + 2)
        return score"""


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
        return pkmn

    def evaluate_switch(self, switch: str) -> float:
        """Evaluate a switch based on type matchups and overall strategy."""
        switch_name = switch.split(" ", 1)[1]
        pokemon = self.get_pokemon_by_name(switch_name)

        if not pokemon:
            return 0  # Return a neutral score if the Pokémon is not found

        # Evaluate type resistance against the opponent's active Pokémon
        type_advantage_score = sum(
            constants.TYPE_EFFECTIVENESS.get(pokemon_type, {}).get(opponent_type, 1)
            for opponent_type in self.opponent.active.types
            for pokemon_type in pokemon.types
        )

        # Add additional evaluation criteria (e.g., HP, available setup moves)
        hp_score = (pokemon.hp / pokemon.max_hp) * 10  # Bonus for higher HP
        setup_move_bonus = 10 if any(
            move.name in ['Stealth Rock', 'Spikes', 'Swords Dance'] for move in pokemon.moves) else 0

        # Combine type advantage score, HP score, and setup move bonus
        total_score = type_advantage_score + hp_score + setup_move_bonus

        return total_score

    def switch_pokemon(self) -> list[str] | None:
        selected_switch = self.find_best_switch()
        
        if selected_switch is None:
            return None

        self.apply_move(f"{constants.SWITCH_STRING} {selected_switch.name}")
        return format_decision(self, f"{constants.SWITCH_STRING} {selected_switch.name}")


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
    """Calculates damage multiplier considering defender's type(s)
    [0-4]"""
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
    attack_stat = utility.get_attack_stat(attacker, move)
    defense_stat = utility.get_defense_stat(defender, move)
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


def stab_modifier(attacking_pokemon: Pokemon, attacking_move: Move) -> float:
    """Calculates the STAB (Same-Type Attack Bonus) multiplier.
    The damage is increased by 50% if the move type matches the type of the Pokémon."""
    if attacking_move.type in attacking_pokemon.types:
            return 1.5  # Standard STAB bonus

    return 1  # No STAB bonus


def apply_damage_conditions(defender: Pokemon, move: Move) -> float:
    """Apply damage reduction modifiers like Reflect, Light Screen, and Multiscale."""
    modifier = 1.0
    side_conditions = getattr(defender, 'side_conditions', {})

    # Critical hits ignore damage reduction
    if getattr(move, 'critical_hint', False):
        return 1.0  # Ignore all reductions

    # Check defensive conditions
    if side_conditions.get(constants.REFLECT) and move.category == constants.PHYSICAL:
        modifier *= 0.5
    if side_conditions.get(constants.LIGHT_SCREEN) and move.category == constants.SPECIAL:
        modifier *= 0.5
    if side_conditions.get(constants.AURORA_VEIL):
        modifier *= 0.67
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
