"""Block 1"""
from copy import deepcopy
import importlib
import json
import logging
import math
import os
import random
import time
import simulation.data.jupyter.constants as constants
import simulation.data.jupyter.utility as utility
import simulation.data.jupyter.state_eval as state_eval
import simulation.data.jupyter.helpers as helpers
from simulation.data.jupyter.utility import game_over, calculate_damage, calculate_type_multiplier
from simulation.data.jupyter.helpers import format_decision

logger = logging.getLogger(__name__)

PWD = os.getcwd() + "/simulation/data/jupyter/"

"""Block 2"""
# Maximum depth of exploration for minimax
MAX_DEPTH = 4

# 10 seconds of tolerance
TIME_TOLLERANCE = 10

# Define the path to the 'moves.json' file using the present working directory (PWD)
move_json_location = os.path.join(PWD, 'moves.json')
# Open the 'moves.json' file and load its content as a Python dictionary
with open(move_json_location) as f:
    all_move_json = json.load(f) # Load the JSON file

# Define the path to the 'pokedex.json' file using the present working directory (PWD)
pkmn_json_location = os.path.join(PWD, 'pokedex.json')
# Open the 'pokedex.json' file and load its content as a Python dictionary
with open(pkmn_json_location, 'r') as f:
    pokedex = json.load(f) # Load the JSON file

"""Block 3"""
from typing import List

def normalize_name(name: str):
    return name\
        .replace(" ", "")\
        .replace("-", "")\
        .replace(".", "")\
        .replace("\'", "")\
        .replace("%", "")\
        .replace("*", "")\
        .replace(":", "")\
        .strip()\
        .lower()\
        .encode('ascii', 'ignore')\
        .decode('utf-8')

def common_pkmn_stat_calc(stat: int, iv: int, ev: int, level: int):
    return math.floor(((2 * stat + iv + math.floor(ev / 4)) * level) / 100)

def calculate_stats(base_stats, level, ivs=(31,) * 6, evs=(85,) * 6, nature='serious'):
    new_stats = dict()

    new_stats[constants.HITPOINTS] = common_pkmn_stat_calc(
        base_stats[constants.HITPOINTS],
        ivs[0],
        evs[0],
        level
    ) + level + 10

    new_stats[constants.ATTACK] = common_pkmn_stat_calc(
        base_stats[constants.ATTACK],
        ivs[1],
        evs[1],
        level
    ) + 5

    new_stats[constants.DEFENSE] = common_pkmn_stat_calc(
        base_stats[constants.DEFENSE],
        ivs[2],
        evs[2],
        level
    ) + 5

    new_stats[constants.SPECIAL_ATTACK] = common_pkmn_stat_calc(
        base_stats[constants.SPECIAL_ATTACK],
        ivs[3],
        evs[3],
        level
    ) + 5

    new_stats[constants.SPECIAL_DEFENSE] = common_pkmn_stat_calc(
        base_stats[constants.SPECIAL_DEFENSE],
        ivs[4],
        evs[4],
        level
    ) + 5

    new_stats[constants.SPEED] = common_pkmn_stat_calc(
        base_stats[constants.SPEED],
        ivs[5],
        evs[5],
        level
    ) + 5

    new_stats = {k: int(v) for k, v in new_stats.items()}
    return new_stats

"""Block 4"""
class Chromosome:
    mutation_rate = 0.15

    def __init__(self, value: float, variance: float):
        self.value = value
        self.variance = variance

    @staticmethod
    def random_mutation(chromosome: 'Chromosome') -> 'Chromosome':
        "Applies random mutation proportional to the value"
        if random.random() <= Chromosome.mutation_rate:
            return chromosome

        # the value of the mutation is a random percentage of the variance, whose sign is randomly chosen
        mutation = (random.random() * random.choice([-1, 1])) * chromosome.variance
        variance = utility.avg([mutation, chromosome.variance])
        return Chromosome(chromosome.value * mutation / 100, variance)

"""Block 5"""
class Genome:
    def __init__(self, genes: dict[str, Chromosome], parent_score: float = 0):
        self.genes = genes
        self.parent_score = parent_score
        self.score: float
    
    @classmethod
    def from_file(cls, filename: str = "base_genome") -> 'Genome':
        file = open(f"simulation/data/jupyter/{filename}.json", "r")
        data: dict[str, dict[str, float] | str] = json.load(file)
        genes: dict[str, Chromosome] = {}
        for element in data:
            if element == "parent_score" or element == "score": continue
            el = data[element]
            assert isinstance(el, dict)
            value = float(el["value"])
            variance = float(el["variance"])

            genes[element] = Chromosome(value, variance)
        s = data.get("score", 0)
        score = float(s)
        return Genome(genes, score)

    def value(self, gene: str | None) -> float | int:
        "returns the value of the gene"
        if not gene:
            return 0
        return self.genes[gene].value

    def __getitem__(self, item_name: str) -> float | int:
        return self.value(item_name)
    
    def save(self, generation: int, genome_index: int):
        data = dict()
        stable = "Stable_"
        data["score"] = self.score
        data["parent_score"] = self.parent_score or 0
        for gene in self.genes:
            gene_map = {"value": self.value(gene), "variance": self.variance(gene)}
            data[gene] = gene_map
            
            # doesn't print stable if the variance isn't
            if self.variance != 0:
                stable = ""
        file = open(f"data/evolution/{stable}gen{generation}_n{genome_index}.json", "w")
        json.dump(data, file, indent=4)

    def variance(self, gene: str) -> float | int:
        "returns the variance/instability of the gene"
        return self.genes[gene].variance

"""Block 6"""
# Define simple move class to simulate move options
class Move:
    def __init__(self, name):
            
        name = normalize_name(name)
        if constants.HIDDEN_POWER in name and not name.endswith(constants.HIDDEN_POWER_ACTIVE_MOVE_BASE_DAMAGE_STRING):
            name = "{}{}".format(name, constants.HIDDEN_POWER_ACTIVE_MOVE_BASE_DAMAGE_STRING)
        if name == "hiddenpower60":
            name = "hiddenpower"
        move_json = all_move_json[name]
        self.name = name
        self.max_pp = int(move_json.get(constants.PP) * 1.6)

        self.disabled = False
        self.can_z = False
        self.current_pp = self.max_pp
        self.accuracy: int | bool
        try:
            self.accuracy = int(move_json.get(constants.ACCURACY))
        except:
            self.accuracy = bool(move_json.get(constants.ACCURACY))

        self.basePower: int = int(move_json.get(constants.BASE_POWER))
        self.type: str = move_json.get(constants.TYPE)
        self.status: str = move_json.get(constants.STATUS)
        self.category: str = move_json.get(constants.CATEGORY)

"""Block 7"""
class Boost:
    def items(self):
        return []
class Pokemon:

    def __init__(self, name: str, moves: list[str], level = 70, nature="serious", evs=(85,) * 6):
        self.name = normalize_name(name)
        self.nickname = None
        self.base_name = self.name
        self.level = level
        self.nature = nature
        self.evs = evs
        #self.speed_range = StatRange(min=0, max=float("inf"))

        try:
            self.base_stats = pokedex[self.name][constants.BASESTATS]
        except KeyError:
            try:
                self.name = [k for k in pokedex if self.name.startswith(name)][0]
                self.base_stats = pokedex[self.name][constants.BASESTATS]
            except:
                print(name)

        self.stats = calculate_stats(self.base_stats, self.level, nature=nature, evs=evs)

        self.max_hp = self.stats.pop(constants.HITPOINTS)
        self.hp = self.max_hp
        if self.name == 'shedinja':
            self.max_hp = 1
            self.hp = 1

        self.ability = None
        self.types = pokedex[self.name][constants.TYPES]
        self.item = constants.UNKNOWN_ITEM

        self.terastallized = False
        self.fainted = False
        self.reviving = False
        self.moves = moves
        self.status: str | None = None
        self.volatile_statuses = []
        self.boosts = Boost()
        self.can_mega_evo = False
        self.can_ultra_burst = False
        self.can_dynamax = False
        self.is_mega = False
        self.can_terastallize = False
        self.can_have_assaultvest = True
        self.can_have_choice_item = True
        self.can_not_have_band = False
        self.can_not_have_specs = False
        self.can_have_life_orb = True
        self.can_have_heavydutyboots = True

    def is_alive(self):
        return self.hp > 0

    def __str__(self):
        return self.name

    def get_move(self, name: str) -> Move | None:
        for move in self.moves:
            if move == name:
                return Move(move)

    @classmethod
    def from_switch_string(cls, switch_string: str, nickname=None) -> 'Pokemon':
        details = switch_string.split(',')
        name = details[0]
        try:
            level = int(details[1].replace('L', '').strip())
        except (IndexError, ValueError):
            level = 100
        pkmn = Pokemon(name, [], level=level)
        pkmn.nickname = nickname
        return pkmn
"""Block 8"""
class Battler:
    def __init__(self, user_dict: dict):
        self.active: Pokemon | None = user_dict["active"]
        self.reserve: list[Pokemon] = user_dict["reserve"]
    
    @staticmethod
    def get_user_options(pokemon: Pokemon) -> List[str]:
        return [move for move in pokemon.moves]            # Extract the names of moves from the Pokémon's move list

    # Function to generate possible switches based on other Pokémon
    def get_switches(self) -> List[Pokemon]:
        return [pkmn for pkmn in self.reserve if pkmn.is_alive()]    # Return all active Pokémon except the current one

"""Block 9"""
class BattleBot:
    def __init__(self, user, opponent):
        assert isinstance(user, Battler)
        assert isinstance(opponent, Battler)
        self.user = user
        self.opponent = opponent
        self.start_time = None
        self.debug = True  # Example, this can be changed based on need
        self.time_remaining = 100  # Example starting time
        self.force_switch = False  # Example, this can be changed based on need
        self.rqid = None
        self.genome = Genome.from_file()
        self.weather = None

    def get_pkmn_by_switch(self, switch: str) -> Pokemon:
        """Returns Pokémon from a switching string"""
        name = switch.split(' ')[1]
        pkmn = self.get_pokemon_by_name(name)
        assert pkmn is not None
        return pkmn

    def restore_state(self, saved_state):
        """Restores battle state after single move simulation"""
        self.__dict__.update(saved_state.__dict__)
    
    def get_pokemon_by_name(self, name: str) -> Pokemon | None:
        """Returns the Pokémon with the name took from user reserve."""

        # Remove switch prefix if present
        if name.startswith(f"{constants.SWITCH_STRING} "):
            name = name.split(" ", 1)[1]

        normalized_name: str = normalize_name(name)
        for pokemon in self.user.reserve:
            if pokemon.name.lower() == normalized_name:
                return pokemon
        return None

    def find_best_move(self, user_options: list[str], switches) -> list[str]:
        """Finds best move or best switch using Minimax"""
        assert isinstance(user_options, list), f"user_options: {user_options}"
        assert isinstance(switches, list), f"switches: {switches}"
        assert self.user.active is not None
        best_move = None
        max_value = float('-inf')
        self.start_time = time.time()  # timer start

        # Check if the Pokémon is alive or inactive
        if not self.user.active.is_alive():
            switches = [f"{constants.SWITCH_STRING} {name}" for name in self.user.get_switches()]
            if switches:
                selected_switch = self.find_best_switch()
                if selected_switch:
                    self.apply_move(f"{constants.SWITCH_STRING} {selected_switch.name}")
                    self.time_remaining = utility.adjust_time(int(time.time() - self.start_time), self.time_remaining)
                    return format_decision(self, f"{constants.SWITCH_STRING} {selected_switch.name}")
            else:
                return ["no valid move or switch"]

        # Get all available moves and switches
        moves = user_options

        # If we're forced to switch or there are no available moves the first switch is returned
        if self.force_switch or not moves:
            if not switches:
                
                self.time_remaining = utility.adjust_time(int(time.time() - self.start_time), self.time_remaining)
                return ["no valid move or switch"]

            switch = self.find_best_switch()
            if switch is None:
                switch = self.get_pkmn_by_switch(switches[0])

            selected_switch = format_decision(self, f"{constants.SWITCH_STRING} {switch.name}")
            self.time_remaining = utility.adjust_time(int(time.time() - self.start_time), self.time_remaining)
            return selected_switch

        combined_options = moves.copy()
        combined_options.sort(key=lambda move: self.evaluate_move(move), reverse=True)
        switch = self.find_best_switch()
        if switch:
            combined_options.append(f"{constants.SWITCH_STRING} {switch.name}")

        # Execute MiniMax for each option
        for move in combined_options:
            saved_state = deepcopy(self)  # Saving the battle state
            self.apply_move(move)  # Choice simulation
            move_value = self.minimax(alpha=float('-inf'), beta=float('inf'))

            self.restore_state(saved_state)  # Battle state recovery

            if move_value > max_value:
                
                max_value = move_value
                best_move = move

        # Select highest damage move
        if best_move is not None and max_value > 0 and not game_over(self.user, self.opponent):
            selected_move = format_decision(self, best_move)
            
            self.time_remaining = utility.adjust_time(int(time.time() - self.start_time), self.time_remaining)
            return selected_move  # returns formatted decision

        best_move = random.choice(user_options)  # Random fallback choice
        selected_move = format_decision(self, best_move)
        self.time_remaining = utility.adjust_time(int(time.time() - self.start_time), self.time_remaining)
        return selected_move

    def apply_move(self, move_name: str) -> None:
        """Apply simulated move or switch considering type advantage"""
        move_part = move_name.split()

        if constants.SWITCH_STRING in move_name and len(move_part) > 1:
            self.user.active = Pokemon.from_switch_string(move_part[1])
            return

        # Damage, status effect and stats changes simulation
        if move_name.startswith("/choosemove"):
            move_name = move_name.replace("/choosemove", "")
        elif move_name.startswith("/choose move "):
            move_name = move_name.replace("/choose move ", "")

        move = Move(move_name)

        # Accuracy based move success rate calculation
        if random.randint(1, 100) > move.accuracy:
            return

        # Damage calculation considering types
        assert self.user.active is not None
        assert self.opponent.active is not None
        type_multiplier = calculate_type_multiplier(move.type, self.opponent.active.types)

        damage = calculate_damage(self.user.active, self.opponent.active, move)
        damage *= type_multiplier
        damage *= (self.user.active.level / self.opponent.active.level)

        self.opponent.active.hp -= math.floor(damage)

        # The move has no secondary effects
        if move.status is not None:
            self.opponent.active.status = move.status
            
    def evaluate_state(self) -> float:
        """Battle state evaluation"""
        assert self.user.active is not None
        assert self.opponent.active is not None
        fainted_score = 0
        # Check if active Pokémon are alive
        if not self.user.active.is_alive():
            fainted_score += self.genome["fainted_penalty"]  # Heavy penalty if user's active Pokémon is fainted
        if not self.opponent.active.is_alive():
            fainted_score += self.genome["fainted_bonus"]  # High reward if opponent's active Pokémon is fainted

        # 1. Scores by HP difference and level
        hp_percent_user = self.user.active.hp / self.user.active.max_hp
        hp_percent_opponent = self.opponent.active.hp / self.opponent.active.max_hp
        hp_score = hp_percent_user - hp_percent_opponent
        level_score = self.user.active.level - self.opponent.active.level

        # 2. Bonus/penalty for alive reserve
        user_reserve_score = sum(self.genome["reserve_bonus"] for p in self.user.reserve if p.hp > 0)
        opponent_reserve_score = sum(self.genome["reserve_penalty"] for p in self.opponent.reserve if p.hp > 0)
        reserve_score = utility.scale_range(user_reserve_score - opponent_reserve_score, [self.genome["reserve_penalty"] * -6, self.genome["reserve_bonus"] * 6])

        # 3. Bonus/penalty for type advantage/disadvantage
        type_advantage_multiplier = calculate_type_multiplier(self.user.active.types[0], self.opponent.active.types)
        type_score = 0
        if type_advantage_multiplier > 1:
            type_score = self.genome["type_advantage"] * type_advantage_multiplier  # Bonus for type advantage
        elif type_advantage_multiplier < 1 and type_advantage_multiplier != 0:
            type_score = -self.genome["type_disadvantage"] * (1 / type_advantage_multiplier)  # Penalty for type disadvantage
        else:
            type_score = self.genome["type_immunity_penalty"]

        # 4. Bonus for weather conditions
        weather_score = state_eval.weather_condition(self.user.active, self.opponent.active, self.weather)

        # 5. Penalty for status conditions
        status_score = self.genome.value(self.opponent.active.status) - self.genome.value(self.user.active.status)
        status_score = utility.scale_range(status_score, [- self.genome[constants.FROZEN], self.genome[constants.FROZEN]])

        # 6. Evaluate user's boost status
        boost_score = 0
        for _, value in self.user.active.boosts.items():
            if value < 0:
                boost_score += self.genome["boost_negative"]
            elif 0 < value <= 4:
                boost_score += self.genome["boost_positive"]
            elif value > 4:
                boost_score += self.genome["boost_excess"]

        status_move = [move for move in self.user.active.moves if
                       "heal" in all_move_json.get(move.lower(), {}).get("flags", {})]
        if status_move and hp_percent_user >= 0.85:
            boost_score += self.genome["useless_heal_penalty"]

        boost_score = utility.scale_range(boost_score, [-10 * self.genome["boost_negative"] - self.genome["useless_heal_penalty"], 10 * self.genome["boost_positive"]])

        # 7. Integrate worst-case opponent move analysis
        # opponent_moves = self.opponent.active.moves
        # opponent_move_score = 0
        # if opponent_moves:
        #     worst_opponent_score = min(
        #         self.evaluate_move_risk(move, self.user.active) for move in opponent_moves
        #     )
        #     opponent_move_score = worst_opponent_score  # Adjust score with worst-case scenario
        opponent_move_score = utility.scale_range(0, [-300, 300])

        # 8. Linear combination of scores
        score = 0
        score += self.genome["fainted"] * fainted_score
        score += self.genome["hp"] * hp_score
        score += self.genome["level"] * level_score
        score += self.genome["reserve"] * reserve_score
        score += self.genome["type"] * type_score
        score += self.genome["weather"] * weather_score
        score += self.genome["status"] * status_score
        score += self.genome["boost"] * boost_score
        score += self.genome["opponent_move"] * opponent_move_score

        # 9. Random factor for tie-breaking decisions
        score += random.uniform(-1, 1)

        return score

    def find_best_switch(self) -> Pokemon | None:
        # Find the best Pokémon in the team to make the switch.
        assert self.opponent.active is not None
        best_pokemon = None
        max_score = float('-inf')
        best_move_score = float('-inf')
        best_pokemon_candidates = []
        opponent_types = self.opponent.active.types
        pokemon_to_switch = None

        for switch in self.user.get_switches():
            pokemon_to_switch = switch

            if pokemon_to_switch is None:
                    continue

            # Evaluate the resistance of the reserve Pokémon against the opponent's type
            resistance = 0
            resistance += sum(constants.TYPE_EFFECTIVENESS.get(switch_type, {}).get(opponent_type, 1)
                              for opponent_type in opponent_types
                              for switch_type in pokemon_to_switch.types)

            # Combine resistance and move score for overall evaluation
            total_move_score = resistance

            if total_move_score > max_score:
                max_score = total_move_score
                best_pokemon_candidates = [pokemon_to_switch]  # Reset the list of best candidates
            elif total_move_score == max_score:
                best_pokemon_candidates.append(pokemon_to_switch)  # Add to the list of best candidates

        if not pokemon_to_switch:
            return None
        # Choose the best Pokémon from the candidates
        if best_pokemon_candidates:
            # if we have more than one candidate choose a heuristic to select the best Pokémon
            # we choose the Pokémon with the best move
            for pokemon in best_pokemon_candidates:

                # Calculate the move score
                move_score = self.pokemon_score_moves(pokemon_to_switch.name)

                if move_score > best_move_score:
                    best_move_score = move_score
                    best_pokemon = pokemon
            # if self.debug:
            

    def pokemon_score_moves(self, pokemon_name: str) -> float:
        """Find if the moves of the Pokémon are good or not"""
        pokemon = self.get_pokemon_by_name(pokemon_name)  # Get Pokémon object from the name
        max_score = float('-inf')

        if pokemon is None:
            
            return 0

        total_score = 0

        for mv in pokemon.moves:
            move = Move(mv)
            move_score = self.evaluate_move(move.name)
            accuracy_multiplier = move.accuracy / 100 if isinstance(move.accuracy,
                                                                    int) else 1  # Assuming accuracy is between 0 and
            # 100 (adjust the move score based on the accuracy of the move)
            total_score = move_score * accuracy_multiplier

            if total_score > max_score:
                max_score = total_score


        return total_score

    def evaluate_move(self, move_name: str) -> float:
        """Evaluate the move based on the type effectiveness against the opponent's Pokémon."""
        if isinstance(move_name, Move):            
            move_name = move_name.name  # Usa il nome se è un oggetto Move


        move = Move(move_name)  # Create an instance of Move using the move name as a string
        if not move:
            return 0

        # Calculate the type multiplier
        assert self.opponent.active is not None
        type_multiplier = calculate_type_multiplier(move.type, self.opponent.active.types)
        # Calculate the potential damage inflicted
        damage = calculate_damage(self.user.active, self.opponent.active, move) * type_multiplier

        # Consider the opponent's Pokémon level
        assert self.user.active is not None
        damage *= (self.user.active.level / self.opponent.active.level)

        return damage

    def minimax(self, alpha: float, beta: float, max_depth: int = MAX_DEPTH) -> float:
        """Minimax algorithm with Alpha-Beta cutting-out."""

        if self.is_terminal(max_depth):
            score = self.evaluate_state()
            return score

        return self.max_eval(alpha, beta, max_depth)

    def max_eval(self, alpha: float, beta: float, max_depth: int) -> float:
        assert self.user.active is not None
        max_eval = float('-inf')
        user_options = self.user.active.moves

        if self.is_terminal(max_depth):
            score = self.evaluate_state()
            return score

        # Separate moves from switches
        moves = [move for move in user_options if not move.startswith(constants.SWITCH_STRING)]

        # Sort moves by importance
        moves.sort(key=lambda move: self.evaluate_move(move), reverse=True)
        switch = self.find_best_switch()
        if switch:
            moves.append(f"{constants.SWITCH_STRING} {switch.name}")

        # Evaluate moves
        for move in moves:
            saved_state = deepcopy(self)  # Save the battle state before simulating the move

            self.apply_move(move)
            eval = self.min_eval(alpha, beta, max_depth - 1)
            self.restore_state(saved_state)  # Restore the battle state

            if eval > max_eval:
                max_eval = eval
                alpha = max(alpha, eval)

            if max_eval >= beta:
                return max_eval  # Alpha-Beta pruning

        return max_eval

    def min_eval(self, alpha: float, beta: float, max_depth: int) -> float | int:
        assert self.opponent.active is not None
        min_eval = float('inf')
        opponent_options = self.opponent.active.moves

        if self.is_terminal(max_depth):
            score = self.evaluate_state()
            return score

        for move in opponent_options:
            saved_state = deepcopy(self)  # Save battle state before moving

            self.apply_move(move)
            eval = self.max_eval(alpha, beta, max_depth - 1)  # Bot turn, maximizing
            self.restore_state(saved_state)

            if eval < min_eval:
                min_eval = eval
                beta = min(beta, eval)

            if min_eval <= alpha:
                return min_eval

        return min_eval
    
    def is_terminal(self, max_depth: int) -> bool:
        """Checks wether or not the game is in a terminal state"""
        if self.is_time_over():
            
            return True

        # End conditions: max-depth reached or match ended
        if max_depth == 0 or utility.game_over(self.user, self.opponent):
            return True

        return False
    
    def is_time_over(self) -> bool:
        """Checks if timer of a battle is over"""
        if self.time_remaining is None:
            self.time_remaining = 150

        assert self.start_time is not None
        effective_timer = self.time_remaining - TIME_TOLLERANCE
        elapsed_time = time.time() - self.start_time


        return elapsed_time > effective_timer

"""Block 17"""
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


"""Block 20"""
from itertools import combinations
class Evolution:
    def __init__(self, genomes: list[Genome] | None = None, generation: int = 0):
        if genomes:
            self.genomes = genomes
            self.population_size: int = len(genomes)
        else:
            self.population_size = 5
            self.genomes = Evolution.create_genomes(self.population_size)
        self.generation = generation
        self.restore = False
        logger.info(f"Generation {self.generation} created, population size: {self.population_size}")

    @staticmethod
    def create_genomes(n: int, base: Genome = Genome.from_file()) -> list[Genome]:
        "Creates n chromosomes from a base Chromosome"
        logger.debug(f"Created {n} new genomes with a base of score {base.parent_score or 0}")
        genomes = []
        for _ in range(0, n):
            gen: dict[str, Chromosome] = {}
            for gene in base.genes:
                gene_value = base.value(gene) + random.gauss(0, pow(base.variance(gene), 2))
                gen[gene] = Chromosome(gene_value, base.variance(gene))
            genomes.append(Genome(gen))
        return genomes

    @staticmethod
    def recombine(genome1: Genome, genome2: Genome) -> Genome:
        "Recombination procedure for two genomes (crossover)"
        genes = {}
        for gene in genome1.genes:
            gene_value = (genome1.value(gene) + genome2.value(gene)) / 2

            max_gene = max(genome1, genome2, key=lambda x: x.value(gene)).value(gene)
            min_gene = min(genome1, genome2, key=lambda x: x.value(gene)).value(gene)
            value_variance = utility.scale_range(gene_value, [min_gene, max_gene], [0, 1]) / 100
            gene_variance = value_variance * (genome1.variance(gene) * genome2.variance(gene) / 100)

            genes[gene] = Chromosome.random_mutation(Chromosome(gene_value, gene_variance))

        parent_score = utility.avg([genome1.score, genome2.score])
        return Genome(genes, parent_score)

    def culling(self) -> list[Genome]:
        """
        Next generation makeup tecnique, all the genomes that scored lower than their parent average are discarded
        """
        genomes = [genome for genome in self.genomes if genome.score >= genome.parent_score]
        return genomes

    def selection(self, genomes: list[Genome] | None = None) -> list[tuple[Genome, Genome]]:
        if not genomes:
            genomes = self.genomes
        return list(combinations(genomes, 2))

    def next_generation(self) -> 'Evolution':
        "Creates the next generation from the current one"
        logger.debug(f"Generating generation {self.generation+1}")
        genomes = self.culling() #performs culling
        couples = self.selection(genomes) #selects the couples that will create offsprings
        logger.error(f"Culling procedure let {len(genomes)} survive the selection, {len(couples)} offsprings will be generated")

        # offsprings generation
        offsprings: list[Genome] = []
        for couple in couples:
            offspring = self.recombine(couple[0], couple[1])
            offsprings.append(offspring)
        
        # Adds offsprings to prevent population collapse
        if self.restore and len(offsprings) < self.population_size:
            best_genome = max(genomes, key=lambda genome: genome.score)

            # adds offsprings as varaitions of the best scoring genome
            offsprings += Evolution.create_genomes(self.population_size - len(offsprings), best_genome)

        return Evolution(offsprings, self.generation + 1)



