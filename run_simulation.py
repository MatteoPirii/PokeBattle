from simulation.jupyter_code import *
import random

TEAM_SIZE = 2
MOVES_SET_SIZE = 2
PKMN_LEVEL = 75
RUNS_FOR_GENOME = 32

def generate_teams() -> list[list[Pokemon]]:
    teams = [[], []]
    for i in range(2 * TEAM_SIZE):
        pkmn = random.choice(list(pokedex.keys()))
        moves = []
        mvs = [all_move_json[move] for move in all_move_json if all_move_json[move]["type"] in pokedex[pkmn]["types"]]
        attack = True
        for _ in range(MOVES_SET_SIZE):
            
            move = random.choice(mvs)
            while attack and move["category"] == constants.STATUS:
                move = random.choice(mvs)
            moves.append(move["name"])
            attack = False
        pokemon = Pokemon(pokedex[pkmn]["name"], moves, PKMN_LEVEL)

        teams[math.floor(i/2)].append(pokemon)
    return teams

def battle_setup(user_team, opponent_team) -> BattleBot:
    user = Battler({"active": user_team[0], "reserve": user_team[1::]})
    opponent = Battler({"active": opponent_team[0], "reserve": opponent_team[1::]})
    battle = BattleBot(user, opponent)
    return battle

def alive(battler: Battler) :
    assert battler.active is not None
    if not battler.active.is_alive():
        try:
            battler.active = random.choice(battler.reserve)
            battler.reserve = [res for res in battler.reserve if res != battler.active]
        except:
            battler.active = None
        return False, battler
    return True, battler

def switch_battler(battle: BattleBot) -> BattleBot:
    user = battle.user
    opponent = battle.opponent
    battle.user = opponent
    battle.opponent = user
    return battle

def battle(battle: BattleBot) -> int:
    turn = 0
    logger.info(f"Battle between {battle.user.active} + {battle.user.reserve} VS {battle.opponent.active} + {battle.opponent.reserve}")
    while not utility.game_over(battle.user, battle.opponent):
        if turn > 200:
            return 0

        assert battle.user.active is not None
        user_move = battle.find_best_move(battle.user.active.moves, battle.user.reserve)
        battle = switch_battler(battle)

        assert battle.user.active is not None
        opponent_move = battle.find_best_move(battle.user.active.moves, battle.user.reserve)
        battle = switch_battler(battle)
        # battler in normal state
        assert battle.user.active is not None
        assert battle.opponent.active is not None
        if battle.user.active.stats[constants.SPEED] > battle.opponent.active.stats[constants.SPEED]:
            battle.apply_move(user_move[0])
            battle = switch_battler(battle)
            can_move, battle.user = alive(battle.user)
            if can_move:
                battle.apply_move(opponent_move[0])
            battle = switch_battler(battle)
            #normal state
        else:
            battle = switch_battler(battle)
            battle.apply_move(opponent_move[0])
            battle = switch_battler(battle)
            #normal state
            can_move, battle.user = alive(battle.user)
            if can_move:
                battle.apply_move(user_move[0])
                battle.user.reserve = [res for res in battle.user.reserve if res != battle.user.active]
        
        if battle.user.active and battle.opponent.active:
            logger.info(f"Turn {turn} \n\tChallenger has {len(battle.user.reserve)} pokemons in reserve. {battle.user.active.name} has {battle.user.active.hp}HPs left\n\tOpponent has {len(battle.opponent.reserve)} pokemons in reserve. {battle.opponent.active.name} has {battle.user.active.hp}HPs left")
        turn += 1

    if battle.user.active and battle.user.active.is_alive():
        return 1
    return 0


def search():
    evolution = Evolution()
    max_score = 0
    m = ""
    gt = ""
    while evolution.population_size > 0:
        logger.error(f"Generation {evolution.generation} started with {evolution.population_size} genomes")
        genome_index = 0
        for genome in evolution.genomes:
            win = 0
            for i in range(RUNS_FOR_GENOME):
                # logger.error(f"Battle {i+1}/{RUNS_FOR_GENOME} of genome {genome_index} of generation {evolution.generation}")
                teams = generate_teams()
                user = battle_setup(teams[0], teams[1])
                win += battle(user)
            genome.score = win / RUNS_FOR_GENOME
            if genome.score > genome.parent_score:
                genome.save(evolution.generation, genome_index)
                gt = ">>"
            if genome.score >= max_score:
                max_score = genome.score
                m = "MAX! "
            logger.error(f"{m}Genome {genome_index} of generation {evolution.generation} scored {genome.score} {gt}")
            genome_index += 1
            m = ""
            gt = ""
        evolution = evolution.next_generation()

search()
