import asyncio
import json
import logging
import traceback
from datetime import datetime
from copy import deepcopy

import constants
from config import ShowdownConfig, init_logging, env

from showdown.battle_bots.PokeBattle.evolution import Evolution
from teams import load_team
from showdown.run_battle import pokemon_battle
from showdown.websocket_client import PSWebsocketClient

from data import all_move_json
from data import pokedex
from data.mods.apply_mods import apply_mods


logger = logging.getLogger(__name__)


def check_dictionaries_are_unmodified(original_pokedex, original_move_json):
    # The bot should not modify the data dictionaries
    # This is a "just-in-case" check to make sure and will stop the bot if it mutates either of them
    if original_move_json != all_move_json:
        logger.critical("Move JSON changed!\nDumping modified version to `modified_moves.json`")
        with open("modified_moves.json", 'w') as f:
            json.dump(all_move_json, f, indent=4)
        exit(1)
    else:
        logger.debug("Move JSON unmodified!")

    if original_pokedex != pokedex:
        logger.critical(
            "Pokedex JSON changed!\nDumping modified version to `modified_pokedex.json`"
        )
        with open("modified_pokedex.json", 'w') as f:
            json.dump(pokedex, f, indent=4)
        exit(1)
    else:
        logger.debug("Pokedex JSON unmodified!")


async def showdown():
    ShowdownConfig.configure()
    init_logging(
        ShowdownConfig.log_level,
        ShowdownConfig.log_to_file
    )
    apply_mods(ShowdownConfig.pokemon_mode)

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)

    ps_websocket_client = await PSWebsocketClient.create(
        ShowdownConfig.username,
        ShowdownConfig.password,
        ShowdownConfig.websocket_uri
    )
    await ps_websocket_client.login()

    battles_run = 0
    wins = 0
    losses = 0
    i=0
    evolution = Evolution()
    while True:
        genome = evolution.genomes[i]
        for _ in range(int(env("RUNS_FOR_GENOME"))):
            if ShowdownConfig.log_to_file:
                ShowdownConfig.log_handler.do_rollover(datetime.now().strftime("%Y-%m-%dT%H:%M:%S.log"))
            team = load_team(ShowdownConfig.team)
            if ShowdownConfig.bot_mode == constants.CHALLENGE_USER:
                await ps_websocket_client.challenge_user(
                    ShowdownConfig.user_to_challenge,
                    ShowdownConfig.pokemon_mode,
                    team
                )
            elif ShowdownConfig.bot_mode == constants.ACCEPT_CHALLENGE:
                await ps_websocket_client.accept_challenge(
                    ShowdownConfig.pokemon_mode,
                    team,
                    ShowdownConfig.room_name
                )
            elif ShowdownConfig.bot_mode == constants.SEARCH_LADDER:
                await ps_websocket_client.search_for_match(ShowdownConfig.pokemon_mode, team)
            else:
                raise ValueError("Invalid Bot Mode: {}".format(ShowdownConfig.bot_mode))

            winner = await pokemon_battle(ps_websocket_client, ShowdownConfig.pokemon_mode, genome=genome)
            battles_run += 1
            logger.info(f"Generation {evolution.generation}, genome {i+1} battle {battles_run} finished")
            
            if winner == ShowdownConfig.username:
                wins += 1
            else:
                losses += 1
            logger.info("W: {}\tL: {}".format(wins, losses))                

            check_dictionaries_are_unmodified(original_pokedex, original_move_json)

        if battles_run >= ShowdownConfig.run_count:
            break
            
        genome.score = wins / battles_run # compute its score
        wins = losses = 0
        # if the test for all genomes is done restart the search with the next generation
        logger.debug(f"genome {i}, poplation size {evolution.population_size}")
        i += 1
        if i >= evolution.population_size:
            evolution = evolution.next_generation()
            i = 0


if __name__ == "__main__":
    try:
        asyncio.run(showdown())
    except Exception as e:
        logger.error(traceback.format_exc())
        raise
