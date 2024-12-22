# PokéBattle
A Pokémon battle-bot that can play battles on [Pokemon Showdown](https://pokemonshowdown.com/) thanks to the API [Showdown](https://github.com/pmariglia/showdown).

<p align = "center">
  <img src = "Stemma_unipi.svg.png" width="256" height="256">
</p>

<p align = "center">
  Computer Science Department
  <br>
  A project for
  <br>
  Artificial Intelligence Foundaments
  <br>
  courses of AI Master at University of Pisa.
</p>

# Project description and introduction

In this section we introduce context information for the project.

## Introduction


## Authors
* **Matteo Piredda**      - *Developer*         - [MatteoPirii](https://github.com/MatteoPirii)
* **Margherita Merialdo** - *Developer*         - [margheritamerialdo](https://github.com/margheritamerialdo)
* **Michela Faella**      - *Developer*         - [MichelaFaella](https://github.com/MichelaFaella)
* **Udit Gagnani**        - *Developer*         - [UditKGagnani](https://github.com/UditKGagnani)

## Python version
Developed and tested using Python 3.12.7

## Getting Started

### Configuration
Environment variables are used for configuration.
You may either set these in your environment before running,
or populate them in the [env](./env) file.

The configurations available are:

| Config Name | Type | Required | Description |
|---|:---:|:---:|---|
| **`BATTLE_BOT`** | string | yes | The BattleBot module to use. More on this below in the Battle Bots section |
| **`WEBSOCKET_URI`** | string | yes | The address to use to connect to the Pokemon Showdown websocket |
| **`PS_USERNAME`** | string | yes | Pokemon Showdown username |
| **`PS_PASSWORD`** | string | yes | Pokemon Showdown password  |
| **`BOT_MODE`** | string | yes | The mode the the bot will operate in. Options are `CHALLENGE_USER`, `SEARCH_LADDER`, or `ACCEPT_CHALLENGE` |
| **`POKEMON_MODE`** | string | yes | The type of game this bot will play: `gen8ou`, `gen7randombattle`, etc. |
| **`USER_TO_CHALLENGE`** | string | only if `BOT_MODE` is `CHALLENGE_USER` | If `BOT_MODE` is `CHALLENGE_USER`, this is the name of the user you want your bot to challenge |
| **`RUN_COUNT`** | int | no | The number of games the bot will play before quitting |
| **`TEAM_NAME`** | string | no | The name of the file that contains the team you want to use. More on this below in the Specifying Teams section. |
| **`ROOM_NAME`** | string | no | If `BOT_MODE` is `ACCEPT_CHALLENGE`, the bot will join this chatroom while waiting for a challenge. |
| **`SAVE_REPLAY`** | boolean | no | Specifies whether or not to save replays of the battles (`True` / `False`) |
| **`LOG_LEVEL`** | string | no | The Python logging level (`DEBUG`, `INFO`, etc.) |
Genetic algorithm
| **`RUNS_FOR_GENOME`** | int | yes | Number of matches used to score a genome
| **`POPULATION_SIZE`** | int | yes | Population size for the genetic search algorithm
| **`MUTATION_RATE`** | float | yes | Chance of a random mutation to occur during the recombination process

### Running without Docker

**1. Clone**

Clone the repository with `git clone https://github.com/MatteoPiri/PokeBattle.git`

**2. Install Requirements**

Install the requirements with `pip install -r requirements.txt`.

**3. Configure your [env](./env) file**

This is our current env:
```
BATTLE_BOT=PokeBattle
WEBSOCKET_URI=wss://sim3.psim.us/showdown/websocket
PS_USERNAME=AIFPokeBattleTRAIN
PS_PASSWORD=(S*aE74zP53b
BOT_MODE=SEARCH_LADDER
POKEMON_MODE=gen7randombattle
RUN_COUNT=1
```
But you can decide to play in any configuration you want.

**4. Run**

Run with `python run.py`

### Running with Docker
Requires Docker and the Docker Comopose plugin.

The project has been tested on *docker 27.4.1* with *docker compose 2.3.1*

1. **Clone the repository**

   `git clone https://github.com/MatteoPiri/PokeBattle.git`

1. **Configure your [env](./env) file**

   Configure the .env file following the table above or use our own

1. **Run docker**
   
   `docker compose up` will start the project
   
>Connect to [PokemonShowdown](https://play.pokemonshowdown.com/) and log-in with the same credentials in the .env file to see the battle in real-time

## Battle Bot  

The original Pokémon Showdown API provides various battle bots with diverse strategies to choose from. However, we have 
designed and developed our custom bot, **PokéBattle**, to deliver a smarter and more competitive gameplay experience. 
You can explore the source code of **PokéBattle** [here](https://github.com/MatteoPirii/PokeBattle/tree/master/showdown/battle_bots/PokeBattle).  

Our **PokéBattle** bot leverages advanced artificial intelligence techniques, specifically implementing the **Minimax 
algorithm with Alpha-Beta Pruning**. This strategy allows the bot to efficiently evaluate all possible game scenarios up
to a certain depth, optimizing decision-making by pruning less promising branches of the game tree.  

### **Key Features of PokéBattle Bot**  
1. **Optimized Decision-Making**:  
   The bot evaluates moves by considering all possible opponent responses, ensuring decisions are not only offensive but also strategically defensive.  

2. **Performance Efficiency**:  
   By using Alpha-Beta pruning, the bot eliminates redundant computations, significantly reducing the time required to make high-quality decisions.  

3. **Adaptability**:  
   It dynamically adjusts its strategy based on the game state, prioritizing moves that maximize its chances of winning while mitigating potential losses.  

4. **Enhanced Gameplay**:  
   The bot's intelligence creates a challenging opponent for both human players and other bots, raising the bar for competitive battles.

### **Why Minimax with Alpha-Beta Pruning?**  
The **Minimax algorithm** is a foundational method in game theory, used to determine the optimal move in zero-sum games. By incorporating **Alpha-Beta Pruning**, we drastically enhance the efficiency of this algorithm, allowing the bot to:  
- Analyze deeper into the game tree within the same computational limits.  
- Focus on the most promising sequences of moves while discarding suboptimal ones early.  

This combination ensures that **PokéBattle** operates at a level comparable to professional human players, making it a formidable adversary in Pokémon battles.

