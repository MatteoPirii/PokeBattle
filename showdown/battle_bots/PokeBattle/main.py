import random

from PokeBattle import constants
from PokeBattle.showdown.battle import Battle
from copy import deepcopy
import json
import time

from PokeBattle.showdown.battle_bots.helpers import format_decision

# Percorso al file JSON con tutte le mosse
with open('data/moves.json', 'r') as f:
    all_move_json = json.load(f)


class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    def find_best_move(self):
        """Trova la miglior mossa o il miglior switch usando Minimax con formattazione delle decisioni."""
        best_move = None
        max_value = float('-inf')

        # Verifica che il Pokémon attivo sia valido
        if not self.user.active:
            print("Errore: Pokémon attivo non valido o esausto.")
            # Restituisci il primo switch disponibile
            switches = [f"switch {pokemon.name}" for pokemon in self.user.get_switches()]
            if switches:
                selected_switch = format_decision(self, switches[0])
                print(f"Switch selezionato: {selected_switch}")
                return selected_switch
            return ["no valid move or switch"]

        # Otteniamo tutte le mosse e gli switch disponibili
        user_options, _ = self.get_all_options()
        print(f"Mosse disponibili: {user_options}")

        start_time = time.time()  # Inizio del timer
        time_limit = 120  # Limite di tempo di 2 minuti

        moves = []
        switches = []

        # Separa le mosse dagli switch
        for option in user_options:
            if option.startswith(constants.SWITCH_STRING + " "):
                switches.append(option)
            else:
                moves.append(option)

        # Se siamo obbligati a cambiare o non ci sono mosse, restituisci il primo switch
        if self.force_switch or not moves:
            if not switches:
                print("Errore: Nessuno switch disponibile.")
                return ["no valid move or switch"]

            selected_switch = format_decision(self, switches[0])
            print(f"Switch selezionato: {selected_switch}")
            return selected_switch

        # Eseguiamo il MinMax su tutte le opzioni possibili (sia mosse che cambi)
        for move in moves:
            if time.time() - start_time > time_limit:
                print("Tempo scaduto, restituisco la miglior mossa trovata finora.")
                break

            saved_state = deepcopy(self)  # Salviamo lo stato della battaglia
            self.apply_move(move)  # Simuliamo l'effetto della nostra scelta
            move_value = self.minimax(depth=3, is_maximizing=True, alpha=float('-inf'), beta=float('inf'))

            self.restore_state(saved_state)  # Ripristiniamo lo stato di battaglia

            if move_value > max_value:
                max_value = move_value
                best_move = move

        # Seleziona la mossa con il danno massimo e restituisci la decisione formattata
        if best_move:
            selected_move = format_decision(self, best_move)
            print(f"Miglior mossa trovata: {selected_move}")
            return selected_move

        # Se non c'è una mossa valida, restituisci il primo switch
        selected_switch = format_decision(self, switches[0])
        print(f"Switch selezionato: {selected_switch}")
        return selected_switch

    def apply_move(self, move):
        """Applica la mossa o il cambio simulato tenendo conto dell'efficacia dei tipi."""
        move_part = move.split()

        if constants.SWITCH_STRING in move and len(move_part) > 1:
            self.user.active = self.get_pokemon_by_name(move.split()[1])  # Simula lo switch
        else:
            # Simuliamo il danno, gli effetti di stato o le modifiche statistiche di una mossa
            move_data = all_move_json.get(move.lower(),
                                          None)  # Convertiamo il nome della mossa in minuscolo per uniformità

            if move_data:
                # Calcoliamo il successo della mossa in base all'accuracy
                accuracy = move_data.get('accuracy', 100)
                if random.randint(1, 100) <= accuracy:
                    print(f"{self.user.active.name} usa {move_data['name']}")

                    # Calcoliamo il danno tenendo conto dell'efficacia del tipo
                    type_multiplier = calculate_type_multiplier(move_data['type'], self.opponent.active.types)

                    damage = calculate_damage(self.user.active, self.opponent.active, move_data)
                    damage *= type_multiplier  # Moltiplica il danno per il moltiplicatore di tipo

                    self.opponent.active.hp -= damage
                    print(
                        f"La mossa ha inflitto {damage} danni a {self.opponent.active.name} con un moltiplicatore di tipo {type_multiplier}.")

                    # Se la mossa ha effetti secondari (come status), applicali
                    if move_data.get('status'):
                        self.opponent.active.status = move_data['status']
                        print(f"{self.opponent.active.name} è stato {move_data['status']}!")
                else:
                    print(f"{self.user.active.name} ha fallito la mossa {move_data['name']}.")
            else:
                print(f"Errore: La mossa {move} non è stata trovata.")

    def restore_state(self, saved_state):
        """Ripristina lo stato della battaglia dopo una mossa simulata."""
        self.__dict__.update(saved_state.__dict__)  # Ripristiniamo lo stato della battaglia

    def minimax(self, depth, is_maximizing, alpha, beta):
        """Algoritmo Minimax con potatura Alpha-Beta."""
        # Condizioni di terminazione: profondità raggiunta o partita finita
        if depth == 0 or self.game_over():
            return self.evaluate_state()

        # Verifica se il Pokémon attivo è valido
        if self.user.active is None or not hasattr(self.user.active, 'hp'):
            print("Errore: Nessun Pokémon attivo o Pokémon esausto.")
            # Forziamo uno switch se non abbiamo un Pokémon attivo valido
            switch = self.find_best_switch()
            if switch:
                return self.apply_move(f"switch {switch}")
            return float('-inf')  # Se non c'è un'opzione valida, restituiamo una valutazione bassa

        ineffective_moves = True  # Traccia se tutte le mosse sono inefficaci

        if is_maximizing:
            max_eval = float('-inf')
            user_options, _ = self.get_all_options()

            for move in user_options:
                saved_state = deepcopy(self)  # Salviamo lo stato prima della mossa
                self.apply_move(move)

                # Controlliamo l'efficacia della mossa
                move_type = all_move_json.get(move.lower(), {}).get('type', None)
                if move_type:
                    type_multiplier = calculate_type_multiplier(move_type, self.opponent.active.types)
                    print(type_multiplier)
                    if type_multiplier > 1:
                        ineffective_moves = False  # Abbiamo trovato una mossa efficace

                eval = self.minimax(depth - 1, False, alpha, beta)  # Ora è il turno dell'avversario
                self.restore_state(saved_state)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-Beta pruning

            # Se tutte le mosse sono inefficaci o neutre, esegui lo switch
            if ineffective_moves and depth == 3:
                print("Tutte le mosse sono inefficaci, cerco uno switch migliore.")
                switch = self.find_best_switch()
                if switch:
                    print(f"Effettuo lo switch con {switch}.")
                    return self.apply_move(f"switch {switch}")

            return max_eval

        else:
            min_eval = float('inf')
            _, opponent_options = self.get_all_options()

            for move in opponent_options:
                saved_state = deepcopy(self)  # Salviamo lo stato prima della mossa
                self.apply_move(move)
                eval = self.minimax(depth - 1, True, alpha, beta)  # Torna al turno del bot, massimizza
                self.restore_state(saved_state)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha-Beta pruning
            return min_eval

    def evaluate_state(self):
        """Valuta lo stato corrente della battaglia"""
        score = 0

        # Controlla se il Pokémon attivo dell'utente è valido prima di accedere agli HP
        if not (self.user.active and hasattr(self.user.active, 'hp')):
            print("Errore: Pokémon attivo dell'utente non valido. Restituisco lo score corrente.")
            return score

        # Controlla se il Pokémon attivo dell'avversario è valido prima di accedere agli HP
        if not (self.opponent.active and hasattr(self.opponent.active, 'hp')):
            print("Errore: Pokémon attivo dell'avversario non valido. Restituisco lo score corrente.")
            return score

        # Calcolo del punteggio basato sugli HP dei Pokémon attivi
        score += self.user.active.hp
        score -= self.opponent.active.hp

        # Aggiungi un bonus per ogni Pokémon di riserva ancora vivo
        score += len([pokemon for pokemon in self.user.reserve if pokemon.hp > 0])
        score -= len([pokemon for pokemon in self.opponent.reserve if pokemon.hp > 0])

        # Penalità se il Pokémon dell'utente è in svantaggio di tipo
        if is_type_disadvantageous(self.user.active, self.opponent.active):
            score -= 50  # Penalità per lo svantaggio di tipo

        return score

    def find_best_switch(self):
        """Trova il miglior pokemon nel team per effettuare lo switch"""
        best_switch = None
        max_resistence = float('-inf')

        for switch in self.user.get_switches():
            pokemon_to_switch = self.get_pokemon_by_name(switch)

            if pokemon_to_switch is None:
                print(f"Errore: Non trovato il Pokémon per lo switch {switch}")
                continue  # Salta questo ciclo se il Pokémon non è valido

            # Valuta la resistenza dei Pokémon di riserva contro il tipo dell'avversario
            resistence = 0
            for opponent_pokemon_type in self.opponent.active.types:
                for pokemon_to_switch_type in pokemon_to_switch.types:
                    resistence += constants.TYPE_EFFECTIVENESS[opponent_pokemon_type][pokemon_to_switch_type]

            if resistence > max_resistence:
                max_resistence = resistence
                best_switch = switch

        return best_switch

    def get_pokemon_by_name(self, name):
        """Restituisce il Pokémon con il nome fornito dalla riserva del giocatore."""
        for pokemon in self.user.reserve:
            if pokemon.name.lower() == name.lower():
                return pokemon
        return None  # Se non trovato

    def game_over(self):
        """Verifica se la battaglia è terminata."""
        # Verifica se il Pokémon attivo è valido
        user_pokemon_alive = False
        if self.user.active and hasattr(self.user.active, 'hp'):
            user_pokemon_alive = self.user.active.hp > 0

        # Verifica se ci sono Pokémon di riserva
        reserve_pokemon_alive = any(pokemon.hp > 0 for pokemon in self.user.reserve)

        # Verifica lo stato dell'avversario
        opponent_pokemon_alive = False
        if self.opponent.active and hasattr(self.opponent.active, 'hp'):
            opponent_pokemon_alive = self.opponent.active.hp > 0

        reserve_opponent_alive = any(pokemon.hp > 0 for pokemon in self.opponent.reserve)

        # Se nessuno dei Pokémon dell'utente o dell'avversario è vivo, la battaglia è finita
        return not (user_pokemon_alive or reserve_pokemon_alive) or not (
                    opponent_pokemon_alive or reserve_opponent_alive)


def is_type_disadvantageous(user_pokemon, opponent_pokemon):
    """Controlla se il tipo del Pokémon usato dal bot è adatto allo scontro contro i due tipi avversari"""
    user_pokemon_types = user_pokemon.types
    opponent_pokemon_types = opponent_pokemon.types

    # Calcoliamo se i tipi del nostro Pokémon sono svantaggiosi rispetto ai due tipi dell'avversario
    total_disadvantage = 1
    for user_pokemon_type in user_pokemon_types:
        for opponent_pokemon_type in opponent_pokemon_types:
            disadvantage = constants.TYPE_EFFECTIVENESS[opponent_pokemon_type][user_pokemon_type]
            total_disadvantage *= disadvantage  # Moltiplichiamo il risultato su entrambi i tipi

    return total_disadvantage < 1  # Se il moltiplicatore totale è inferiore a 1, è svantaggioso


def calculate_type_multiplier(move_type, defender_types):
    """Calcola il moltiplicatore di danno tenendo conto di entrambi i tipi del difensore"""
    multiplier = 1.0

    for defender_type in defender_types:
        multiplier *= constants.TYPE_EFFECTIVENESS[move_type][
            defender_type]  # Calcola il moltiplicatore per ogni tipo del difensore

    return multiplier


def calculate_damage(attacker, defender, move_data):
    """Calcola il danno inflitto da una determinata mossa"""

    # Otteniamo le statistiche di attacco e di difesa e le moltiplichiamo per il power della mossa
    attack_stat = attacker.stats[constants.ATTACK] if move_data['category'] == 'physical' else attacker.stats[
        constants.SPECIAL_ATTACK]
    defense_stat = defender.stats[constants.DEFENSE] if move_data['category'] == 'physical' else defender.stats[
        constants.SPECIAL_DEFENSE]

    # Calcolo del danno base (calcolo del gioco non mio)
    damage = ((2 * attacker.level / 5 + 2) * move_data['basePower'] * (attack_stat / defense_stat)) / 50 + 2

    # Viene aggiunto un moltiplicatore di danno per il tipo
    type_multiplier = calculate_type_multiplier(move_data['type'], defender.types)
    return damage * type_multiplier
