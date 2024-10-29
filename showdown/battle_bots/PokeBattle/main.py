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
        """Find the best move to choose using the MinMax approach."""
        best_move = None
        max_value = float('-inf')

        # Preliminary check: we change the Pokémon if the type is disadvantageous
        if is_type_disadvantageous(self.user.active, self.opponent.active):
            best_switch = self.find_best_switch()
            if best_switch:
                print(f"Suggerito cambio per vantaggio di tipo: {best_switch}")
                return format_decision(self, f"switch {best_switch}")

        # Check if the Pokémon is alive or inactive
        if not self.user.active:
            print("Errore: Pokémon attivo non valido o esausto.")
            # Restituisci il primo switch disponibile
            switches = [f"switch {pokemon.name}" for pokemon in self.user.get_switches()]
            if switches:
                selected_switch = self.find_best_switch()
                print(f"Switch selezionato: {selected_switch}")
                return selected_switch
            return ["no valid move or switch"]

        user_options, _ = self.get_all_options()
        print(f"Mosse disponibili: {user_options}")

        start_time = time.time()  # Timer
        time_limit = 140  # Two minutes to choose the best moves

        moves, switches = [], []

        # Separa le mosse dagli switch
        for option in user_options:
            if option.startswith(constants.SWITCH_STRING + " "):
                switches.append(option)
            else:
                moves.append(option)

        # Se siamo obbligati a cambiare o non ci sono mosse, restituisci il primo switch
        if self.force_switch or not moves:
            if switches:
                selected_switch = format_decision(self, switches[0])
                print(f"Switch selezionato: {selected_switch}")
                return selected_switch
            return ["no valid move or switch"]

        # Eseguiamo il MinMax su tutte le opzioni possibili (sia mosse che cambi)
        for move in moves:
            if time.time() - start_time > time_limit:
                print("Tempo scaduto, restituisco la miglior mossa trovata finora.")
                break

            saved_state = deepcopy(self)  # Salviamo lo stato della battaglia
            self.apply_move(move)  # Simuliamo l'effetto della nostra scelta
            move_value = self.minmax(depth=5, is_maximizing=True, alpha=float('-inf'), beta=float('inf'))

            self.restore_state(saved_state)  # Ripristiniamo lo stato di battaglia

            if move_value > max_value:
                max_value = move_value
                best_move = move

        # Seleziona la mossa migliore e restituisci la decisione formattata
        if best_move:
            selected_move = format_decision(self, best_move)
            print(f"Miglior mossa trovata: {selected_move}")
            return selected_move

        # Se non c'è una mossa valida, restituisci il miglior switch
        selected_switch = self.find_best_switch()
        print(f"Switch selezionato: {selected_switch}")
        return selected_switch

    def apply_move(self, move):
        """Applica la mossa o il cambio simulato tenendo conto dell'efficacia dei tipi."""
        move_part = move.split()

        # Controlliamo se la mossa è uno switch
        if constants.SWITCH_STRING in move and len(move_part) > 1:
            target_pokemon = self.get_pokemon_by_name(move.split()[1])
            if target_pokemon:
                self.user.active = target_pokemon  # Simula lo switch
                print(f"Effettuato switch a {self.user.active.name}")
            else:
                print(f"Errore: Pokémon {move.split()[1]} non trovato.")
            return

        # Recuperiamo dei dati della mossa
        move_data = all_move_json.get(move.lower())
        if not move_data:
            print(f"Errore: La mossa {move} non è stata trovata.")
            return

        # Calcoliamo dell'accuracy e controllo se la mossa va a segno
        accuracy = move_data.get('accuracy', 100)
        if random.randint(1, 100) > accuracy:
            print(f"{self.user.active.name} ha fallito la mossa {move_data['name']}.")
            return

        print(f"{self.user.active.name} usa {move_data['name']}")

        # Calcolo del moltiplicatore di tipo e del danno
        type_multiplier = calculate_type_multiplier(move_data['type'], self.opponent.active.types)
        damage = calculate_damage(self.user.active, self.opponent.active, move_data) * type_multiplier

        # Consideriamo anche il livello avversario
        damage *= (self.user.active.level / self.opponent.active.level)

        # Applica il danno ai punti vita dell'avversario
        self.opponent.active.hp -= damage
        print(
            f"La mossa ha inflitto {damage:.2f} danni a {self.opponent.active.name} con un moltiplicatore di tipo "
            f"{type_multiplier}.")

        # Applicazione degli effetti di stato (se presenti)
        status_effect = move_data.get('status')
        if status_effect:
            self.opponent.active.status = status_effect
            print(f"{self.opponent.active.name} è stato {status_effect}!")

    def restore_state(self, saved_state):
        """Ripristina lo stato della battaglia dopo una mossa simulata."""
        self.__dict__.update(saved_state.__dict__)  # Ripristiniamo lo stato della battaglia

    def minmax(self, depth, is_maximizing, alpha, beta):
        """Algoritmo Minimax con potatura Alpha-Beta."""
        # Condizioni di terminazione: profondità raggiunta o partita finita
        if depth == 0 or self.game_over():
            return self.evaluate_state()

        # Verifica se il Pokémon attivo è valido
        if self.user.active is None or not hasattr(self.user.active, 'hp'):
            print("Errore: Nessun Pokémon attivo o Pokémon esausto.")
            # Forziamo uno switch se non abbiamo un Pokémon attivo valido
            switch = self.find_best_switch()
            return self.apply_move(f"switch {switch}") if switch else float('-inf')  # Se non c'è un'opzione valida,
            # restituiamo una valutazione bassa

        if is_maximizing:
            max_eval = float('-inf')
            user_options, _ = self.get_all_options()

            for move in user_options:
                saved_state = deepcopy(self)  # Salviamo lo stato prima della mossa
                self.apply_move(move)

                eval = self.minmax(depth - 1, False, alpha, beta)  # Ora è il turno dell'avversario
                self.restore_state(saved_state)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-Beta pruning

            return max_eval

        else:
            min_eval = float('inf')
            _, opponent_options = self.get_all_options()

            print("DEBUG: ", opponent_options)

            for move in opponent_options:
                saved_state = deepcopy(self)  # Salviamo lo stato prima della mossa
                self.apply_move(move)

                eval = self.minmax(depth - 1, True, alpha, beta)  # Torna al turno del bot, massimizza
                self.restore_state(saved_state)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha-Beta pruning
            return min_eval

    def evaluate_state(self):
        """Valuta lo stato corrente della battaglia"""
        score = 0

        # Controlliamo se il Pokémon attivo dell'utente è valido prima di accedere agli HP
        if not (self.user.active and hasattr(self.user.active, 'hp')):
            print("Errore: Pokémon attivo dell'utente non valido. Restituisco lo score corrente.")
            return score

        # Controlliamo se il Pokémon attivo dell'avversario è valido prima di accedere agli HP
        if not (self.opponent.active and hasattr(self.opponent.active, 'hp')):
            print("Errore: Pokémon attivo dell'avversario non valido. Restituisco lo score corrente.")
            return score

        if hasattr(self.user, 'consecutive_switches') and self.user.consecutive_switches > 1:
            score -= self.user.consecutive_switches * 10

        # 1. Punteggio per differenza di hp e livello
        score += (self.user.active.hp - self.opponent.active.hp) + (
                self.user.active.level - self.opponent.active.level) * 10

        # 2. Aggiungiamo un bonus/penalità per le riserve ancora vive
        score += sum(10 for p in self.user.reserve if p.hp > 0)
        score -= sum(10 for p in self.opponent.reserve if p.hp > 0)

        # 3. Aggiungiamo un bonus/penalità in base all'efficacia dei tipi
        if is_type_disadvantageous(self.user.active, self.opponent.active):
            score -= -40
        else:
            score += 50

        # 4. Aggiungiamo un bonus in base alle condizioni meteorologiche
        if self.weather == 'sunnyday' and 'Fire' in self.user.active.types:
            score += 10
        elif self.weather == 'raindance' and 'Water' in self.user.active.types:
            score += 10

        # 5. Penalità per le condizioni di stato sfavorevoli
        if self.user.active.status == 'paralyzed':
            score -= 20
        elif self.user.active.status == 'poisoned':
            score -= 15
        elif self.user.active.status == 'badly poisoned':
            score -= 25
        elif self.user.active.status == 'burned':
            score -= 20
        elif self.user.active.status == 'asleep':
            score -= 30
        elif self.user.active.status == 'frozen':
            score -= 40

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
                print(best_switch)

        return best_switch

    def get_pokemon_by_name(self, name):
        """Restituisce il Pokémon con il nome fornito dalla riserva del giocatore."""
        # Rimuove il prefisso "switch" se presente
        if name.startswith("switch "):
            name = name.split(" ", 1)[1]  # Prende solo il nome del Pokémon

        normalized_name = name.lower().strip()  # Rende il nome uniforme
        for pokemon in self.user.reserve:
            if pokemon.name.lower() == normalized_name:
                return pokemon
        print(f"Errore: Pokémon {name} non trovato nella riserva.")
        return None

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
    """Controlla se il tipo del Pokémon usato dal bot è adatto allo scontro contro il tipo avversari"""
    user_pokemon_types = user_pokemon.types
    opponent_pokemon_types = opponent_pokemon.types

    # Valutazioni per il vantaggio e lo svantaggio complessivo
    advantage_count = 0
    disadvantage_count = 0
    total_multiplier = 1  # Variabile che tiene conto del moltiplicatore totale

    for user_type in user_pokemon_types:
        for opponent_type in opponent_pokemon_types:

            # Recuperiamo il moltiplicatore di efficacia del tipo
            multiplier = constants.TYPE_EFFECTIVENESS[opponent_type].get(user_type, 1)

            # Controlliamo l'efficacia del tipo
            if multiplier >= 1:
                advantage_count += 1
            elif multiplier < 1:
                disadvantage_count += 1

            # Aggiorniamo il moltiplicatore totale
            total_multiplier *= multiplier

    # Se ci sono più svantaggi o il moltplicatore totale è minore di uno, allora è svantaggioso
    if disadvantage_count > advantage_count or total_multiplier < 1:
        return True

    # Altrimenti non è svantaggioso
    return False


def calculate_type_multiplier(move_type, defender_types):
    """Calcola il moltiplicatore di danno tenendo conto di entrambi i tipi del difensore"""
    multiplier = 1.0

    for defender_type in defender_types:
        multiplier *= constants.TYPE_EFFECTIVENESS[move_type][
            defender_type]  # Calcola il moltiplicatore per ogni tipo del difensore

    return multiplier


def calculate_damage(attacker, defender, move_data):
    """Calcola il danno inflitto da una determinata mossa"""

    level = attacker.level

    # Otteniamo le statistiche di attacco e di difesa e le moltiplichiamo per il power della mossa
    attack_stat = attacker.stats[constants.ATTACK] if move_data['category'] == 'physical' else attacker.stats[
        constants.SPECIAL_ATTACK]
    defense_stat = defender.stats[constants.DEFENSE] if move_data['category'] == 'physical' else defender.stats[
        constants.SPECIAL_DEFENSE]

    # Calcolo del danno base (calcolo del gioco non mio)
    damage = (((2 * level / 5 + 2) * move_data['basePower'] * (attack_stat / defense_stat)) / 50 + 2)

    # Applichiamo Same-Type Attack Bonus (STAB)
    if move_data['type'] in attacker.types:
        damage *= 1.5

    # Viene aggiunto un moltiplicatore di danno per il tipo
    type_multiplier = calculate_type_multiplier(move_data['type'], defender.types)
    damage *= type_multiplier

    # Applichiamo un random variance
    damage *= random.uniform(0.85, 1.0)

    # Modifichiamo il damage tenendo conto del livello
    return int(damage)
