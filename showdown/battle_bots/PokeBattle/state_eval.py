from showdown.battle import Pokemon

# Weather conditions - type bonuses derived form https://bulbapedia.bulbagarden.net/wiki/Weather
CONDITIONS = {
    'sunnyday': {
        'fire': 20,
        'water': -20
    },
    'raindance': {
        'water': 20,
        'fire': -20
    },
    'sandstorm': {
        'rock': 20,
        'ground': 20,
        'steel': 20
    },
    'snow': {
        'ice': 20
    },
    'hail': {
        'ice': 20
    }
}


# State evaluation funcitons
def weather_condition(user_pokemon: Pokemon, opponent_pokemon: Pokemon, weather: str | None) -> float:
    """
    Evaluates the weather conditions effect on a pokemon and returns an advantage or disadvantage score
    """
    if weather is None:
        return 0

    condition = CONDITIONS.get(weather, {'NoneType': 0})
    score = 0

    for user_type, opponent_type in zip(user_pokemon.types, opponent_pokemon.types):
        score += condition.get(user_type, 0)
        score -= condition.get(opponent_type, 0)

    return score
