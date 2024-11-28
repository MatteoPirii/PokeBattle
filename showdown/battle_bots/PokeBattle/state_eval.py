from showdown.battle import Pokemon

# Weather conditions - type bonuses derived form https://bulbapedia.bulbagarden.net/wiki/Weather
CONDITIONS = {
    'sunnyday': {
        'Fire': 20,
        'Whater': -20
    },
    'raindance': {
        'Water': 20
    }
}


# State evaluation funcitons
def weather_condition(pokemon: Pokemon, weather: str | None) -> float:
    """
    Evaluates the weather conditions effect on a Pokémon and returns an advantage score
    """
    if weather is None:
        return 0

    condition = CONDITIONS.get(weather, {'NoneType': 0})
    score = 0

    for move in pokemon.moves:
        score += condition.get(move.type, 0)

    return score
