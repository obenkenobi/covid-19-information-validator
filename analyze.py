import json, os

def covid_valid_freq(data):
    res = {}
    for i in range(3):
        res[i] = len([data for entry in data if entry['valid'] == i])
    return res

def json_covid_valid_freq(filepath = None):
    if filepath == None:
        filepath = os.path.join(os.path.dirname(__file__), 'text_data.json')
    with open(filepath) as f:
        data = json.load(f)
    return covid_valid_freq(data)