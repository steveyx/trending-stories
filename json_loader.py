import json


class JsonLoader:

    @staticmethod
    def save_json(data, filename):
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=2)

    @staticmethod
    def load_json(filename):
        with open(filename) as data_file:
            data = json.load(data_file)
        return data
