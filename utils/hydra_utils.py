"""Class offering utilities for the hydra configuration manager"""


def prettyPrint(dict_in, message=None):
    """prints a nested dictionary in a more readable way"""
    if message is not None:
        print(message)
    print(tabAdder(dict_in))


def tabAdder(dict_in, tabs=0):
    out = []
    for key in dict_in:
        if isinstance(dict_in[key], dict):
            out.append(key + ":\n" + tabAdder(dict_in[key], tabs + 1))
        else:
            out.append(key + ": " + dict_in[key].__str__())

    tab_string = ["\t" for _ in range(tabs)]
    repeated_string = "\n" + "".join(tab_string)

    return repeated_string[1:] + repeated_string.join(out)


def unflatten_dict(config: dict):
    unflattened = {}
    for key, value in config.items():
        keys = key.split('.')
        current = unflattened
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return unflattened

