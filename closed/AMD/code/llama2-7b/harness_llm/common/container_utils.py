def remove_value_from_dict(dictionary: dict, value):
    return {
        k: v for k, v in dictionary.items() if v != value
    }

def remove_none_from_dict(dictionary: dict):
    return remove_value_from_dict(dictionary, None)