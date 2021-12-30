
def encode_color(color):
    match color:
        case "red":
            return 0
        case "yellow":
            return 1
        case "green":
            return 2
        case "blue":
            return 3
        case "white":
            return 4
def encode_value(value):
    return value-1
        