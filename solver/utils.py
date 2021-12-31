
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

def decode_value(value):
    return value+1

def decode_color(color):
    match color:
        case 0:
            return "red"
        case 1:
            return "yellow"
        case 2:
            return "green"
        case 3:
            return "blue"
        case 4:
            return "white"
        