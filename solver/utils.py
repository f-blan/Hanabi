
def encode_color(color):
    if color == "red":
        return 0
    elif color == "yellow":
        return 1
    elif color == "green":
        return 2
    elif color == "blue":
        return 3
    elif color == "white":
            return 4
    else:
        print(f"color corresponding to {color} was not found")
def encode_value(value):
    assert value<6
    assert value >0
    return value-1

def decode_value(value):
    assert value>-1
    assert value <5
    return value+1

def decode_color(color):

    if color == 0:
        return "red"
    elif color == 1:
        return "yellow"
    elif color == 2:
        return "green"
    elif color == 3:
        return "blue"
    elif color == 4:
        return "white"
    else:
        print(f"color corresponding to {color} was not found")
def decode_hint_value(value):
    if value == 0:
        return "value"
    else:
        return "color"
        