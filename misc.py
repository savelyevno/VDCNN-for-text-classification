def Print(str, *args):
    print(str.format(*args))


def print_log(to_log, str, *args):
    if to_log:
        Print(str, *args)
