import sys

def read_number_of_elements() -> int:
    while True:
        match try_to_int(input('Insert number of elements (n): ')):
            case None:
                continue
            case integer:
                return integer


def try_to_int(string_to_int: str) -> int | None:
    try:
        return int(string_to_int)
    except ValueError as ex:
        print(f'Passed argument {string_to_int} is not int!')
        return None
