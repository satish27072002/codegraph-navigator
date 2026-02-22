import os
from pkg.util import helper


def greet(name: str) -> str:
    print(helper(name))
    return os.path.basename(name)


class Greeter:
    def run(self, value: str) -> str:
        return greet(value)


if __name__ == "__main__":
    greet("demo")
