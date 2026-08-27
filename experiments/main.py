import json

from graph.extract import extract


def run():
    user_input = input("Enter message: ")
    result = extract(user_input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
