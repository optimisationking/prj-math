import os
import sys 

INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
KEYS = ["r", "h", "lambda", "alpha"]

def read_input():
    values = {}
    try:
        with open(INPUT_FILE, encoding="utf-8") as file:
            for line in file:
                if "=" not in line:
                    continue 

                name, text = line.split("=", 1)
                name = name.strip()
                if name not in KEYS:
                    continue 

                try:
                    values[name] = float(text.strip().replace(",", "."))
                except ValueError:
                    sys.exit(f"Erreur dans {INPUT_FILE} : la valeur de '{name}' doit être un nombre.")
    except FileNotFoundError:
        sys.exit(f"Erreur : le fichier {INPUT_FILE} est introuvable.")

    for name in KEYS:
        if name not in values:
            sys.exit(f"Erreur dans {INPUT_FILE} : la valeur de '{name}' est manquante.")
    if values["r"] <= 0 or values["h"] <= 0 or values["alpha"] <= 0:
        sys.exit(f"Erreur in {INPUT_FILE}: r, h et alpha doit être strictement positifs.")

    X0 = [values["r"], values["h"], values["lambda"]]
    return X0, values["alpha"]

if __name__ == "__main__":
    X0, alpha = read_input()
    print("X0 = ", X0)
    print("alpha = ", alpha)