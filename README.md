# Projet de Mathématiques — Minimisation sous une contrainte d'égalité : Optimisation numérique

- **Groupe :** Quang Minh DO, Arin BILGIN, Liam GODARD
- **Encadrant :** Adel HAMDI

## Contenu de l'archive

```
prj-math/
├── main.py
├── methode1.py
├── methode2.py
├── methode3.py
├── methode4.py
├── input.py
├── input.txt
├── log.py
├── probleme.py
└── README.md
```

## Exécutables disponibles

| Fichier | Description |
|---|---|
| `main.py` | Lance les quatre méthodes depuis le même `X0` et affiche un tableau de comparaison |
| `methode1.py`, `methode2.py`, `methode3.py`, `methode4.py` | Lancent chacun une méthode (gradient à pas fixe, gradient à pas optimal, Newton, quasi-Newton BFGS) et affichent les données et les résultats finaux. |
| `input.py` | Affiche `X0` et `alpha` lus dans `input.txt` |

## Dépendances

| Dépendance | Version |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.5.3 |

Pour installer NumPy, lancer :

```
pip install numpy
```

## Exécution

Dans un terminal, il faut se placer dans le dossier qui contient les fichiers, puis lancer :

```
python main.py
```

Pour exécuter une seule méthode, par exemple la méthode 4, lancer :

```
python methode4.py
```

Selon l'installation, la commande peut être `python3` à la place de `python`.

## Fichier d'entrée

Pour modifier les paramètres (`X0` et `alpha`), ouvrez le fichier `input.txt` et remplacez les valeurs. Si une donnée est manquante ou invalide, le programme s'arrête et affiche un message d'erreur.

## Fichier de sortie

Quand on lance un fichier `methodeN.py`, avec `N` le nombre de méthode, le module `log.py` enregistre toutes les itérations dans un fichier CSV `resultat_methodeN.csv` pour facilier la création des graphiques.