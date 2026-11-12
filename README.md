# Projet de Mathématiques : Minimisation sous une contrainte d'égalité

- **Groupe :** Quang Minh DO, Arin BILGIN, Liam GODARD
- **Encadrant :** Adel HAMDI

## Contenu de l'archive

```
prj-math/
├── main.py
├── method1.py
├── method2.py
├── method3.py
├── method4.py
├── input.py
├── input.txt
├── problem.py
└── README.md
```

## Exécutables disponibles
| Fichier | Description |
|---|---|
| `main.py` | Lance les quatre méthodes depuis le même `X0` et affiche un tableau de comparaison |
| `method1.py`, `method2.py`, `method3.py`, `method4.py` | Lancent chacun une méthode (gradient à pas fixe, gradient à pas optimal, Newton, quasi-Newton BFGS) et affichent les données et les résultats. |
| `input.py` | Affiche `X0` et `alpha` lus dans `input.txt` |

## Dépendances

| Dépendance | Version |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.5.3 |

## Exécution
Pour lancer les quatres méthodes :
```
python main.py
```

Pour exécuter une seule méthode, par exemple la méthode 4 :
```
python method4.py
```

## Fichier d'entrée
Pour modifier les paramètres (`X0` et `alpha`), ouvrez le fichier `input.txt` et remplacez les valeurs. Si une donnée est manquante ou invalide, le programme s'arrête et affiche un message d'erreur.