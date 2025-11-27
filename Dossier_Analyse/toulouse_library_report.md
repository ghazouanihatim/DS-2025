# COMPTE RENDU D'ANALYSE
## Dataset : Toulouse Public Library Loans (Médiathèque de Toulouse)

---

## 1. PRÉSENTATION DU DATASET

### 1.1 Description générale
Le dataset **"Toulouse Public Library Loans"** contient les statistiques annuelles de circulation et les métadonnées bibliographiques des films imprimés de la collection de la bibliothèque publique de Toulouse. Ce jeu de données est destiné à l'analyse des tendances de prêt, la gestion de collection, le comportement des usagers et la recherche sur l'utilisation des médias.

### 1.2 Source des données
- **Plateforme** : Kaggle
- **Nom du dataset** : `toulouse-public-library-mdiathque-dataset`
- **Auteur** : grimespoint
- **Format** : CSV (séparateur : point-virgule `;`)
- **Fichier** : `toulouse_public_library_loans.csv`

---

## 2. STRUCTURE DES DONNÉES

### 2.1 Colonnes du dataset

Le dataset contient **11 colonnes** principales :

| Colonne | Type | Description |
|---------|------|-------------|
| **year** | Float | Année d'enregistrement des prêts |
| **nb_loans** | Integer | Nombre de prêts pour l'item |
| **title** | String | Titre du film |
| **author** | String | Réalisateur/Auteur du film |
| **publisher** | String | Éditeur et lieu de publication |
| **classification** | String | Code de classification de l'item |
| **library** | String | Bibliothèque où l'item est disponible |
| **spine_label** | String | Étiquette de dos du support |
| **audience** | String | Public cible (ex: "A" pour adulte) |
| **media_subtype** | String | Sous-type de média (ex: DVDFIC) |
| **media_type** | String | Type de média (ex: films) |

### 2.2 Échantillon des données

```
Année  | Nb Prêts | Titre                          | Auteur              | Bibliothèque
-------|----------|--------------------------------|---------------------|-------------
2023   | 207      | Top Gun : Maverick             | Kosinski, Joseph    | CABANIS
2023   | 193      | Trois mille ans à t'attendre   | Miller, George      | CABANIS
2023   | 152      | Compétition officielle         | Cohn, Mariano       | CABANIS
2023   | 151      | Licorice pizza                 | Anderson, Paul T.   | CABANIS
2023   | 144      | Le discours                    | Tirard, Laurent     | CABANIS
```

---

## 3. CARACTÉRISTIQUES DU DATASET

### 3.1 Période couverte
- **Année principale** : 2023
- Le dataset semble se concentrer sur les statistiques annuelles de l'année 2023

### 3.2 Bibliothèques représentées
D'après l'échantillon, au moins la bibliothèque **CABANIS** est représentée dans le dataset.

### 3.3 Types de médias
- **media_type** : films
- **media_subtype** : DVDFIC (DVD Fiction)
- **audience** : A (Adulte)

---

## 4. OBSERVATIONS PRÉLIMINAIRES

### 4.1 Films les plus empruntés (Top 5 - 2023)

1. **Top Gun : Maverick** - 207 prêts
   - Réalisateur : Joseph Kosinski
   - Genre d'action très populaire

2. **Trois mille ans à t'attendre** - 193 prêts
   - Réalisateur : George Miller
   - Film fantastique/romantique

3. **Compétition officielle** - 152 prêts
   - Réalisateur : Mariano Cohn
   - Comédie dramatique

4. **Licorice pizza** - 151 prêts
   - Réalisateur : Paul Thomas Anderson
   - Drame/Romance

5. **Le discours** - 144 prêts
   - Réalisateur : Laurent Tirard
   - Comédie française

### 4.2 Encodage des caractères
⚠️ **Problème détecté** : Le dataset présente des problèmes d'encodage (caractères accentués mal affichés, ex: "Ã " au lieu de "à"). 
- Solution appliquée : `encoding='utf-8'` ou `encoding='latin-1'` lors de la lecture du fichier

---

## 5. ANALYSES POSSIBLES

Ce dataset permet plusieurs types d'analyses :

### 5.1 Analyse des tendances de prêt
- Identifier les films les plus populaires
- Analyser la saisonnalité des emprunts
- Étudier l'évolution des prêts au fil du temps

### 5.2 Gestion de collection
- Optimiser l'acquisition de nouveaux titres
- Identifier les œuvres sous-utilisées
- Planifier le renouvellement des collections

### 5.3 Comportement des usagers
- Préférences de genre cinématographique
- Répartition géographique des emprunts par bibliothèque
- Analyse du public cible

### 5.4 Recherche sur l'utilisation des médias
- Comparaison entre différents types de supports
- Étude de la pertinence des DVD à l'ère du streaming
- Impact des sorties cinéma sur les emprunts

---

## 6. PISTES D'EXPLORATION DATA SCIENCE

### 6.1 Analyses descriptives
```python
- Distribution du nombre de prêts
- Films les plus empruntés par bibliothèque
- Analyse des réalisateurs les plus populaires
- Statistiques par éditeur
```

### 6.2 Visualisations recommandées
- **Graphiques en barres** : Top 10/20 des films les plus empruntés
- **Histogrammes** : Distribution des prêts
- **Cartes de chaleur** : Prêts par bibliothèque et par période
- **Nuages de mots** : Titres ou auteurs les plus fréquents

### 6.3 Analyses avancées possibles
- **Clustering** : Regrouper les films par patterns de prêts similaires
- **Analyse de séries temporelles** : Prédire les tendances futures
- **Systèmes de recommandation** : Suggérer des films aux usagers
- **NLP** : Analyse des titres pour identifier les thèmes populaires

---

## 7. LIMITATIONS ET POINTS D'ATTENTION

### 7.1 Limitations identifiées
1. **Encodage** : Problèmes d'affichage des caractères accentués
2. **Période limitée** : Données semblent se concentrer sur 2023
3. **Type de média** : Focus uniquement sur les films (DVD)
4. **Données manquantes** : Nécessité de vérifier les valeurs nulles

### 7.2 Nettoyage nécessaire
- Correction de l'encodage UTF-8
- Vérification des doublons
- Traitement des valeurs manquantes (NaN)
- Standardisation des noms d'auteurs/réalisateurs

---

## 8. CHARGEMENT ET PRÉPARATION DES DONNÉES

### 8.1 Code de chargement
```python
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Chargement du dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "grimespoint/toulouse-public-library-mdiathque-dataset",
    "toulouse_public_library_loans.csv",
    pandas_kwargs={
        'engine': 'python', 
        'on_bad_lines': 'skip', 
        'sep': ';',
        'encoding': 'utf-8'
    }
)
```

### 8.2 Exploration initiale
```python
# Dimensions du dataset
print(df.shape)

# Informations sur les colonnes
print(df.info())

# Statistiques descriptives
print(df.describe())

# Vérification des valeurs manquantes
print(df.isnull().sum())

# Vérification des doublons
print(df.duplicated().sum())
```

---

## 9. APPLICATIONS PRATIQUES

### 9.1 Pour la bibliothèque
- **Optimisation des achats** : Identifier les genres et réalisateurs populaires
- **Gestion des stocks** : Ajuster le nombre de copies selon la demande
- **Programmation culturelle** : Organiser des événements autour des films populaires

### 9.2 Pour la recherche
- **Sociologie de la culture** : Comprendre les goûts du public toulousain
- **Politiques publiques** : Évaluer l'impact des médiathèques
- **Marketing culturel** : Cibler les communications

---

## 10. CONCLUSION

Le dataset **"Toulouse Public Library Loans"** constitue une ressource précieuse pour l'analyse des comportements d'emprunt dans une bibliothèque publique française. Malgré quelques problèmes d'encodage, il offre des opportunités d'analyses riches pour :

✅ Comprendre les préférences culturelles des usagers  
✅ Optimiser la gestion des collections  
✅ Développer des systèmes de recommandation  
✅ Étudier l'évolution de la consommation de médias physiques  

### Prochaines étapes recommandées
1. Nettoyer les données (encodage, doublons)
2. Effectuer une analyse exploratoire approfondie (EDA)
3. Créer des visualisations interactives
4. Développer des modèles prédictifs si données temporelles disponibles
5. Comparer avec d'autres bibliothèques si données disponibles

---

**Date du rapport** : 27 novembre 2024  
**Dataset analysé** : Toulouse Public Library Loans  
**Plateforme** : Kaggle  
**Outil d'analyse** : Python (Pandas, KaggleHub)