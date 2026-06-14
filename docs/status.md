# Etat du projet Tox21

Derniere verification: 2026-06-07.

## Statut global

Le projet est fonctionnel et couvre le flux principal :

- dataset Tox21 telecharge et conserve dans `data/raw/`;
- analyse exploratoire et figures dans `reports/figures/`;
- descripteurs RDKit, Morgan fingerprints, MACCS keys et RDKit fingerprints dans `data/processed/`;
- modeles classiques entraines et sauvegardes dans `models/`;
- modele deep learning ToxNet sauvegarde dans `models/`;
- metriques JSON et comparaison finale dans `reports/`;
- explications SHAP dans `reports/figures/`;
- API FastAPI dans `api/main.py`;
- interface Streamlit dans `webapp/app.py`;
- rapport LaTeX dans `latex/rapport_tox21.tex`;
- rapport PDF compile dans `latex/rapport_tox21.pdf`;
- presentation PowerPoint de 20 slides dans `reports/Tox21_Presentation.pptx`;
- notebooks executes dans `notebooks/`.

## Resultats principaux

Meilleur modele par ROC-AUC sur NR-AhR:

- SVM + Morgan fingerprints: ROC-AUC 0.9112, F1 0.5879.
- RandomForest: ROC-AUC 0.9092, PR-AUC 0.6351.
- ToxNet: ROC-AUC 0.8768, F1 0.2105, PR-AUC 0.5589.

Le dataset contient 7,831 molecules, dont 7,823 valides apres parsing RDKit.
Pour NR-AhR: 768 positifs et 5,774 negatifs parmi les molecules labelisees.

## Verification

Commande executee:

```powershell
pytest -q
```

Resultat:

```text
28 passed, 1 skipped, 3 warnings
```

Les 9 notebooks demandes ont ete generes puis executes avec `jupyter nbconvert --execute --inplace`.

Le rapport PDF compile contient 52 pages.

## Outils installes

- Tectonic portable installe dans `tools/tectonic/tectonic.exe` pour compiler LaTeX sans installation systeme.
- Git installe via winget (`Git.Git` 2.54.0). Dans cette session, l'executable est accessible via `C:\Program Files\Git\cmd\git.exe`.
- `pypdf` installe pour verifier le nombre de pages du rapport PDF.

Commande de compilation PDF utilisee:

```powershell
.\tools\tectonic\tectonic.exe -X compile .\latex\rapport_tox21.tex --outdir .\latex
```
