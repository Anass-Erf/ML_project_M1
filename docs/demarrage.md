# Guide de demarrage du projet Tox21

Ce fichier explique comment demarrer, verifier et utiliser le projet.

## 1. Aller dans le dossier du projet

Ouvrir PowerShell puis executer:

```powershell
cd "C:\Desktop\ML\project"
```

## 2. Lancer l'application Streamlit

C'est l'interface graphique principale.

```powershell
python -m streamlit run webapp/app.py --server.address 127.0.0.1 --server.port 8501
```

Puis ouvrir:

```text
http://127.0.0.1:8501
```

Dans l'interface, tu peux:

- entrer un SMILES;
- choisir un modele;
- predire la toxicite;
- voir les figures;
- consulter les resultats SHAP;
- faire une prediction par lot.

## 3. Lancer l'API FastAPI

L'API est utile si tu veux tester les endpoints `/predict`, `/metrics`, etc.

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Puis ouvrir:

```text
http://127.0.0.1:8000/docs
```

Exemple de prediction avec l'API:

```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
```

## 4. Executer les tests

Pour verifier que le projet fonctionne:

```powershell
pytest -q
```

Resultat attendu:

```text
28 passed, 1 skipped
```

## 5. Regenerer les notebooks

Les notebooks sont deja executes dans `notebooks/`.
Pour les regenerer:

```powershell
python src/generate_notebooks.py
```

Pour les reexecuter:

```powershell
Get-ChildItem .\notebooks\*.ipynb | ForEach-Object {
    jupyter nbconvert --to notebook --execute --inplace $_.FullName --ExecutePreprocessor.timeout=120
}
```

## 6. Relancer le pipeline complet

Attention: cette commande peut prendre plusieurs minutes, car elle refait les features, l'entrainement, les figures et SHAP.

```powershell
python src/pipeline.py
```

## 7. Regenerer le rapport et la presentation

Rapport LaTeX:

```powershell
python src/generate_latex.py
```

Presentation PowerPoint:

```powershell
python src/generate_presentation.py
```

Compiler le PDF avec Tectonic portable:

```powershell
.\tools\tectonic\tectonic.exe -X compile .\latex\rapport_tox21.tex --outdir .\latex
```

Fichiers generes:

- `latex/rapport_tox21.pdf`
- `latex/rapport_tox21.tex`
- `reports/Tox21_Presentation.pptx`

## 8. Fichiers importants

- Application Streamlit: `webapp/app.py`
- API FastAPI: `api/main.py`
- Pipeline complet: `src/pipeline.py`
- Tests: `tests/test_pipeline.py`
- Rapport PDF: `latex/rapport_tox21.pdf`
- Presentation: `reports/Tox21_Presentation.pptx`
- Etat du projet: `docs/status.md`

## 9. Notes utiles

Git est installe, mais si PowerShell ne le trouve pas encore dans le PATH, utiliser:

```powershell
& "C:\Program Files\Git\cmd\git.exe" --version
```

Tectonic est installe localement dans:

```text
tools/tectonic/tectonic.exe
```

Si le port `8501` est deja utilise, lancer Streamlit sur un autre port:

```powershell
python -m streamlit run webapp/app.py --server.address 127.0.0.1 --server.port 8502
```
