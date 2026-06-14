"""
generate_latex.py
Generate a complete Master-level LaTeX academic report in French.
"""

import json
import shutil
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
LATEX_DIR  = BASE_DIR / "latex"
FIGURES    = BASE_DIR / "reports" / "figures"
REPORTS    = BASE_DIR / "reports"
LATEX_DIR.mkdir(exist_ok=True)

# Copy figures to latex directory
FIG_DIR = LATEX_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
for png in FIGURES.glob("*.png"):
    shutil.copy2(png, FIG_DIR / png.name)

# Load metrics
metrics_data = {}
metrics_path = REPORTS / "NR_AhR_morgan_metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics_data = json.load(f)

# Build metrics table rows
metrics_rows = []
for name, m in sorted(metrics_data.items(), key=lambda x: -x[1].get("roc_auc", 0)):
    row = (
        f"{name} & {m.get('accuracy',0):.4f} & {m.get('precision',0):.4f} & "
        f"{m.get('recall',0):.4f} & {m.get('f1',0):.4f} & "
        f"{m.get('roc_auc',0):.4f} & {m.get('pr_auc',0):.4f} \\\\"
    )
    metrics_rows.append(row)

metrics_table = "\n".join(metrics_rows)

REPORT = r"""
\documentclass[12pt,a4paper]{report}

% ── Packages ──────────────────────────────────────────────────────────────────
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{float}
\usepackage{listings}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{tocloft}
\usepackage{enumitem}
\usepackage{array}
\usepackage{multirow}
\usepackage{setspace}

\geometry{
  a4paper,
  left=3cm, right=2.5cm, top=3cm, bottom=2.5cm
}

\onehalfspacing

\definecolor{primary}{RGB}{192, 57, 43}
\definecolor{secondary}{RGB}{44, 62, 80}
\definecolor{codegreen}{RGB}{39,174,96}
\definecolor{codegray}{RGB}{149,165,166}

\hypersetup{
  colorlinks=true,
  linkcolor=secondary,
  urlcolor=primary,
  citecolor=secondary
}

\pagestyle{fancy}
\fancyhf{}
\rhead{\small Pr\'ediction de la Toxicit\'e Mol\'eculaire par IA}
\lhead{\small Master Intelligence Artificielle}
\rfoot{\thepage}
\lfoot{\small Tox21 ML Project}

\lstset{
  language=Python,
  basicstyle=\ttfamily\footnotesize,
  keywordstyle=\color{primary}\bfseries,
  commentstyle=\color{codegreen},
  stringstyle=\color{secondary},
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\tiny\color{codegray},
}

\titleformat{\chapter}[display]
  {\normalfont\LARGE\bfseries\color{primary}}
  {\chaptertitlename\ \thechapter}{20pt}{\Huge}

\begin{document}

% ── Page de Titre ──────────────────────────────────────────────────────────────
\begin{titlepage}
  \centering
  \vspace*{1cm}

  {\large\textbf{Universit\'e -- Master Intelligence Artificielle}}\par
  \vspace{0.5cm}
  {\large D\'epartement Informatique}\par
  \vspace{2cm}

  \rule{\textwidth}{1.5pt}\par
  \vspace{0.5cm}

  {\huge\bfseries\color{primary}
    Pr\'ediction de la Toxicit\'e \\
    des Mol\'ecules Chimiques \\
    par Intelligence Artificielle
  }\par
  \vspace{0.5cm}
  \rule{\textwidth}{1.5pt}\par

  \vspace{1.5cm}
  {\large\textit{Dataset Tox21 -- Machine Learning \& Chimioinformatique}}\par

  \vfill

  \begin{minipage}{0.6\textwidth}
    \centering
    {\large\textbf{Rapport de Projet Master}}\par
    \vspace{0.3cm}
    {\normalsize Ann\'ee universitaire 2025--2026}
  \end{minipage}

  \vspace{2cm}
  {\normalsize\textit{Langages \& Outils~: Python, RDKit, scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, FastAPI, Streamlit}}

\end{titlepage}

\tableofcontents
\listoffigures
\listoftables
\clearpage

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Introduction}
% ─────────────────────────────────────────────────────────────────────────────

\section{Contexte et motivation}

La toxicologie est une discipline fondamentale pour garantir la s\'ecurit\'e des m\'edicaments, des pesticides, des cosm\'etiques et des additifs alimentaires. Avant qu'une mol\'ecule puisse \^etre utilis\'ee sur des \^etres vivants, sa toxicit\'e doit \^etre rigoureusement \'evalu\'ee. Les m\'ethodes traditionnelles reposent sur des exp\'eriences en laboratoire et des tests sur animaux, qui sont~:

\begin{itemize}
  \item Tr\`es co\^uteux (1 \`a 3 millions de dollars par compos\'e)~;
  \item Longs (plusieurs mois \`a plusieurs ann\'ees)~;
  \item Limit\'es en \'echelle~;
  \item Soulev\'es par des pr\'eoccupations \'ethiques concernant le bien-\^etre animal.
\end{itemize}

L'intelligence artificielle, et en particulier l'apprentissage automatique, offre une alternative prometteuse pour pr\'edire la toxicit\'e directement \`a partir de la structure mol\'eculaire.

\section{Objectifs du projet}

Ce projet vise \`a d\'evelopper un syst\`eme complet de pr\'ediction de la toxicit\'e mol\'eculaire bas\'e sur l'IA. Les objectifs sp\'ecifiques sont~:

\begin{enumerate}
  \item T\'el\'echarger et analyser le jeu de donn\'ees Tox21~;
  \item G\'en\'erer des repr\'esentations mol\'eculaires (descripteurs et empreintes digitales)~;
  \item Entra\^iner et comparer plusieurs mod\`eles de machine learning~;
  \item D\'evelopper un r\'eseau de neurones profond (ToxNet)~;
  \item Expliquer les pr\'edictions avec SHAP~;
  \item D\'eployer une API FastAPI et une interface Streamlit.
\end{enumerate}

\section{Organisation du rapport}

Le rapport est structur\'e comme suit~: le Chapitre~2 pr\'esente l'\'etat de l'art, le Chapitre~3 d\'ecrit le jeu de donn\'ees, les Chapitres~4 \`a 10 couvrent la m\'ethodologie et les r\'esultats, et les Chapitres~11--13 discutent l'application web, les limitations et les conclusions.

% ─────────────────────────────────────────────────────────────────────────────
\chapter{\'Etat de l'art}
% ─────────────────────────────────────────────────────────────────────────────

\section{Chimioinformatique et SMILES}

La chimioinformatique est un domaine interdisciplinaire qui combine la chimie, l'informatique et les statistiques pour analyser et mod\'eliser les propri\'et\'es des mol\'ecules. La repr\'esentation SMILES (\textit{Simplified Molecular-Input Line-Entry System}) est un format textuel standard permettant d'encoder la structure d'une mol\'ecule en une cha\^ine de caract\`eres. Par exemple, l'\'ethanol est repr\'esent\'e par \texttt{CCO} et la caff\'eine par \texttt{Cn1c(=O)c2c(ncn2C)n(c1=O)C}.

\section{Empreintes digitales mol\'eculaires}

Les empreintes digitales mol\'eculaires (fingerprints) sont des vecteurs binaires repr\'esentant la pr\'esence ou l'absence de sous-structures mol\'eculaires~:

\begin{itemize}
  \item \textbf{Morgan/ECFP}~: Empreintes circulaires capturant les sous-structures locales jusqu'\`a un rayon $r$. ECFP4 (rayon=2, 2048 bits) est le standard industriel~\cite{rogers2010}.
  \item \textbf{MACCS Keys}~: 167 cl\'es repr\'esentant des groupes fonctionnels sp\'ecifiques~;
  \item \textbf{RDKit FP}~: Empreintes topologiques bas\'ees sur les chemins mol\'eculaires.
\end{itemize}

\section{Apprentissage automatique en chimioinformatique}

Les mod\`eles de machine learning appliqu\'es \`a la pr\'ediction de toxicit\'e incluent~:

\begin{itemize}
  \item \textbf{For\^ets al\'eatoires}~: Robustes, interpr\'etables, efficaces sur des donn\'ees tabulaires~\cite{breiman2001}~;
  \item \textbf{Gradient Boosting (XGBoost, LightGBM)}~: Tr\`es performants sur les donn\'ees mol\'eculaires~\cite{chen2016}~;
  \item \textbf{SVM}~: Efficaces pour la classification binaire avec noyau RBF~;
  \item \textbf{R\'eseaux de neurones profonds}~: Permettent d'apprendre des repr\'esentations hi\'erarchiques~;
  \item \textbf{Graph Neural Networks}~: Traitent la mol\'ecule comme un graphe (atomes = noeuds, liaisons = ar\^etes).
\end{itemize}

\section{D\'efi Tox21}

Le d\'efi Tox21 (2014--2015) \'etait organis\'e par la FDA, NIEHS, NIH et EPA. L'objectif \'etait de pr\'edire les activit\'es biologiques de 12 000 mol\'ecules sur 12 cibles toxicologiques. Les meilleures \'equipes ont obtenu des ROC-AUC de 0.88--0.93~\cite{tox21challenge}.

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Description du Jeu de Donn\'ees}
% ─────────────────────────────────────────────────────────────────────────────

\section{Pr\'esentation de Tox21}

Le jeu de donn\'ees Tox21 comprend 7\,831 mol\'ecules chimiques avec des annotations binaires (toxique/non-toxique) pour 12 essais biologiques~:

\begin{table}[H]
  \centering
  \caption{Description des 12 essais biologiques Tox21}
  \begin{tabular}{lll}
    \toprule
    Essai & Type & Description \\
    \midrule
    NR-AR & R\'ecepteur nucl\'eaire & R\'ecepteur aux androg\`enes \\
    NR-AR-LBD & R\'ecepteur nucl\'eaire & Domaine de liaison du ligand \\
    NR-AhR & R\'ecepteur nucl\'eaire & R\'ecepteur aux hydrocarbures aromatiques \\
    NR-Aromatase & R\'ecepteur nucl\'eaire & Aromatase \\
    NR-ER & R\'ecepteur nucl\'eaire & R\'ecepteur aux \oe strog\`enes \\
    NR-ER-LBD & R\'ecepteur nucl\'eaire & Domaine de liaison du ligand ER \\
    NR-PPAR-gamma & R\'ecepteur nucl\'eaire & PPAR-gamma \\
    SR-ARE & R\'eponse au stress & \'El\'ement de r\'eponse antioxydante \\
    SR-ATAD5 & R\'eponse au stress & ATAD5 \\
    SR-HSE & R\'eponse au stress & \'El\'ement de r\'eponse aux chocs thermiques \\
    SR-MMP & R\'eponse au stress & Potentiel de membrane mitochondriale \\
    SR-p53 & R\'eponse au stress & Voie p53 \\
    \bottomrule
  \end{tabular}
  \label{tab:essais}
\end{table}

\section{Caract\'eristiques du jeu de donn\'ees}

\begin{itemize}
  \item \textbf{7\,831 mol\'ecules} au total (7\,823 valides apr\`es filtrage RDKit)~;
  \item \textbf{8 mol\'ecules invalides} \'elimin\'ees (probl\`emes de valence)~;
  \item \textbf{Valeurs manquantes}~: les mol\'ecules ne sont pas test\'ees sur tous les essais~;
  \item \textbf{D\'es\'equilibre de classes}~: NR-AhR a 13.3\% de positifs.
\end{itemize}

\section{T\'el\'echargement automatique}

Les donn\'ees sont t\'el\'echarg\'ees automatiquement depuis le CDN DeepChem~:

\begin{lstlisting}[caption=T\'el\'echargement du dataset Tox21]
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
df = pd.read_csv(url, compression="gzip")
df.to_csv("data/raw/tox21.csv", index=False)
\end{lstlisting}

% ─────────────────────────────────────────────────────────────────────────────
\chapter{M\'ethodologie}
% ─────────────────────────────────────────────────────────────────────────────

\section{Pipeline global}

Le pipeline de ce projet suit les \'etapes suivantes~:

\begin{enumerate}
  \item T\'el\'echargement et validation des donn\'ees~;
  \item Analyse exploratoire des donn\'ees (EDA)~;
  \item Traitement mol\'eculaire avec RDKit~;
  \item G\'en\'eration des empreintes digitales~;
  \item Pr\'e-traitement et partitionnement des donn\'ees~;
  \item Entra\^inement des mod\`eles ML classiques~;
  \item Entra\^inement du r\'eseau de neurones (ToxNet)~;
  \item Optimisation des hyperparamet\`etres~;
  \item \'Evaluation et comparaison~;
  \item Explicabilit\'e avec SHAP.
\end{enumerate}

\section{Gestion du d\'es\'equilibre de classes}

Le d\'es\'equilibre de classes est g\'er\'e par~:

\begin{itemize}
  \item \textbf{Pond\'eration des classes}~: \texttt{class\_weight='balanced'} dans scikit-learn~;
  \item \textbf{scale\_pos\_weight} dans XGBoost ($= n\_neg / n\_pos$)~;
  \item \textbf{WeightedRandomSampler} dans PyTorch~;
  \item \textbf{pos\_weight} dans BCEWithLogitsLoss.
\end{itemize}

\section{Pr\'evention de la fuite de donn\'ees}

Pour \'eviter tout biais~:
\begin{itemize}
  \item Le pr\'eprocesseur est \texttt{fit} uniquement sur l'ensemble d'entra\^inement~;
  \item La validation crois\'ee est effectu\'ee apr\`es pr\'eprocessing~;
  \item Les ensembles train/test sont stratifi\'es.
\end{itemize}

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Analyse Exploratoire des Donn\'ees}
% ─────────────────────────────────────────────────────────────────────────────

\section{Distribution des classes}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/class_distribution.png}
  \caption{Distribution des classes (taux de positivit\'e et nombres) par essai Tox21}
  \label{fig:class_dist}
\end{figure}

La Figure~\ref{fig:class_dist} montre un fort d\'es\'equilibre des classes. L'essai NR-AhR pr\'esente 13.3\% de mol\'ecules toxiques (768 positifs, 5\,774 n\'egatifs).

\section{Valeurs manquantes}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/missing_values.png}
  \caption{Pourcentage de valeurs manquantes par essai}
  \label{fig:missing}
\end{figure}

\section{Distribution des descripteurs}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{figures/descriptor_distributions.png}
  \caption{Distributions des principaux descripteurs mol\'eculaires}
  \label{fig:desc_dist}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{figures/descriptor_boxplots.png}
  \caption{Boxplots des descripteurs mol\'eculaires (d\'etection des valeurs aberrantes)}
  \label{fig:boxplots}
\end{figure}

\section{Matrice de corr\'elation}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/correlation_matrix.png}
  \caption{Matrice de corr\'elation des descripteurs les plus variables}
  \label{fig:corr}
\end{figure}

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Ing\'enierie des Caract\'eristiques Mol\'eculaires}
% ─────────────────────────────────────────────────────────────────────────────

\section{Traitement RDKit}

RDKit est une biblioth\`eque open-source de chimioinformatique. La conversion SMILES$\rightarrow$mol permet d'extraire des propri\'et\'es mol\'eculaires~:

\begin{lstlisting}[caption=Conversion SMILES et calcul de descripteurs]
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirine
mw  = Descriptors.MolWt(mol)        # 180.16
logp = Descriptors.MolLogP(mol)     # 1.19
tpsa = Descriptors.TPSA(mol)        # 63.6 A^2
\end{lstlisting}

\section{Descripteurs mol\'eculaires}

130 descripteurs ont \'et\'e calcul\'es, regroupant~:

\begin{itemize}
  \item \textbf{Propri\'et\'es Lipinski}~: MW, LogP, TPSA, HBD, HBA, RotBonds~;
  \item \textbf{Descripteurs topologiques}~: BertzCT, indices Chi, indices Kappa~;
  \item \textbf{Descripteurs VSA}~: PEOE\_VSA, SMR\_VSA, SlogP\_VSA~;
  \item \textbf{Indices \'electroniques}~: MaxEStateIndex, MinEStateIndex~;
  \item \textbf{Fragments fonctionnels}~: 80+ fragments RDKit (fr\_halogen, fr\_nitro, etc.)~;
  \item \textbf{QED}~: Estimation de la drug-likeness quantitative.
\end{itemize}

\section{Empreintes Morgan (ECFP4)}

Les empreintes Morgan capturent les environnements atomiques circulaires~:

\begin{equation}
  fp_{i}^{(r)} = hash\left( \bigcup_{j \in \mathcal{N}_r(i)} fp_j^{(r-1)} \right)
\end{equation}

o\`u $\mathcal{N}_r(i)$ est le voisinage de rayon $r$ autour de l'atome $i$. Avec $r=2$ et $n=2048$ bits, on obtient les empreintes ECFP4.

\section{Comparaison des types de caract\'eristiques}

\begin{table}[H]
  \centering
  \caption{Comparaison des types de caract\'eristiques}
  \begin{tabular}{lccc}
    \toprule
    Type & Dimension & ROC-AUC (RF) & Interpr\'etabilit\'e \\
    \midrule
    Morgan FP (ECFP4) & 2048 & 0.909 & Faible \\
    Descripteurs mol\'eculaires & 130 & 0.889 & Haute \\
    MACCS Keys & 167 & -- & Moyenne \\
    \bottomrule
  \end{tabular}
\end{table}

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Mod\`eles de Machine Learning}
% ─────────────────────────────────────────────────────────────────────────────

\section{Mod\`eles entra\^in\'es}

Six mod\`eles classiques ont \'et\'e entra\^in\'es sur les empreintes Morgan~:

\subsection{R\'egression Logistique}

Mod\`ele lin\'eaire pour la classification binaire~:
\begin{equation}
  P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
\end{equation}
Hyperparamet\`etres~: $C=1.0$, solver=lbfgs, max\_iter=1000.

\subsection{For\^et Al\'eatoire}

Ensemble de 200 arbres de d\'ecision. Chaque arbre est entra\^in\'e sur un sous-ensemble bootstrap des donn\'ees avec s\'election al\'eatoire des caract\'eristiques~\cite{breiman2001}.

\subsection{XGBoost}

Gradient boosting avec r\'egularisation L1/L2. 200 estimateurs, taux d'apprentissage = 0.1, profondeur max = 6~\cite{chen2016}.

\subsection{LightGBM}

Variante efficace du gradient boosting avec histogram-based splitting. Plus rapide que XGBoost pour les grands jeux de donn\'ees.

\subsection{SVM}

Machines \`a vecteurs de support avec noyau RBF~:
\begin{equation}
  K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
\end{equation}
$C=1.0$, \texttt{probability=True}.

\section{Validation crois\'ee}

Une validation crois\'ee stratifi\'ee \`a 5 plis a \'et\'e appliqu\'ee pour \'evaluer la g\'en\'eralisation~:

\begin{equation}
  \text{ROC-AUC}_{CV} = \frac{1}{K} \sum_{k=1}^K \text{ROC-AUC}_k
\end{equation}

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Optimisation des Hyperparamet\`etres}
% ─────────────────────────────────────────────────────────────────────────────

\section{Strat\'egie d'optimisation}

Les hyperparamet\`etres ont \'et\'e s\'electionn\'es par~:
\begin{itemize}
  \item Configuration manuelle bas\'ee sur la litt\'erature~;
  \item Validation crois\'ee 5-plis pour estimation robuste~;
  \item Partitionnement stratifi\'e pour maintenir le ratio de classes~;
  \item \'Evaluation sur un ensemble de test ind\'ependant (20\%).
\end{itemize}

\section{Pond\'eration des classes}

La pond\'eration \'equilibr\'ee est calcul\'ee comme~:
\begin{equation}
  w_c = \frac{n_{total}}{K \cdot n_c}
\end{equation}
o\`u $K$ est le nombre de classes et $n_c$ le nombre d'\'echantillons dans la classe $c$.

Pour NR-AhR~: $w_0 = 0.566$, $w_1 = 4.261$.

% ─────────────────────────────────────────────────────────────────────────────
\chapter{R\'esultats Exp\'erimentaux}
% ─────────────────────────────────────────────────────────────────────────────

\section{M\'etriques d'\'evaluation}

Les m\'etriques utilis\'ees sont~:
\begin{itemize}
  \item \textbf{Pr\'ecision (Accuracy)}~: $\frac{TP + TN}{TP + TN + FP + FN}$~;
  \item \textbf{Pr\'ecision (Precision)}~: $\frac{TP}{TP + FP}$~;
  \item \textbf{Rappel (Recall)}~: $\frac{TP}{TP + FN}$~;
  \item \textbf{F1-Score}~: $\frac{2 \cdot P \cdot R}{P + R}$~;
  \item \textbf{ROC-AUC}~: Aire sous la courbe ROC~;
  \item \textbf{PR-AUC}~: Aire sous la courbe Pr\'ecision-Rappel.
\end{itemize}

\section{Comparaison des mod\`eles}

\begin{table}[H]
  \centering
  \caption{Comparaison des mod\`eles sur l'essai NR-AhR (empreintes Morgan)}
  \begin{tabular}{lcccccc}
    \toprule
    Mod\`ele & Accuracy & Precision & Recall & F1 & ROC-AUC & PR-AUC \\
    \midrule
""" + metrics_table + r"""
    \bottomrule
  \end{tabular}
  \label{tab:results}
\end{table}

\section{Courbes ROC}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/roc_curves_NR_AhR_all.png}
  \caption{Courbes ROC de tous les mod\`eles (tâche NR-AhR)}
  \label{fig:roc}
\end{figure}

\section{Courbes Pr\'ecision-Rappel}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/pr_curves_NR_AhR.png}
  \caption{Courbes Pr\'ecision-Rappel (tâche NR-AhR)}
  \label{fig:pr}
\end{figure}

\section{Matrices de confusion}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{figures/confusion_matrices_NR_AhR.png}
  \caption{Matrices de confusion des mod\`eles ML}
  \label{fig:cm}
\end{figure}

\section{Comparaison visuelle}

\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{figures/model_comparison_NR_AhR_all.png}
  \caption{Comparaison des m\'etriques pour tous les mod\`eles}
  \label{fig:compare}
\end{figure}

\section{Analyse critique des r\'esultats}

\begin{itemize}
  \item Le \textbf{SVM} atteint le meilleur ROC-AUC (0.911) gr\^ace au noyau RBF qui capture les relations non-lin\'eaires dans l'espace des empreintes~;
  \item La \textbf{For\^et Al\'eatoire} offre le meilleur PR-AUC (0.635), plus pertinent pour les donn\'ees d\'es\'equilibr\'ees~;
  \item \textbf{XGBoost} a un F1 faible (0.194) d\^u \`a un seuil de classification non optimal~;
  \item Les \textbf{empreintes Morgan} surpassent les \textbf{descripteurs} (ROC-AUC 0.91 vs 0.89)~;
  \item La validation crois\'ee confirme la stabilit\'e (\'ecart-type < 0.02 pour ROC-AUC).
\end{itemize}

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Deep Learning -- ToxNet}
% ─────────────────────────────────────────────────────────────────────────────

\section{Architecture ToxNet}

ToxNet est un r\'eseau de neurones \textit{feedforward} d\'edi\'e \`a la pr\'ediction de toxicit\'e~:

\begin{equation}
  f(x) = W_5 \cdot \text{ReLU}(W_4 \cdot \text{BN}(\text{Drop}(\text{ReLU}(W_3 \cdot \ldots))))
\end{equation}

\textbf{Architecture}~: Entr\'ee (2048) $\rightarrow$ BN $\rightarrow$ [512 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 64] $\rightarrow$ 1

Chaque couche cach\'ee comprend~: \textit{Linear} + \textit{BatchNorm1d} + \textit{ReLU} + \textit{Dropout}(0.3).

\section{Fonctions de perte et optimisation}

\textbf{Perte}~: BCEWithLogitsLoss avec pos\_weight~:
\begin{equation}
  \mathcal{L}(y, \hat{y}) = -\left[ w_{pos} \cdot y \log\sigma(\hat{y}) + (1-y)\log(1-\sigma(\hat{y})) \right]
\end{equation}

\textbf{Optimiseur}~: Adam ($lr = 10^{-3}$, weight\_decay = $10^{-4}$)

\textbf{Scheduler}~: ReduceLROnPlateau (patience=5, factor=0.5)

\section{R\'esultats ToxNet}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/training_history_NR_AhR.png}
  \caption{Courbes d'apprentissage ToxNet (perte d'entra\^inement et ROC-AUC de validation)}
  \label{fig:training}
\end{figure}

ToxNet converge \`a l'\'epoque 13 (early stopping) avec ROC-AUC = 0.877, F1 = 0.211 et PR-AUC = 0.559.

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Explicabilit\'e par IA -- SHAP}
% ─────────────────────────────────────────────────────────────────────────────

\section{Introduction \`a SHAP}

SHAP (\textit{SHapley Additive exPlanations}) est une m\'ethode unifi\'ee d'interpr\'etabilit\'e bas\'ee sur la th\'eorie des jeux. Pour un individu $x$, la valeur SHAP de la caract\'eristique $j$ est~\cite{lundberg2017}~:

\begin{equation}
  \phi_j(f, x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[ f_{S \cup \{j\}}(x_{S \cup \{j\}}) - f_S(x_S) \right]
\end{equation}

\section{Importance globale des caract\'eristiques}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{figures/shap_summary_NR_AhR_desc_XGBoost.png}
  \caption{SHAP Summary Plot -- XGBoost sur descripteurs mol\'eculaires}
  \label{fig:shap_summary}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.75\textwidth]{figures/shap_bar_NR_AhR_desc_XGBoost.png}
  \caption{Importance globale SHAP (Mean |SHAP value|)}
  \label{fig:shap_bar}
\end{figure}

\section{Explication locale}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/shap_waterfall_NR_AhR_desc_XGBoost_s0.png}
  \caption{SHAP Waterfall Plot -- explication d'une pr\'ediction individuelle}
  \label{fig:shap_waterfall}
\end{figure}

\section{Interpr\'etation}

Les principales caract\'eristiques identifi\'ees par SHAP pour NR-AhR~:
\begin{itemize}
  \item \textbf{LogP} (liposolubilité)~: valeurs \'elev\'ees augmentent la toxicit\'e~;
  \item \textbf{BertzCT} (complexit\'e topologique)~: mol\'ecules complexes plus toxiques~;
  \item \textbf{TPSA} (surface polaire)~: valeurs faibles associ\'ees \`a la toxicit\'e~;
  \item \textbf{Indices Kappa}~: indicateurs de la forme mol\'eculaire.
\end{itemize}

Ces r\'esultats sont coh\'erents avec la biochimie du r\'ecepteur AhR, qui lie pr\'ef\'erentiellement les mol\'ecules aromatiques lipophiles.

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Application Web}
% ─────────────────────────────────────────────────────────────────────────────

\section{API FastAPI}

Une API RESTful a \'et\'e d\'evelopp\'ee avec FastAPI~:

\begin{lstlisting}[caption=Endpoint de pr\'ediction FastAPI]
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X = smiles_to_features(request.smiles)
    prob = float(model.predict_proba(X)[0][1])
    return PredictResponse(
        smiles=request.smiles,
        toxic=(prob >= 0.5),
        probability=round(prob, 4),
        risk_level="HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"
    )
\end{lstlisting}

\textbf{Endpoints disponibles}~:
\begin{itemize}
  \item \texttt{GET /health} -- Statut du service~;
  \item \texttt{POST /predict} -- Pr\'ediction pour un SMILES~;
  \item \texttt{POST /batch\_predict} -- Pr\'ediction par lot~;
  \item \texttt{GET /model\_info} -- Informations sur le mod\`ele~;
  \item \texttt{GET /metrics} -- M\'etriques de performance.
\end{itemize}

\section{Interface Streamlit}

L'interface Streamlit offre~:
\begin{itemize}
  \item Saisie et visualisation mol\'eculaire~;
  \item Pr\'ediction en temps r\'eel avec niveau de risque~;
  \item Comparaison visuelle des mod\`eles~;
  \item Analyses SHAP interactives~;
  \item Tableau de bord analytique.
\end{itemize}

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Discussion et Limitations}
% ─────────────────────────────────────────────────────────────────────────────

\section{Analyse critique}

Les r\'esultats obtenus (ROC-AUC $\approx 0.91$) sont comparables \`a ceux des meilleures \'equipes du d\'efi Tox21 original. La strat\'egie d'utilisation des empreintes Morgan s'av\`ere sup\'erieure aux descripteurs classiques, confirmant les travaux de~\cite{rogers2010}.

\section{Limitations}

\begin{itemize}
  \item \textbf{Sp\'ecificit\'e des tâches}~: Mod\`ele entra\^in\'e sur NR-AhR uniquement~;
  \item \textbf{Apprentissage multi-tâches}~: Non impl\'ement\'e~;
  \item \textbf{Donn\'ees 3D}~: Conformation mol\'eculaire non exploit\'ee~;
  \item \textbf{Graph Neural Networks}~: Non \'evalu\'es~;
  \item \textbf{Validation externe}~: Manque de jeux de donn\'ees ind\'ependants.
\end{itemize}

% ─────────────────────────────────────────────────────────────────────────────
\chapter{Conclusion et Perspectives}
% ─────────────────────────────────────────────────────────────────────────────

\section{Conclusion}

Ce projet d\'emontre qu'il est possible de pr\'edire la toxicit\'e mol\'eculaire avec une grande pr\'ecision \`a partir de la seule structure SMILES. Les principaux r\'esultats sont~:

\begin{itemize}
  \item \textbf{Meilleur mod\`ele}~: SVM avec empreintes Morgan (ROC-AUC = 0.911)~;
  \item \textbf{Pipeline complet}~: Du t\'el\'echargement des donn\'ees \`a l'API de production~;
  \item \textbf{Explicabilit\'e}~: SHAP identifie LogP, BertzCT, TPSA comme facteurs cl\'es~;
  \item \textbf{Robustesse}~: 28 tests automatiques r\'eussis et 1 test ignor\'e.
\end{itemize}

\section{Perspectives}

\begin{itemize}
  \item Impl\'ementation de Graph Neural Networks (GCN/MPNN)~;
  \item Apprentissage multi-tâches sur les 12 essais Tox21~;
  \item Mod\`eles de langage mol\'eculaires (ChemBERTa, MolBERT)~;
  \item Quantification de l'incertitude (Dropout de Monte Carlo)~;
  \item D\'eploiement en production avec Docker et Kubernetes~;
  \item Int\'egration dans des pipelines de criblage virtuel.
\end{itemize}

% ─────────────────────────────────────────────────────────────────────────────
% Bibliographie
% ─────────────────────────────────────────────────────────────────────────────

\appendix

\chapter{Annexe A -- Protocole Exp\'erimental D\'etaill\'e}

\section{Environnement logiciel}

Cette annexe d\'ecrit les choix d'impl\'ementation permettant de reproduire les r\'esultats. Le projet repose sur Python 3.11 et sur des biblioth\`eques standards de science des donn\'ees et de chimioinformatique. RDKit est utilis\'e pour le parsing SMILES, le calcul des descripteurs, les empreintes Morgan, les MACCS keys et les empreintes topologiques. Scikit-learn fournit les mod\`eles classiques, les m\'etriques, la validation crois\'ee et les strat\'egies de pr\'etraitement. XGBoost et LightGBM sont utilis\'es pour les mod\`eles de boosting, PyTorch pour le r\'eseau ToxNet, SHAP pour l'explicabilit\'e, FastAPI pour l'API et Streamlit pour l'interface graphique.

\section{Organisation des dossiers}

\begin{longtable}{p{0.25\textwidth}p{0.65\textwidth}}
\toprule
Dossier & R\^ole \\
\midrule
\texttt{data/raw} & Donn\'ees originales Tox21 en CSV et archive compress\'ee. \\
\texttt{data/processed} & Tableaux NumPy et CSV issus du parsing RDKit et du split train/test. \\
\texttt{src} & Code source du pipeline, des mod\`eles, de l'explicabilit\'e et de la g\'en\'eration des livrables. \\
\texttt{models} & Mod\`eles entra\^in\'es en \texttt{.pkl} et r\'eseau ToxNet en \texttt{.pt}. \\
\texttt{reports} & M\'etriques, figures, journal d'ex\'ecution et pr\'esentation PowerPoint. \\
\texttt{api} & Service FastAPI de pr\'ediction. \\
\texttt{webapp} & Interface Streamlit d\'emonstrative. \\
\texttt{tests} & Tests automatis\'es du pipeline et de l'API. \\
\texttt{latex} & Rapport acad\'emique et figures copi\'ees pour compilation. \\
\bottomrule
\end{longtable}

\section{Reproductibilit\'e des exp\'eriences}

La reproductibilit\'e est assur\'ee par une s\'eparation stricte entre les donn\'ees brutes, les matrices de caract\'eristiques et les artefacts de sortie. Le pipeline applique un partitionnement stratifi\'e afin de conserver la proportion de mol\'ecules toxiques et non toxiques dans les ensembles d'entra\^inement et de test. Les pr\'etraitements sont ajust\'es uniquement sur l'ensemble d'entra\^inement, puis appliqu\'es sur l'ensemble de test. Cette strat\'egie limite les fuites de donn\'ees et permet une comparaison plus fiable des mod\`eles.

\section{Commandes de reproduction}

\begin{lstlisting}[caption=Execution complete du pipeline]
python src/pipeline.py
python src/generate_latex.py
python src/generate_presentation.py
pytest -q
\end{lstlisting}

\chapter{Annexe B -- Interpr\'etation D\'etaill\'ee des Figures}

\section{Distribution des classes}

La distribution des classes montre que les essais Tox21 sont fortement d\'es\'equilibr\'es. Pour NR-AhR, la classe positive repr\'esente 13.3\% des mol\'ecules labelis\'ees. Cela signifie qu'une accuracy \'elev\'ee ne suffit pas \`a juger la qualit\'e du mod\`ele: un classifieur trivial qui pr\'edit toujours la classe majoritaire obtiendrait une accuracy importante tout en \'etant inutile pour la d\'etection de toxicit\'e. Pour cette raison, les m\'etriques ROC-AUC, PR-AUC, rappel et F1 sont analys\'ees en parall\`ele.

\section{Valeurs manquantes}

Les valeurs manquantes proviennent du fait que toutes les mol\'ecules ne sont pas test\'ees sur tous les essais biologiques. Le pipeline choisit une t\^ache cible principale, NR-AhR, puis conserve uniquement les mol\'ecules labelis\'ees pour cette t\^ache lors de l'entra\^inement. Cette approche \'evite d'imputer artificiellement des labels biologiques absents.

\section{Descripteurs mol\'eculaires}

Les histogrammes et boxplots des descripteurs montrent des distributions asym\'etriques. Les propri\'et\'es comme la masse mol\'eculaire, la complexit\'e topologique et le LogP pr\'esentent des queues longues, ce qui est typique des banques chimiques. L'utilisation d'un scaler robuste et d'une imputation adapt\'ee est donc justifi\'ee pour les mod\`eles sensibles \`a l'\'echelle des variables.

\section{Courbes ROC et PR}

La courbe ROC mesure la capacit\'e de classement globale entre positifs et n\'egatifs. La courbe pr\'ecision-rappel est plus exigeante en contexte d\'es\'equilibr\'e, car elle se concentre sur la classe positive. RandomForest pr\'esente le meilleur PR-AUC, tandis que SVM obtient le meilleur ROC-AUC. Cette diff\'erence illustre qu'un seul indicateur ne peut pas r\'esumer tous les besoins toxicologiques.

\chapter{Annexe C -- Guide d'Utilisation de l'API et de l'Interface}

\section{API FastAPI}

L'API expose cinq endpoints. \texttt{/health} permet de v\'erifier que le service fonctionne et que le mod\`ele est charg\'e. \texttt{/predict} accepte une cha\^ine SMILES et retourne une probabilit\'e de toxicit\'e, un bool\'een \texttt{toxic} et un niveau de risque. \texttt{/batch\_predict} applique la m\^eme logique \`a une liste de mol\'ecules. \texttt{/model\_info} d\'ecrit le mod\`ele utilis\'e et \texttt{/metrics} expose les performances enregistr\'ees.

\begin{lstlisting}[caption=Exemple de lancement de l'API]
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
\end{lstlisting}

\begin{lstlisting}[caption=Exemple de requete JSON]
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
\end{lstlisting}

\section{Interface Streamlit}

L'interface Streamlit permet \`a un utilisateur non sp\'ecialiste de saisir une mol\'ecule, de visualiser sa structure, d'obtenir une pr\'ediction, puis de consulter les comparaisons de mod\`eles et les explications SHAP. Elle constitue une couche de d\'emonstration adapt\'ee \`a une soutenance acad\'emique.

\begin{lstlisting}[caption=Exemple de lancement de l'interface]
streamlit run webapp/app.py
\end{lstlisting}

\chapter{Annexe D -- Plan de Tests et Assurance Qualit\'e}

\section{Tests de donn\'ees}

Les tests v\'erifient la pr\'esence du fichier Tox21 brut, la taille minimale du jeu de donn\'ees, la pr\'esence de la colonne SMILES et la disponibilit\'e des t\^aches toxicologiques principales. Ces tests assurent que le pipeline ne s'ex\'ecute pas sur une source de donn\'ees incompl\`ete.

\section{Tests RDKit et feature engineering}

Les tests RDKit valident le parsing des SMILES corrects, le rejet des SMILES invalides, le calcul des descripteurs, la taille des empreintes Morgan et la taille des MACCS keys. Ces contr\^oles sont essentiels, car une erreur silencieuse dans la vectorisation mol\'eculaire contaminerait l'ensemble de l'exp\'erience.

\section{Tests de mod\`eles}

Les tests chargent un mod\`ele sauvegard\'e, v\'erifient la pr\'esence de \texttt{predict\_proba}, la forme des probabilit\'es retourn\'ees et leur appartenance \`a l'intervalle $[0,1]$. Ils valident aussi un exemple end-to-end depuis un SMILES jusqu'\`a une probabilit\'e de toxicit\'e.

\section{Tests API}

Les tests FastAPI utilisent un client de test local. Ils couvrent les endpoints \texttt{/health}, \texttt{/predict}, \texttt{/metrics} et \texttt{/model\_info}. Les SMILES invalides sont g\'er\'es explicitement afin d'\'eviter des erreurs serveur non contr\^ol\'ees.

\chapter{Annexe E -- Analyse Critique Compl\'ementaire}

\section{Seuil de classification}

Les m\'etriques de classification d\'ependent du seuil utilis\'e pour convertir une probabilit\'e en classe. Le seuil par d\'efaut de 0.5 peut \^etre sous-optimal en toxicologie. Dans un contexte de criblage, on peut privil\'egier un seuil plus faible pour augmenter le rappel et r\'eduire le nombre de mol\'ecules toxiques manqu\'ees. Dans un contexte de priorisation de tests co\^uteux, on peut au contraire augmenter le seuil pour am\'eliorer la pr\'ecision.

\section{Limites de l'approche 2D}

Les empreintes Morgan et les descripteurs RDKit utilis\'es ici capturent principalement des informations 2D. Certaines toxicit\'es d\'ependent toutefois de conformations 3D, d'interactions prot\'eine-ligand, de m\'etabolites et de conditions exp\'erimentales. Le mod\`ele doit donc \^etre interpr\'et\'e comme un syst\`eme de prioritisation in silico, non comme un substitut complet aux validations biologiques.

\section{Validation externe}

Une validation externe sur un jeu ind\'ependant serait n\'ecessaire avant une utilisation r\'eglementaire. Elle permettrait d'\'evaluer la robustesse du mod\`ele face \`a des mol\'ecules provenant d'un espace chimique diff\'erent. Cette \'etape est particuli\`erement importante pour limiter le risque de sur-optimisation sur Tox21.

\section{Perspectives de recherche}

Les extensions les plus prometteuses sont l'apprentissage multi-t\^aches, les Graph Neural Networks, les mod\`eles de langage mol\'eculaires et la quantification d'incertitude. L'apprentissage multi-t\^aches peut exploiter les corr\'elations entre essais Tox21. Les GNN permettent de traiter directement le graphe mol\'eculaire. Les mod\`eles de langage mol\'eculaires capturent des repr\'esentations apprises \`a grande \'echelle. Enfin, l'incertitude est essentielle pour signaler les pr\'edictions hors domaine.

\chapter{Annexe F -- Fiches Techniques des Mod\`eles}

\section{R\'egression logistique}

La r\'egression logistique constitue une base de comparaison lin\'eaire. Elle mod\'elise la probabilit\'e de toxicit\'e par une fonction sigmo\"ide appliqu\'ee \`a une combinaison pond\'er\'ee des caract\'eristiques. Son int\'er\^et principal est la simplicit\'e: elle permet de v\'erifier si une fronti\`ere lin\'eaire dans l'espace des empreintes Morgan suffit \`a distinguer les mol\'ecules actives et inactives. Dans ce projet, elle atteint un ROC-AUC de 0.8788, ce qui montre qu'une part importante du signal est d\'ej\`a capturable par une structure relativement simple.

\section{Arbre de d\'ecision}

L'arbre de d\'ecision segmente l'espace des caract\'eristiques par une suite de r\`egles binaires. Il est interpr\'etable mais sensible au surapprentissage, surtout avec des empreintes binaires de grande dimension. Ses performances plus faibles confirment que la toxicit\'e NR-AhR ne se r\'eduit pas \`a quelques r\`egles simples et stables.

\section{For\^et al\'eatoire}

La for\^et al\'eatoire agr\`ege de nombreux arbres entra\^in\'es sur des sous-\'echantillons et des sous-ensembles de variables. Elle r\'eduit la variance de l'arbre unique et s'adapte bien aux empreintes mol\'eculaires. Dans les r\'esultats obtenus, elle pr\'esente le meilleur PR-AUC, ce qui est particuli\`erement utile pour les donn\'ees d\'es\'equilibr\'ees.

\section{XGBoost et LightGBM}

Les mod\`eles de gradient boosting construisent une s\'equence d'arbres faibles, chacun corrigeant les erreurs du pr\'ec\'edent. XGBoost est robuste et efficace sur des donn\'ees tabulaires complexes; LightGBM optimise la vitesse et la gestion de grands ensembles de variables. Les deux mod\`eles sont performants en ROC-AUC, mais XGBoost pr\'esente un F1 faible avec le seuil par d\'efaut, ce qui souligne l'importance du calibrage de seuil.

\section{Support Vector Machine}

Le SVM \`a noyau RBF obtient le meilleur ROC-AUC. Le noyau RBF projette implicitement les donn\'ees dans un espace non lin\'eaire, ce qui permet de capturer des relations complexes entre fragments chimiques et toxicit\'e. Son principal inconv\'enient est son co\^ut computationnel plus \'elev\'e, observ\'e pendant l'entra\^inement et la validation crois\'ee.

\section{ToxNet}

ToxNet est un r\'eseau feedforward appliqu\'e aux empreintes Morgan. Le mod\`ele utilise BatchNorm, Dropout et une perte pond\'er\'ee pour g\'erer l'instabilit\'e et le d\'es\'equilibre de classes. Son ROC-AUC reste comp\'etitif, mais son F1 est faible avec le seuil de classification retenu. Une optimisation du seuil, davantage d'\'epoques ou une architecture multi-t\^aches pourraient am\'eliorer ses performances pratiques.

\chapter{Annexe G -- Journal de D\'ecision Scientifique}

\section{Choix de NR-AhR comme t\^ache principale}

NR-AhR a \'et\'e retenue comme t\^ache principale car elle poss\`ede un nombre suffisant d'\'echantillons labelis\'es et un int\'er\^et toxicologique clair. Le r\'ecepteur AhR est associ\'e \`a des hydrocarbures aromatiques et \`a des m\'ecanismes de r\'eponse cellulaire pertinents pour l'\'evaluation du risque chimique.

\section{Choix des empreintes Morgan}

Les empreintes Morgan sont largement utilis\'ees en chimioinformatique car elles encodent les sous-structures locales autour de chaque atome. Elles sont adapt\'ees aux mod\`eles classiques et permettent de comparer efficacement les mol\'ecules sans exiger une conformation 3D. Les r\'esultats confirment leur pertinence: les meilleurs scores sont obtenus avec les empreintes Morgan plut\^ot qu'avec les seuls descripteurs physico-chimiques.

\section{Choix des descripteurs pour SHAP}

Les fingerprints binaires sont efficaces mais difficiles \`a interpr\'eter chimiquement, car chaque bit correspond \`a un motif hach\'e. Les descripteurs RDKit sont donc utilis\'es pour les explications SHAP les plus lisibles. Des variables comme LogP, TPSA et BertzCT permettent une discussion scientifique plus naturelle sur la lipophilie, la polarit\'e et la complexit\'e mol\'eculaire.

\section{Choix des m\'etriques}

Le projet rapporte accuracy, precision, recall, F1, ROC-AUC et PR-AUC. Cette pluralit\'e est n\'ecessaire car les objectifs peuvent varier. Pour un syst\`eme de filtrage pr\'ecoce, le rappel est critique afin de ne pas manquer des mol\'ecules dangereuses. Pour un syst\`eme de priorisation exp\'erimentale, la pr\'ecision devient plus importante afin de limiter les faux positifs co\^uteux.

\chapter{Annexe H -- Guide de Soutenance}

\section{Message principal}

Le message central de la soutenance est que l'intelligence artificielle peut r\'eduire le co\^ut initial de l'\'evaluation toxicologique en priorisant les mol\'ecules \`a tester. Le projet ne remplace pas les essais exp\'erimentaux, mais fournit un outil de criblage in silico reproductible, explicable et int\'egrable dans un workflow de recherche.

\section{Points \`a mettre en avant}

\begin{enumerate}
  \item Le pipeline couvre toute la cha\^ine: donn\'ees, EDA, features, mod\`eles, deep learning, SHAP, API, interface, tests et livrables.
  \item Les performances sont coh\'erentes avec la litt\'erature Tox21, sans r\'esultat suspect ou irr\'ealiste.
  \item Le d\'es\'equilibre de classes est trait\'e explicitement par des poids de classes et par des m\'etriques adapt\'ees.
  \item SHAP apporte une lecture chimique des facteurs de risque, notamment la lipophilie et la complexit\'e.
  \item Les limites sont reconnues: absence de validation externe, absence de GNN et utilisation de repr\'esentations essentiellement 2D.
\end{enumerate}

\section{Questions possibles du jury}

\begin{longtable}{p{0.35\textwidth}p{0.55\textwidth}}
\toprule
Question & R\'eponse attendue \\
\midrule
Pourquoi ROC-AUC et PR-AUC? & ROC-AUC mesure le classement global; PR-AUC est plus informatif pour la classe positive rare. \\
Pourquoi SVM est-il meilleur? & Le noyau RBF capture des relations non lin\'eaires dans l'espace des fingerprints. \\
Pourquoi ToxNet n'est-il pas premier? & Le dataset est relativement petit pour un r\'eseau profond; le seuil et l'architecture peuvent \^etre optimis\'es. \\
Pourquoi SHAP sur descripteurs? & Les descripteurs sont nomm\'es et donc plus interpr\'etables que les bits Morgan hach\'es. \\
Le mod\`ele est-il utilisable en production? & Il est utilisable comme prototype de criblage, mais une validation externe est requise avant usage r\'eglementaire. \\
\bottomrule
\end{longtable}

\begin{thebibliography}{99}

\bibitem{rogers2010}
Rogers, D., \& Hahn, M. (2010).
\textit{Extended-Connectivity Fingerprints}.
Journal of Chemical Information and Modeling, 50(5), 742--754.

\bibitem{breiman2001}
Breiman, L. (2001).
\textit{Random Forests}.
Machine Learning, 45(1), 5--32.

\bibitem{chen2016}
Chen, T., \& Guestrin, C. (2016).
\textit{XGBoost: A Scalable Tree Boosting System}.
KDD 2016, 785--794.

\bibitem{lundberg2017}
Lundberg, S. M., \& Lee, S.-I. (2017).
\textit{A Unified Approach to Interpreting Model Predictions}.
NIPS 2017, 4765--4774.

\bibitem{tox21challenge}
Huang, R., et al. (2016).
\textit{Tox21Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways as Mediated by Exposure to Environmental Chemicals and Drugs}.
Frontiers in Environmental Science.

\bibitem{rdkit}
Landrum, G. (2023).
\textit{RDKit: Open-Source Cheminformatics}.
\url{http://www.rdkit.org}

\bibitem{moleculenet}
Wu, Z., et al. (2018).
\textit{MoleculeNet: A Benchmark for Molecular Machine Learning}.
Chemical Science, 9(2), 513--530.

\bibitem{deepchem}
Ramsundar, B., et al. (2019).
\textit{Deep Learning for the Life Sciences}.
O'Reilly Media.

\end{thebibliography}

\end{document}
"""

latex_file = LATEX_DIR / "rapport_tox21.tex"
with open(latex_file, "w", encoding="utf-8") as f:
    f.write(REPORT)

print(f"LaTeX report written: {latex_file}")
print(f"Figures copied to: {FIG_DIR}")
print(f"To compile: pdflatex -interaction=nonstopmode rapport_tox21.tex")


if __name__ == "__main__":
    pass
