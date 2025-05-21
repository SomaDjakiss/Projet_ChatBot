import os
import re
import pandas as pd
import streamlit as st
from collections import defaultdict
from itertools import islice
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI



st.set_page_config(page_title="🎓 Analyse Scolaire", layout="centered")
st.title("🎓 Chatbot Scolaire - Analyse des Performances")

# ✅ Chargement et cache des données
df=pd.read_csv("https://raw.githubusercontent.com/SomaDjakiss/Projet_ChatBot_Kix_Seeds/main/data_kix_seeds.csv",encoding="ISO-8859-1",sep=";")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# ✅ Chargement de la clé API OpenAI depuis les secrets Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ✅ Configuration du modèle GPT
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    openai_api_key=openai_api_key
)
# Prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "donnees"],
    template="""
Tu es un expert en analyse pédagogique, conçu pour fournir des réponses précises, structurées et basées sur des données scolaires.
Voici des données sur les performances scolaires d'élèves. Utilise ces informations pour réaliser une analyse pédagogique approfondie selon le type de demande.

## ANALYSE AU NIVEAU ÉLÈVE
Si la question concerne un élève spécifique (par nom, prénom ou ID) :

### Profil de l'élève
- Informations personnelles : nom_eleve, prenom_eleve, nom_complet_eleve, date_naissance_eleve, lieu_naissance_eleve, genre_eleve (1:masculin, 2:féminin)
- Statut: est_redoublant, statut_eleve (vérifie si PDI statut_eleve =2), eleve_a_handicap
- Contexte familial: vit_avec_parents, vit_au_domicile_parents, vit_avec_tuteur, profession_pere, profession_mere, niveau_education_pere, niveau_education_mere
### Performances académiques
- Notes et moyennes : notes_matieres, moyenne_t1, moyenne_t2, moyenne_t3, moyenne_annuelle_t1, moyenne_annuelle_t2, moyenne_annuelle_t3
- Classement : rang_t1, rang_t2, rang_t3, rang_annuel_t1, rang_annuel_t2, rang_annuel_t3
- Progression : Analyse l'évolution entre les trimestres (amélioration, détérioration, stabilité)
- Comparaison avec la classe : Positionne l'élève par rapport à moyenne_classe_t1, moyenne_classe_t2, moyenne_classe_t3
- Matières : Identifie les forces (notes ≥ 7/10) et faiblesses (notes < 5/10)
### Assiduité et comportement
- Présence : type_presence (présent, absent, retard), motif_absence, date_debut_absence, date_fin_absence
- Conduite : appreciation_conduite_t1, appreciation_conduite_t2, conduite_label_t3, sanction_disciplinaire_t1, sanction_disciplinaire_t2, sanction_t3
- Appréciations : appreciation_enseignant_t1, appreciation_enseignant_t2, appreciation_t3
###Contexte de vie et bien-être
- Logistique scolaire : distance_domicile, mode_transport, residence_eleve
- Équipement éducatif : possede_bureau, possede_livres, possede_tableaux, possede_tablette, possede_autres_materiels
- Contexte familial : menage_a_television, menage_a_radio, menage_a_internet, menage_a_electricite
- Bien-être et sécurité : dort_sous_moustiquaire, victime_violence, victime_violence_physique, victime_stigmatisation, victime_violence_sexuelle, victime_violence_emotionnelle, victime_autre_violence
### Recommandations personnalisées
- Soutien académique : Propose des stratégies d'amélioration pour les matières faibles
- Soutien socio-éducatif : Conseils adaptés aux conditions de vie et au contexte familial
- Suivi spécifique : Si élève vulnérable (PDI, handicap, victime de violence), propose un accompagnement adapté

## ANALYSE AU NIVEAU CLASSE
Si la question concerne une classe spécifique :
### Profil de la classe
- Informations générales : nom_salle_classe, effectif_classe_t1, effectif_classe_t2, effectif_t3
- Composition : Répartition par genre_eleve (1:masculin, 2:féminin)
- Statuts particuliers : Nombre d'élèves est_redoublant, PDI (statut_eleve=2), eleve_a_handicap
### Performances globales
- Moyennes de la classe : moyenne_classe_t1, moyenne_classe_t2, moyenne_classe_t3
- Dispersion : Écart entre moyenne_la_plus_elevee_t1/t2 et moyenne_la_plus_basse_t1/t2, max_moyenne_t3 et min_moyenne_t3
- Taux de réussite : Pourcentage d'élèves avec moyenne (moyenne_t1, moyenne_t2, moyenne_t3) ≥ 5/10, analyse par genre genre_eleve
- Progression : Évolution des résultats entre les trimestres
- Analyse par matière : Matières avec meilleurs et moins bons résultats
### Assiduité et comportement
- Présence : Statistiques sur type_presence (présences, absences, retards)
- Motifs d'absence : Analyse des motif_absence les plus fréquents
- Abandons : Analyse des date_abandon si existantes
### Analyse comparative
- Par genre : Compare les performances moyennes_t1/t2/t3 selon le genre_eleve
- Par statut : Compare les performances des élèves ordinaires vs PDI vs avec handicap
- Par contexte familial : Analyse l'impact des conditions familiales sur les résultats
### Recommandations pédagogiques
- Renforcement : Stratégies pour consolider les acquis dans les matières réussies
- Remédiation : Approches pour améliorer les résultats dans les matières faibles
- Accompagnement : Mesures pour soutenir les élèves en difficulté
- Dynamique de classe : Suggestions pour améliorer la cohésion et l'environnement d'apprentissage

## ANALYSE AU NIVEAU ÉCOLE
Si la question concerne une école spécifique :
### Profil de l'établissement
- Informations générales : nom_ecole, code_ecole, type_ecole, statut_ecole, milieu_ecole (urbain/rural)
- Localisation : region_ecole, province_ecole, commune_ecole, ceb_ecole, secteur_village_ecole
- Administration : nom_complet_directeur, sexe_directeur, poste_directeur, responsabilites_directeur
- Structure : Nombre total d'élèves, nombre total d'enseignants, répartition par genre
### Infrastructure et équipement
- Bâtiments : Nombre de salles de classe, état des infrastructures
- Équipements essentiels : Présence de cantine, latrines/toilettes/WC, fontaine/pompe/eau potable, électricité
- Ressources pédagogiques : Disponibilité de matériels didactiques
### Performances par classe
- Moyennes : moyenne_classe_t1, moyenne_classe_t2, moyenne_classe_t3 pour chaque nom_salle_classe
- Taux de réussite : Pourcentage d'élèves avec moyenne ≥ 5/10 par nom_salle_classe et par genre_eleve  genre 
- Progression : Évolution des résultats entre les trimestres par classe
- Analyse comparative : Classement des classes selon leurs performances
### Statistiques socio-éducatives
- Présence*: Statistiques globales sur type_presence (présences, absences, retards)
- Statuts particuliers : Proportion de PDI (statut_eleve=2), élèves avec handicap
- Bien-être : Cas signalés de violence (victime_violence, types de violences)
### Recommandations institutionnelles
- Gestion : Suggestions pour l'amélioration de la gouvernance scolaire
- Pédagogie : Stratégies pour renforcer la qualité de l'enseignement
- Équité : Mesures pour réduire les disparités de performance
- Bien-être : Actions pour améliorer l'environnement scolaire et la sécurité

## ANALYSE AU NIVEAU CEB OU COMMUNE
Si la question concerne une CEB ou une commune :
### Cartographie éducative
- Structure : Nombre d'écoles dans la CEB/commune, répartition par type_ecole et statut_ecole
- Personnel : Nombre total d'enseignants, répartition par sexe_directeur et genre des enseignants
- Population scolaire : Nombre total d'élèves, répartition par genre_eleve
- Ratios : Élèves/enseignant par école, élèves/classe
### Infrastructure territoriale
- Équipements essentiels : Proportion d'écoles avec/sans cantine, latrines, eau potable, électricité
- Accessibilité : Analyse des distance_domicile et mode_transport dominants
- Ressources : Disponibilité et distribution des matériels didactiques
### Performances comparatives
- Moyennes : Classement des écoles selon moyenne_classe_t1, moyenne_classe_t2, moyenne_classe_t3
- Taux de réussite : Comparaison du pourcentage d'élèves avec moyenne ≥ 5/10 par école
- Disparités : Identification des écarts de performance significatifs
- Facteurs explicatifs : Analyse des corrélations entre performances et facteurs contextuels
### Vulnérabilités et inclusion
- Populations spécifiques : Nombre de PDI (statut_eleve=2), élèves avec handicap par école
- Violences et protection : Cartographie des signalements de victime_violence et types
- Abandons : Analyse comparative des taux d'abandon par école
### Recommandations territoriales
- Planification : Stratégies pour une meilleure répartition des ressources
- Formation : Besoins en renforcement des capacités des enseignants
- Infrastructures : Priorités d'investissement dans les équipements essentiels
- Protection : Mesures coordonnées pour améliorer la sécurité des élèves

## DIRECTIVES GÉNÉRALES
### Format de réponse
- Structure claire : Utilise des titres, sous-titres et listes pour organiser l'information
- Visualisation : Propose des tableaux synthétiques pour les données comparatives
- Progressivité : Commence par les constats, puis analyse, puis recommandations
- Concision : Privilégie la pertinence à l'exhaustivité
### Méthodologie d'analyse
- Objectivité : Base toutes les affirmations sur les données disponibles
- Prudence : Signale clairement les données manquantes ou incomplètes
- Contextualisation : Tiens compte des spécificités locales (milieu_ecole, etc.)
- Équité : Analyse systématiquement les disparités de genre et les vulnérabilités
### Recommandations
- Pragmatisme : Propose des solutions réalistes et adaptées au contexte
- Progressivité : Distingue les actions à court, moyen et long terme
- Responsabilisation : Identifie les acteurs concernés par chaque recommandation
- Inclusivité : Veille à l'adaptation des recommandations aux besoins spécifiques

**Ne jamais inventer de données**. Si les données sont manquantes, indique-le clairement.


Question : {question}

Données :
{donnees}

Fais une réponse claire et structurée.
"""
)

def extraire_filtre(question, valeurs_connues):
    for val in valeurs_connues:
        if val and str(val).lower() in question.lower():
            return val
    return None

def get_response_from_dataframe(question, df):
    from functools import reduce
    import operator
    reponses = []

    question_lower = question.lower()

    # Recherche des filtres possibles
    id_eleve = extraire_filtre(question_lower, df['id_eleve'].astype(str).unique())
    identifiant_unique = extraire_filtre(question_lower, df['identifiant_unique_eleve'].astype(str).unique())
    id_classe = extraire_filtre(question_lower, df['id_classe'].astype(str).unique())
    code_classe = extraire_filtre(question_lower, df['code_classe'].astype(str).unique())
    nom_classe = extraire_filtre(question_lower, df['nom_classe'].astype(str).unique())
    nom_ecole = extraire_filtre(question_lower, df['nom_ecole'].astype(str).unique())
    code_ecole = extraire_filtre(question_lower, df['code_ecole'].astype(str).unique())
    ceb = extraire_filtre(question_lower, df['ceb_ecole'].astype(str).unique())
    commune = extraire_filtre(question_lower, df['commune_ecole'].astype(str).unique())
    
    # 🔍 Élève
    if id_eleve or identifiant_unique:
        ident = id_eleve or identifiant_unique
        ligne = df[(df['id_eleve'].astype(str) == ident) | (df['identifiant_unique_eleve'].astype(str) == ident)]
        if not ligne.empty:
            ligne = ligne.iloc[0]
            donnees_texte = "\n".join([f"{col} : {ligne[col]}" for col in df.columns if col in ligne])
            prompt = prompt_template.format(question=question, donnees=donnees_texte)
            resultat = llm.invoke(prompt)
            return resultat.content if hasattr(resultat, 'content') else resultat

    # 🔍 Classe / école
    filtres = []
    if nom_ecole: filtres.append(df['nom_ecole'].str.lower() == nom_ecole.lower())
    if code_ecole: filtres.append(df['code_ecole'].astype(str) == str(code_ecole))
    if ceb: filtres.append(df['ceb_ecole'].astype(str) == str(ceb))
    if commune: filtres.append(df['commune_ecole'].astype(str) == str(commune))
    if code_classe: filtres.append(df['code_classe'].astype(str) == str(code_classe))
    if nom_classe: filtres.append(df['nom_classe'].str.lower() == nom_classe.lower())
    if id_classe: filtres.append(df['id_classe'].astype(str) == str(id_classe))
    

    if filtres:
        condition = reduce(operator.and_, filtres)
        df_filtre = df[condition]
        if df_filtre.empty:
            return "Aucune donnée trouvée avec les critères spécifiés."

        # Fixer automatiquement le nombre d’élèves dans la classe /On évite d’afficher tous les élèves si ce n’est pas explicitement demandé.
        nb_eleves = df_filtre.shape[0]

        # 🎯 Analyse par classe
        if "classe" in question_lower or "classes" in question_lower:
            classes = df_filtre['nom_classe'].unique()
            for classe in classes:
                df_classe = df_filtre[df_filtre['nom_classe'] == classe]
                resume = {col: df_classe[col].mean() for col in df_classe.columns if df_classe[col].dtype != 'object'}
                donnees_texte = f"Classe : {classe}\n" + "\n".join([f"{k} : {v:.2f}" for k, v in resume.items()])
                prompt = prompt_template.format(question=question, donnees=donnees_texte)
                resultat = llm.invoke(prompt)
                if hasattr(resultat, 'content'):
                    resultat = resultat.content
                reponses.append(f"Classe {classe} :\n{resultat}")
            return "\n\n---\n\n".join(reponses)

        # 🎯 Analyse globale de l’école
        elif "école" in question_lower or "ecole" in question_lower or "établissement" in question_lower:
            resume = {col: df_filtre[col].mean() for col in df_filtre.columns if df_filtre[col].dtype != 'object'}
            donnees_texte = f"Ecole : {df_filtre['nom_ecole'].iloc[0]}\n" + "\n".join([f"{k} : {v:.2f}" for k, v in resume.items()])
            prompt = prompt_template.format(question=question, donnees=donnees_texte)
            resultat = llm.invoke(prompt)
            return resultat.content if hasattr(resultat, 'content') else resultat

        # 🎯 Si CEB ou commune
        elif "ceb" in question_lower or "commune" in question_lower:
            resume = df_filtre.groupby("nom_ecole").mean(numeric_only=True)
            donnees_texte = resume.round(2).to_string()
            prompt = prompt_template.format(question=question, donnees=donnees_texte)
            resultat = llm.invoke(prompt)
            return resultat.content if hasattr(resultat, 'content') else resultat

        # 🔄 Sinon (traitement classe sans mention explicite) : résumé sans nommer les élèves
        resume = {col: df_filtre[col].mean() for col in df_filtre.columns if df_filtre[col].dtype != 'object'}
        donnees_texte = "Résumé global :\n" + "\n".join([f"{k} : {v:.2f}" for k, v in resume.items()])
        prompt = prompt_template.format(question=question, donnees=donnees_texte)
        resultat = llm.invoke(prompt)
        return resultat.content if hasattr(resultat, 'content') else resultat

    return "Aucun filtre détecté dans la question. Veuillez spécifier un élève, une classe ou une école."
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#  Formulaire avec champ texte et bouton Submit
with st.form("formulaire_question"):
    user_input = st.text_input("Pose ta question sur un élève, une école ou une classe")
    submitted = st.form_submit_button("Submit")

#  Traitement après envoi
if submitted and user_input:
    response = get_response_from_dataframe(user_input, df)
    st.session_state.chat_history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ])

# Affichage de l’historique
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
