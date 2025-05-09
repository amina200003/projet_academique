# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:57:08 2024

@author: amina
"""
#import glob
import glob
import spacy
import time
import matplotlib.pyplot as plt
import numpy as np
import re 
import json
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")


#fonction
def lirefich(chemin):#lire un fichier
    with open(chemin,encoding="utf-8") as r:
        chaine= r.read()
    return chaine

def tokenisation_spacy(txt_spacy):
    debut= time.time()
    token=[]
    for mot in txt_spacy:
        token.append(mot.text)
    print("les mot ont été récupéré")
    fin= time.time()
    duree= fin- debut
    print("la tokenisation avec spacy à duré:", duree)
    return token

def lemmatisation(txt_spacy):
    lemmes=[]
    for word in txt_spacy:
        lemmes.append(word.lemma_)
    return lemmes

def entite_nomme(txt_spacy):
    ent=[]
    for w in txt_spacy.ents:
        ent.append(w.text)
    return ent

def postag_vocab(txt_spacy):
    debut= time.time()
    postagging=[]
    for m in txt_spacy:
        postagging.append([m.text,m.pos_,m.is_stop])
    vocabulaire= vocab(postagging)
    fin=time.time()
    duree= fin-debut
    print("le processus de generation du vocab manuelle dure:", duree)
    return vocabulaire

def vocab(liste_pos): #genere un vocabulaire en excluant les stop words, punct, nom propre et space
    voc=[]
    for mini_liste in liste_pos:
            if mini_liste[1]!= "PROPN":
                if mini_liste[1] != "SPACE":
                    if mini_liste[1] != "PUNCT":
                        if mini_liste[2]!= True:
                           voc.append(mini_liste[0])
    return voc

def vocab_nltk(txt, language):
    debut= time.time()
    liste= tokenisation_spacy(txt)
    stop_words = set(stopwords.words(f"{language}"))
    list_filtre = [mot for mot in liste if mot.lower() not in stop_words]
    fin= time.time()
    duree= fin-debut
    print("trouver le vocabulaire automatique prends:", duree)
    return list_filtre

def nettoyer_texte(texte):
    # Supprimer les signes de ponctuation
    texte_sans_ponctuation = re.sub(r'[^\w\s]', '', texte)
    # Supprimer les chiffres
    texte_sans_chiffres = re.sub(r'\d', '', texte_sans_ponctuation)
    # Supprimer les liens
    # texte_sans_liens = re.sub(r'^http.*?\s', '', texte_sans_chiffres)
    # Supprimer les parenthèses (et le contenu entre elles si nécessaire)
    texte_nettoye = re.sub(r'\(.*?\)', '', texte_sans_chiffres)
    return texte_nettoye

def nettoyer_liste(liste):
    liste_propre = [nettoyer_texte(mot) for mot in liste if not mot.startswith('http')]
    return [element for element in liste_propre if element]

def sauvegarder_json(liste, nom_liste, langue):
    nom_fichier = f"{nom_liste}_{langue}.json"
    with open(nom_fichier, "w", encoding="utf-8") as json_file:
        json.dump(liste, json_file, ensure_ascii=False, indent=2)
    print(f"Vocabulaire sauvegardé dans {nom_fichier} avec succès !")

########MAIN

liste_modele=["fr_core_news_sm","en_core_web_sm","es_core_news_sm"]
nlp= spacy.load(liste_modele[0])


#TRAITEMENT DES CORPUS
for path in glob.glob("corpus_multi2/fr/*"): #echantillon de 20 txt pour tester
    all_txt=[]
    txt= lirefich(path)
    all_txt.append(txt)
    txt_string= " ".join(all_txt) 
    
#fr_split= txt_string.split()


#tokenisation spacy

texte_spacy= nlp(txt_string) 
#token_spacyfr= tokenisation_spacy(texte_spacy) #0 sec
#token_spacy_en= tokenisation_spacy(texte_spacy) # 0 sec
#token_spacy_es= tokenisation_spacy(texte_spacy) #0 sec

#lemmatisation
#lemma_fr= lemmatisation(texte_spacy) 
#lemma_en=lemmatisation(texte_spacy)
#lemma_es= lemmatisation(texte_spacy)

#detection ENT
ent_fr= entite_nomme(texte_spacy)
#ent_en= entite_nomme(texte_spacy)
#ent_es= entite_nomme(texte_spacy)

#POS TAGGING ET REALISATION DU VOCABULAIRE A PARTIR DU POST TAGGING (manuel)
#postagging_vocab_fr= postag_vocab(texte_spacy) # 0,015 sec 
#postagging_vocab_en= postag_vocab(texte_spacy) #0,015 sec
#postagging_vocab_es= postag_vocab(texte_spacy) #à,015 sec

#REALISATION DU VOCABULAIRE AVEC NLTK (AUTOMATIQUE)
#vocab_auto_fr= vocab_nltk(texte_spacy,"french") #0,015 sec
#vocab_auto_en= vocab_nltk(texte_spacy,"english") #0,015 sec
#vocab_auto_es= vocab_nltk(texte_spacy,"spanish") #0,015 sec

#sauvegarder les vocabulaires en json
#sauvegarder_json(postagging_vocab_es, "vocab_manuel", "espagnol")
#sauvegarder_json(vocab_auto_es,"vocab_auto","espagnol")
#sauvegarder_json(postagging_vocab_en, "vocab_manuel", "anglais")
#sauvegarder_json(vocab_auto_en,"vocab_auto", "anglais")
#sauvegarder_json(postagging_vocab_fr, "vocab_manuel","français")
#sauvegarder_json(vocab_auto_fr, "vocab_auto","français" )

 
#REPRESENTATION GRAPHIQUE

#comparaison tokenisation spacy et split
taille_liste=[17203,22408]   
label= ["split","spacy"]
fig1, ax1= plt.subplots()
x2= np.arange(len(label))
width= 0.1
graph1= plt.bar(range(len(taille_liste)),taille_liste)
plt.bar_label(graph1, padding= 3)
graph1[0].set_color("red")
graph1[1].set_color("green")

plt.title("comparaison entre split et tokenisation spacy")

ax1.set_xticks(x2+width, label)
#plt.savefig("comparer split et token spacy")
#plt.show()

#token
width= 0.1 #specifie la largeur de chaque bar
token_langues=[22408,21306,18052] #donnee a representer dans notre graphique
langue=["fr","es","en"] #label qui corresponds à chaque donnee ci-dessus
x2= np.arange(len(langue)) # permet de representer le graphique correctement en associant les donnees avec le label qui le corresponds
fig2,ax2= plt.subplots() #doit etre redefini sinon les graph se superposent

graph2= plt.bar(range(len(token_langues)),token_langues) #dessine le graphique 
plt.bar_label(graph2, padding=3) #permet d'ajouter des label au graphique
graph2[0].set_color("blue")
graph2[1].set_color("pink")
graph2[2].set_color("purple")

plt.title("nombre de tokens pour chaque langue")

ax2.set_xticks(x2+width, langue) # place correctement le label par rapport à la bar qui le corresponds

#plt.savefig("nbr tokens langues")
#plt.show()

#lemmatisation
lemmes_langues=[22408,21306,18052]
fig3, ax3= plt.subplots()

graph3= plt.bar(range(len(lemmes_langues)),lemmes_langues)
plt.bar_label(graph3, padding= 3)
graph3[0].set_color("red")
graph3[1].set_color("green")
graph3[2].set_color("orange")

plt.title("nombre de lemmes pour chaque langue")

ax3.set_xticks(x2+width, langue)

#plt.show()
#plt.savefig("nbr lemme langues")

#ent
ent_langues=[1062,1385,1500]
fig4, ax4= plt.subplots()

graph4= plt.bar(range(len(ent_langues)),ent_langues)
plt.bar_label(graph4, padding=3)
graph4[0].set_color("yellow")
graph4[1].set_color("cyan")
graph4[2].set_color("pink")

plt.title("nombre de ent pour chaque langues")

ax4.set_xticks(x2+width, langue)
#plt.show()
#plt.savefig("nbr ent langues")

#comparaison de la taille des vocab manuel et automatique
width= 0.1
vocab_langues=[12484,6895,17362,7033,15742,8744]
type_vocab=["en_nltk","en_manuel","es_nltk","es_manuel","fr_nltk","fr_manuel"]
x3= np.arange(len(type_vocab))
fig5,ax5= plt.subplots() #doit etre redefini sinon les graph se superposent

graph5= plt.bar(range(len(vocab_langues)),vocab_langues)
plt.bar_label(graph5, padding=3)
graph5[0].set_color("yellow")
graph5[1].set_color("yellow")
graph5[2].set_color("blue")
graph5[3].set_color("blue")
graph5[4].set_color("orange")
graph5[5].set_color("orange")

plt.title("comparaison des vocabulaires")

ax5.set_xticks(x3+width,type_vocab)

#plt.savefig("comparer vocab")
#plt.show()
 

    
    
    
    
