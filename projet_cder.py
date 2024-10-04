import subprocess
import os
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def xfoil_f(airfoil, M, aoa_list, Re, Ncrit):
    input_file = 'xfoil.inp'
    polar_file = 'currentpolar.pol'

    # Préparez le fichier d'entrée pour XFOIL
    with open(input_file, 'w') as f:
        f.write(f"LOAD {airfoil}.txt\n")   #ecrie toutes les commandes nécessaire pour exécuter xfoil
        f.write("PANE\n")
        f.write("OPER\n")
        f.write("ITER 250\n")
        f.write(f"VISC {Re}\n")
        f.write(f"Mach {M}\n")
        f.write("PACC\n")
        f.write(f"{polar_file}\n\n")
        for aoa in aoa_list:
            f.write(f"A {aoa}\n")
        f.write("PACC\n")
        f.write("QUIT\n")
    
    # Supprimez tout fichier polaire existant
    try:
        os.remove(polar_file)
    except OSError:
        pass
    
    # Exécutez XFOIL et capturez la sortie
    result = subprocess.run(['xfoil.exe', '<', input_file], shell=True, capture_output=True, text=True) #éxecuter xfoil avec les commandes sur input_file
    
    # Vérifiez si le fichier polar a été créé
    if not os.path.isfile(polar_file):
        raise FileNotFoundError(f"XFOIL n'a pas créé le fichier {polar_file}")
    
    # Lisez les résultats
    with open(polar_file, 'r') as f:
        lines = f.readlines()    # lines est une liste qui contient une ligne dans chaque colonne 
    
    # Extraire les données en ignorant les 12 premières lignes
    data = []
    for line in lines[12:]:
        parts = line.split()
        if len(parts) >= 3:
            AoA = float(parts[0])
            Cl = float(parts[1])
            Cd = float(parts[2])
            data.append((AoA, Cl, Cd))
    
    return data        #data c'est une liste qui contient aoa,cl,cd 

def viterna_method(polar_data, AR):
    # Applique l'extrapolation de Viterna pour des angles d'attaque élevés
    polar_data = np.array(polar_data)
    AoA = polar_data[:, 0]
    Cl = polar_data[:, 1]
    Cd = polar_data[:, 2]

    CDmax = 1.11 + 0.018 * AR
    alpha_stall = AoA[-1]
    CLstall = Cl[-1]
    CDstall = Cd[-1]
    A2 = (CLstall - CDmax * np.sin(np.radians(alpha_stall)) * np.cos(np.radians(alpha_stall))) * np.sin(np.radians(alpha_stall)) / np.cos(np.radians(alpha_stall))**2
    B2 = (CDstall - CDmax * np.sin(np.radians(alpha_stall))**2) / np.cos(np.radians(alpha_stall))

    # Extrapolez pour AoA de 20.25 à 90 degrés
    AoA1 = np.arange(20.25, 90.25, 0.25)
    CL1 = CDmax / 2 * np.sin(np.radians(2 * AoA1)) + A2 * np.cos(np.radians(AoA1))**2 / np.sin(np.radians(AoA1))
    CD1 = CDmax * np.sin(np.radians(AoA1))**2 + B2 * np.cos(np.radians(AoA1))

    # Extrapolez pour AoA de 90.25 à 160 degrés
    AoA2 = np.arange(90.25, 160.25, 0.25)
    CL2 = -0.7 * (CDmax / 2 * np.sin(np.radians(2 * (180 - AoA2))) + A2 * np.cos(np.radians(180 - AoA2))**2 / np.sin(np.radians(180 - AoA2)))
    CD2 = CDmax * np.sin(np.radians(180 - AoA2))**2 + B2 * np.cos(np.radians(180 - AoA2))

    # Inclure les valeurs pour 170 et 180 degrés
    CL3 = (CL2[-1] + 0) / 2
    CD3 = (CD2[-1] + Cd[0]) / 2

    # Combinez tous les AoA et coefficients
    AoAnew = np.concatenate((AoA, AoA1, AoA2, [170, 180]))
    Clnew = np.concatenate((Cl, CL1, CL2, [CL3, 0]))
    Cdnew = np.concatenate((Cd, CD1, CD2, [CD3, Cd[0]]))

    # Créez des splines pour extrapolation
    clspline = UnivariateSpline(AoAnew, Clnew, s=0)
    cdspline = UnivariateSpline(AoAnew, Cdnew, s=0)

    # Extrapolez à -180 degrés
    AoA5 = np.arange(-180, -10.25, 0.25)
    CL5 = -0.7 * clspline(abs(AoA5))
    CD5 = cdspline(abs(AoA5))

    # Assemblez les données polaires finales
    AoAfinal = np.concatenate((AoA5, AoAnew))
    Clfinal = np.concatenate((CL5, Clnew))
    Cdfinal = np.concatenate((CD5, Cdnew))

    return AoAfinal, Clfinal, Cdfinal

def noms_airfoils(fichier, afficher=700):
    with open(fichier, 'r') as f:
        airfoils = f.read().splitlines()   #airfoils c'est une liste qui contient les noms qui se trouve dans le fichier airfoils.txt

    def afficher_airfoils(start, end):
        print("Liste des profils disponibles :")
        for i in range(start, min(end, len(airfoils))):
            print(f"{i + 1}. {airfoils[i]}")
        print("\n")
    
    def demander_plus():
        reponse = input("Voulez-vous voir plus de profils ? (o/n) ").strip().lower() 
        return reponse == 'o'

    start = 0
    while start < len(airfoils):
        end = start + afficher
        afficher_airfoils(start, end)
        start = end
        if start < len(airfoils) and demander_plus():
            continue
        else:
            break

    print("\n")
    selectionner = False
    while not selectionner:
        try:
            choix = int(input("Veuillez entrer le numéro du profil aérodynamique : "))
            if 1 <= choix <= len(airfoils):
                selectionner = True
                return airfoils[choix - 1]
            else:
                print("Numéro invalide, veuillez réessayer.")
        except ValueError:
            print("Entrée invalide, veuillez entrer un numéro.")

# Code principal
airfoil = noms_airfoils('airfoils.txt')

M = 0.05  # Nombre de Mach
aoa_list = range(-10, 21)  # Liste des angles d'attaque de -10 à 20
nombre_re = int(input('Combien de nombre de Reynolds avez-vous ? '))
valeurs = []
for i in range(nombre_re):
    REi = int(input(f'Veuillez entrer le nombre de Reynolds RE{i+1} : '))
    valeurs.append(REi)

AR = 6  # Aspect Ratio 

# Initialiser des listes pour stocker les résultats
results_cl = []
results_cd = []
aoa_results = None

for Re in valeurs:
    # Exécutez Xfoil
    polar_data = xfoil_f(airfoil, M, aoa_list, Re, Ncrit=9)
    if polar_data:
        # Appliquez l'extrapolation de Viterna
        AoAfinal, Clfinal, Cdfinal = viterna_method(polar_data, AR)
        # Enregistrez les données dans un fichier
        output_file = f"{airfoil}_{Re}.txt"
        with open(output_file, 'w') as f:
            f.write('AoA Cl Cd\n')
            for aoa, cl, cd in zip(AoAfinal, Clfinal, Cdfinal):
                f.write(f"{aoa} {cl} {cd}\n")
        
        print(f"Résultats enregistrés dans '{output_file}'.")
    else:
        print("Aucune donnée polaire récupérée.")
        
    results_cl.append((Re, AoAfinal, Clfinal))
    results_cd.append((Re, AoAfinal, Cdfinal))
    aoa_results = AoAfinal  # Toutes les AoA devraient être les mêmes, donc on peut réutiliser

# Tracer les graphiques
plt.figure(figsize=(12, 6))

# Graphique Cl en fonction de AoA
plt.subplot(1, 2, 1)
for Re, AoAfinal, Clfinal in results_cl:
    plt.plot(AoAfinal, Clfinal, label=f'Re = {Re}')
plt.xlabel('Angle of Attack (AoA)')
plt.ylabel('Lift Coefficient (Cl)')
plt.title('Cl vs AoA for different Reynolds numbers')
plt.legend()

# Graphique Cd en fonction de AoA
plt.subplot(1, 2, 2)
for Re, AoAfinal, Cdfinal in results_cd:
    plt.plot(AoAfinal, Cdfinal, label=f'Re = {Re}')
plt.xlabel('Angle of Attack (AoA)')
plt.ylabel('Drag Coefficient (Cd)')
plt.title('Cd vs AoA for different Reynolds numbers')
plt.legend()

plt.tight_layout()
plt.show()


# Fin du script principal

# Auteur: KASSAB Mohamed Amine
# Etudiant en premiére année à l'école nationale superieure d'informatique (ESI)
# Dernière modification: 30 juillet 2024
# Instructions: Exécuter ce script en utilisant Python 3.8 ou une version ultérieure.
#N.B: veuillez mettre tout les fichier qui sont sous la forme zip dans le meme répertoire pour l'execution du code  
# Références: 
# - Documentation de Python: https://docs.python.org/3/
# - Tutoriel XFOIL: [https://youtu.be/1BFq8HC-7S4?list=PL67TsbOhn1ThGHe6AefBzOI-6WTmTxpkK&t=5]