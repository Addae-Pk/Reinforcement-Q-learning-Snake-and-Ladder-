#Installating Python modules
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import random as rd

#####################################

#Construction of transition probability matrixes for each dice
#Dice 1 : Safe Dice
def Safe_Matrix(nbr_steps):
    # nbr_steps : Number of squares in the game

    #The goal of this function is to return a transition probability matrix for the safe dice

    Safe_Proba = np.zeros((nbr_steps,nbr_steps))

    for step in range(len(Safe_Proba)):
        #50% probability => Stay on the same place | Only 0 or 1 is possible with the safe dice
        Safe_Proba[step][step] = 1/2

        #On step 3, 25% probability => Go to 4 or 25% probability => Go to 11 | The construction of the game imposes this rule
        if step == 2: 
            Safe_Proba[step][step+1] = (1/2) * (1/2)
            Safe_Proba[step][10] = (1/2) * (1/2)
        
        #On step 10, 50% probability => Go to the final step | The construction of the game imposes this rule
        elif step == 9:
            Safe_Proba[step][14] = 1/2

        #On the final step, 100% probability => Stay on the final step | When we reach the final step, the game is over
        elif step == 14:
             Safe_Proba[step][step] = 1

        #On all the other cases, the game depends on the dice
        else:
            Safe_Proba[step][step+1] = 1/2

    return pd.DataFrame(Safe_Proba)

#Dice 2 : Normal Dice
def Normal_Matrix(layout,nbr_steps,circle):
    #Contrary to Safe Dice where it's no possible to further than 15 naturally and where traps aren't activated, Normal Dice have to consider this elements 

    # layout : Vector indicating the location and type of traps
    # nbr_steps : Number of squares in the game
    # circle : boolean parameter that indicates if the game is in mode circle of not

    #The goal of this function is to return a transition probability matrix for the normal dice

    Normal_Proba =  np.zeros((nbr_steps,nbr_steps))

    for step in range(len(Normal_Proba)):
        #33.3% probability => Stay on the same place | Only 0,1 or 2 is possible with the normal dice
        Normal_Proba[step][step] = 1/3

        #On step 3, 16,6% probability => Go to 4, 16,6% probability => Go to 5 | The construction of the game imposes this rule
        #On step 3, 16,6% probability => Go to 11, 16,6% probability => Go to 12 | The construction of the game imposes this rule
        if step == 2:
            Normal_Proba[step][step+1] = (1/3) * (1/2)
            Normal_Proba[step][step+2] = (1/3) * (1/2)
            Normal_Proba[step][10] = (1/3) * (1/2)
            Normal_Proba[step][11] = (1/3) * (1/2)
        
        #On step 9, 33.3% probability => Go to the final step | The construction of the game imposes this rule
        #On step 9, 33.3% probability => Go to 10 | The properties of this dice imposes this rule
        elif step == 8: 
            Normal_Proba[step][14] = 1/3
            Normal_Proba[step][step+1] = 1/3

        #On step 10 or 14, if circle = True , 33% probability => Go to the final step, 33% probability => Return to the first step | The construction of the game imposes this rule
        #On step 10 or 14, if circle = False , 66% probability => Go to the final step | The construction of the game imposes this rule
        elif step == 9 or step == 13:
            if circle:
                Normal_Proba[step][0] = 1/3
                Normal_Proba[step][14] = 1/3
            else:
                Normal_Proba[step][14] = 2/3
        
        #On the final step, 100% probability => Stay on the final step | When we reach the final step, the game is over
        elif step == 14:
            Normal_Proba[step][step] = 1
        
        #On all the other cases, the game depends on the dice
        else:
            Normal_Proba[step][step+1] = 1/3
            Normal_Proba[step][step+2] = 1/3

    #Like we said before, theses probabilities of transition is affected by traps
    for i in range(nbr_steps - 1):
        for j in range(1,nbr_steps - 1):

            #layout[j] = 1 | Restart trap: Immediately teleports the player back to the first square
            if layout[j] == 1 and j != 0: #A trap can't be activated on the first step
                #On same step, 33.3%/2 probability => Stay to the same step | The properties of this dice imposes this rule
                #On same step, Adding 33,3%/2 probabilty => Go to the first step | The properties of this dice imposes this rule
                Normal_Proba[i][0] = Normal_Proba[i][0] + Normal_Proba[i][j]/2
                Normal_Proba[i][j] = Normal_Proba[i][j]/2
   
            #layout[j] = 2 | Penalty trap: Immediately teleports the player 3 squares backwards
            elif layout[j] == 2 and j != 0: #A trap can't be activated on the first step
                #On steps 2, 3 or 4, 33.3%/2 probability => Stay to the same step | The properties of this dice imposes this rule
                #On steps 2, 3 or 4, Adding 33.3%/2 probability => Go to the first step | The properties of this dice imposes this rule
                if j == 1 or j == 2 or j == 3:
                    Normal_Proba[i][0] = Normal_Proba[i][0] + Normal_Proba[i][j]/2
                    Normal_Proba[i][j] = Normal_Proba[i][j]/2
                        
                #On steps 11, 12 or 13, 33.3%/2 probability => Stay to the same step | The properties of this dice imposes this rule
                #On steps 11, 12 or 13, Adding 33.3%/2 probability => Go to 10 steps back | The construction of the game imposes this rule
                elif j == 10 or j == 11 or j == 12:
                    Normal_Proba[i][j-10] = Normal_Proba[i][j-10] + Normal_Proba[i][j]/2
                    Normal_Proba[i][j] = Normal_Proba[i][j]/2
            
                else:
                    Normal_Proba[i][j-3] = Normal_Proba[i][j-3] + Normal_Proba[i][j]/2
                    Normal_Proba[i][j] = Normal_Proba[i][j]/2
    
    for i in range(nbr_steps - 1):
        for j in range(1,nbr_steps - 1):            

            #layout[j] = 4 | Gamble trap: Randomly teleports the player anywhere on the board, with equal, uniform, probability
            if layout[j] == 4 and j != 0: #A trap can't be activated on the first step
                if j == i or j == i+1 or j == i+2: #Only lines of two last steps before the trap and the step of trap can be affected
                #On same step, 33.3%/2 probability => Stay to the same step | The properties of this dice imposes this rule
                #On all steps, Adding (33.3%/2)/15 probability => Go to an random step | The properties of this dice imposes this rule
                    coefficient = ((Normal_Proba[i][j])/2)/nbr_steps
                    Normal_Proba[i][j] = Normal_Proba[i][j]/2
                    for k in range(nbr_steps):
                        Normal_Proba[i][k] = Normal_Proba[i][k] + coefficient

    return pd.DataFrame(Normal_Proba)

#Dice 3 : Risky Dice
def Risky_Matrix(layout,nbr_steps,circle):
    #Contrary to Safe Dice where it's no possible to further than 15 naturally and where traps aren't activated, Normal Dice have to consider this elements 

    # layout : Vector indicating the location and type of traps
    # nbr_steps : Number of squares in the game
    # circle : boolean parameter that indicates if the game is in mode circle of not

    #The goal of this function is to return a transition probability matrix for the risky dice

    Risky_Proba =  np.zeros((nbr_steps,nbr_steps))

    for step in range(len(Risky_Proba)):
        #25% probability => Stay on the same place | Only 0,1,2 or 3 is possible with the risky dice
        Risky_Proba[step][step] = 1/4

        #On step 3, 12,5% probability => Go to 4, 12,5% probability => Go to 5, 12,5% probability => Go to 6 | The construction of the game imposes this rule
        #On step 3, 12,5% probability => Go to 11, 12,5% probability => Go to 12, 12,5% probability => Go to 13 | The construction of the game imposes this rule
        if step == 2:
            Risky_Proba[step][step+1] = (1/4) * (1/2)
            Risky_Proba[step][step+2] = (1/4) * (1/2)
            Risky_Proba[step][step+3] = (1/4) * (1/2)
            Risky_Proba[step][10] = (1/4) * (1/2)
            Risky_Proba[step][11] = (1/4) * (1/2)
            Risky_Proba[step][12] = (1/4) * (1/2)
        
        #On step 8, 25% probability => Go to the final step | The construction of the game imposes this rule
        #On step 8, 25% probability => Go to the final 10 | The properties of this dice imposes this rule
        #On step 8, 25% probability => Go to the final 9 | The properties of this dice imposes this rule
        elif step == 7:
            Risky_Proba[step][14] = 1/4
            Risky_Proba[step][step+2] = 1/4
            Risky_Proba[step][step+1] = 1/4

        #On step 9 or 13, if circle = True, 25% probability => Go to 10 or 14, 25% probability => Go to the final step, 25% probability => Return to the first step | The construction of the game imposes this rule
        #On step 9 or 13, if circle = False, 25% probability => Go to 10 or 14, 50% probability => Go to the finalt step | The construction of the game imposes this rule    
        elif step == 8 or step == 12:
            if circle:
                Risky_Proba[step][step+1] = 1/4
                Risky_Proba[step][14] = 1/4
                Risky_Proba[step][0] = 1/4
            else:
                Risky_Proba[step][step+1] = 1/4
                Risky_Proba[step][14] = 2/4

        #On step 10 or 14, if circle = True, 25% probability => Go to the final step, 25% probability => Return to the first step, 25% probability => Return to the second step | The construction of the game imposes this rule
        #On step 10 or 14, if circle = False, 75% probability => Go to the final step | The construction of the game imposes this rule
        elif step == 9 or step == 13:
            if circle:
                Risky_Proba[step][14] = 1/4
                Risky_Proba[step][0] = 1/4
                Risky_Proba[step][1] = 1/4
            else:
                Risky_Proba[step][14] = 3/4

        #On the final step, 100% probability => Stay on the final step | When we reach the final step, the game is over
        elif step == 14:
            Risky_Proba[step][step] = 1
        
        #On alle the other cases, the game depends on the dice
        else:
            Risky_Proba[step][step+1] = 1/4
            Risky_Proba[step][step+2] = 1/4
            Risky_Proba[step][step+3] = 1/4

    #Like we said before, theses probabilities of transition is affected by traps
    for i in range(nbr_steps - 1):
        for j in range(1, nbr_steps - 1):
            
            #layout[j] = 1 | Restart trap: Immediately teleports the player back to the first square
            if layout[j] == 1 and j != 0: #A trap can't be activated on the first step
                #On same step, 0% probability => Stay to the same step | The properties of this dice imposes this rule
                #On same step, Adding 25% probabilty => Go to the first step | The properties of this dice imposes this rule
                Risky_Proba[i][0] = Risky_Proba[i][0] + Risky_Proba[i][j]
                Risky_Proba[i][j] = 0

            #layout[j] = 2 | Penalty trap: Immediately teleports the player 3 squares backwards
            elif layout[j] == 2 and j != 0: #A trap can't be activated on the first step
                #On steps 2, 3 or 4, 0% probability => Stay to the same step | The properties of this dice imposes this rule
                #On steps 2, 3 or 4, Adding 25% probability => Go to the first step | The properties of this dice imposes this rule
                if j == 1 or j == 2 or j == 3:
                    Risky_Proba[i][0] = Risky_Proba[i][0] + Risky_Proba[i][j]
                    Risky_Proba[i][j] = 0
                        
                #On steps 11, 12 or 13, 0% probability => Stay to the same step | The properties of this dice imposes this rule
                #On steps 11, 12 or 13, Adding 25% probability => Go to 10 steps back | The construction of the game imposes this rule
                elif j == 10 or j == 11 or j == 12:
                    Risky_Proba[i][j-10] = Risky_Proba[i][j-10] + Risky_Proba[i][j]
                    Risky_Proba[i][j] = 0
            
                else:
                    Risky_Proba[i][j-3] = Risky_Proba[i][j-3] + Risky_Proba[i][j]
                    Risky_Proba[i][j] = 0

    for i in range(nbr_steps - 1):
        for j in range(1,nbr_steps - 1):            

            #layout[j] = 4 | Gamble trap: Randomly teleports the player anywhere on the board, with equal, uniform, probability
            if layout[j] == 4 and j != 0: #A trap can't be activated on the first step
                if j == i or j == i+1 or j == i+2 or j == i+3: #Only lines of two last steps before the trap and the step of trap can be affected
                #On same step, 0% probability => Stay to the same step | The properties of this dice imposes this rule
                #On all steps, Adding 25%/15 probability => Go to an random step | The properties of this dice imposes this rule
                    coefficient = Risky_Proba[i][j]/nbr_steps
                    Risky_Proba[i][j] = 0
                    for k in range(nbr_steps):
                        Risky_Proba[i][k] = Risky_Proba[i][k] + coefficient

    return pd.DataFrame(Risky_Proba)

#####################################

#Construction of markovDecision function

def markovDecision(layout = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],circle = False): #Optimal strategy

    # layout : Vector indicating the location and type of traps
    # circle : Boolean parameter that indicates if the game is in mode circle of not

    nbr_steps = len(layout) #Number of steps/states
    
    P1 = Safe_Matrix(nbr_steps) #Implementation of the result of Safe_Matrix in the variable
    P2 = Normal_Matrix(layout, nbr_steps, circle) #Implementation of the result of Normal_Matrix in the variable
    P3 = Risky_Matrix(layout, nbr_steps, circle) #Implementation of the result of Risky_Matrix in the variable

    #Construction of Rewards vectors which differ according to the use of the chosen dice
    Rewards_Safe = []
    Rewards_Normal = []
    Rewards_Risky = []

    for step in range(len(layout)):
        if step < len(layout)-1: #If it's not the last state
            if layout[step] == 3: #If the state have a prison trap
                Rewards_Safe.append(1) #The safe dice allow us to avoid all the traps
                Rewards_Normal.append(1.5) #The normal dice active the trap with 50% probability, the mean of rewards is 1.5
                Rewards_Risky.append(2) #The risky dice active the trap every time, the rewards is 2

            elif layout[step] != 3:
                Rewards_Safe.append(1)
                Rewards_Normal.append(1)
                Rewards_Risky.append(1)

        if step == len(layout)-1: #If it's the last state the reward is 0 for all dices
            Rewards_Safe.append(0)
            Rewards_Normal.append(0)
            Rewards_Risky.append(0)
    
    Rewards_Safe = np.array(Rewards_Safe, dtype = float)
    Rewards_Normal = np.array(Rewards_Normal, dtype = float)
    Rewards_Risky = np.array(Rewards_Risky, dtype = float)

    #Construction of value iteration
    Expec = [i for i in range(nbr_steps - 1, -1, -1)]
    Max = float("inf") #First implementation of the Max value which will be the value of the largest absolute difference between Expec and Expec_last
    
    while Max > 0.0001:
        #Construction of Safe_V
        Safe_V = Rewards_Safe + np.dot(P1,Expec)

        #Construction of Normal_V
        Normal_V = Rewards_Normal + np.dot(P2,Expec)

        #Construction of Risky_V
        Risky_V = Rewards_Risky + np.dot(P3,Expec)

        #Save of the matrix Expec into another variablee before modifing it
        Expec_last = np.array(deepcopy(Expec), dtype = float)

        Dice = [1] * nbr_steps #Reset of the policy dice because of the new iteration
        for i in range(nbr_steps - 1):
            Values_3V = []
            Values_3V.append(Safe_V[i])
            Values_3V.append(Normal_V[i])
            Values_3V.append(Risky_V[i])
            Expec[i] = min(Values_3V) #Calcul of the minimum between 3 different values
            Dice[i] = int(np.argmin(Values_3V) + 1) #Print of the dice we have to play
    
        #We want to find the maximum absolute difference between Expec and Expec_last
        Difference = np.abs(Expec - Expec_last)
        Max = np.amax(Difference)
    
    #Turn the Expec and Dice lists into a vector of type np.ndarray
    Expec = np.array(Expec)
    Dice = np.array(Dice)
    return [Expec[:14].tolist(), Dice[:14].tolist()]

#####################################

#Construction of strategies functions

def Dice1_strategy(layout = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],circle = False):

    # layout : Vector indicating the location and type of traps
    # circle : Boolean parameter that indicates if the game is in mode circle of not

    nbr_steps = len(layout) #Number of steps/states
    
    P1 = Safe_Matrix(nbr_steps) #Implementation of the result of Safe_Matrix in the variable

    #Construction of Rewards vectors which differ according to the use of the chosen dice
    Rewards_Safe = []

    for step in range(len(layout)):
        if step < len(layout)-1: #If it's not the last state
            if layout[step] == 3: #If the state have a prison trap
                Rewards_Safe.append(1) #The safe dice allow us to avoid all the traps

            elif layout[step] != 3:
                Rewards_Safe.append(1)

        if step == len(layout)-1: #If it's the last state the reward is 0 for all dices
            Rewards_Safe.append(0)

    Rewards_Safe = np.array(Rewards_Safe, dtype = float)

    #Construction of value iteration
    Expec = [i for i in range(nbr_steps - 1, -1, -1)]
    Max = float("inf") #First implementation of the Max value which will be the value of the largest absolute difference between Expec and Expec_last
    
    while Max > 0.0001:
        #Construction of Safe_V
        Safe_V = Rewards_Safe + np.dot(P1,Expec)

        #Save of the matrix Expec into another variablee before modifing it
        Expec_last = np.array(deepcopy(Expec), dtype = float)

        Dice = [1] * nbr_steps #Reset of the policy dice because of the new iteration
        for i in range(nbr_steps - 1):
            Values_3V = []
            Values_3V.append(Safe_V[i])
            Expec[i] = min(Values_3V) #Calcul of the minimum between 3 different values
            Dice[i] = 1 #Print of the dice we have to play
    
        #We want to find the maximum absolute difference between Expec and Expec_last
        Difference = np.abs(Expec - Expec_last)
        Max = np.amax(Difference)
    
    #Turn the Expec and Dice lists into a vector of type np.ndarray
    Expec = np.array(Expec)
    Dice = np.array(Dice)
    return [Expec.tolist(),Dice.tolist()]

def Dice2_strategy(layout = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],circle = False):

    # layout : Vector indicating the location and type of traps
    # circle : Boolean parameter that indicates if the game is in mode circle of not

    nbr_steps = len(layout) #Number of steps/states

    P2 = Normal_Matrix(layout, nbr_steps, circle) #Implementation of the result of Normal_Matrix in the variable

    #Construction of Rewards vectors which differ according to the use of the chosen dice
    Rewards_Normal = []

    for step in range(len(layout)):
        if step < len(layout)-1: #If it's not the last state
            if layout[step] == 3: #If the state have a prison trap
                Rewards_Normal.append(1.5) #The normal dice active the trap with 50% probability, the mean of rewards is 1.5

            elif layout[step] != 3:
                Rewards_Normal.append(1)

        if step == len(layout)-1: #If it's the last state the reward is 0 for all dices
            Rewards_Normal.append(0)

    Rewards_Normal = np.array(Rewards_Normal, dtype = float)

    #Construction of value iteration
    Expec = [i for i in range(nbr_steps - 1, -1, -1)]
    Max = float("inf") #First implementation of the Max value which will be the value of the largest absolute difference between Expec and Expec_last
    
    while Max > 0.0001:
        #Construction of Normal_V
        Normal_V = Rewards_Normal + np.dot(P2,Expec)

        #Save of the matrix Expec into another variablee before modifing it
        Expec_last = np.array(deepcopy(Expec), dtype = float)

        Dice = [1] * nbr_steps #Reset of the policy dice because of the new iteration
        for i in range(nbr_steps - 1):
            Values_3V = []
            Values_3V.append(Normal_V[i])
            Expec[i] = min(Values_3V) #Calcul of the minimum between 3 different values
            Dice[i] = 2 #Print of the dice we have to play
    
        #We want to find the maximum absolute difference between Expec and Expec_last
        Difference = np.abs(Expec - Expec_last)
        Max = np.amax(Difference)
    
    #Turn the Expec and Dice lists into a vector of type np.ndarray
    Expec = np.array(Expec)
    Dice = np.array(Dice)
    return [Expec.tolist(),Dice.tolist()]

def Dice3_strategy(layout = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],circle = False):

    # layout : Vector indicating the location and type of traps
    # circle : Boolean parameter that indicates if the game is in mode circle of not

    nbr_steps = len(layout) #Number of steps/states

    P3 = Risky_Matrix(layout, nbr_steps, circle) #Implementation of the result of Risky_Matrix in the variable

    #Construction of Rewards vectors which differ according to the use of the chosen dice
    Rewards_Risky = []

    for step in range(len(layout)):
        if step < len(layout)-1: #If it's not the last state
            if layout[step] == 3: #If the state have a prison trap
                Rewards_Risky.append(2) #The risky dice active the trap every time, the rewards is 2

            elif layout[step] != 3:
                Rewards_Risky.append(1)

        if step == len(layout)-1: #If it's the last state the reward is 0 for all dices
            Rewards_Risky.append(0)
    
    Rewards_Risky = np.array(Rewards_Risky, dtype = float)

    #Construction of value iteration
    Expec = [i for i in range(nbr_steps - 1, -1, -1)]
    Max = float("inf") #First implementation of the Max value which will be the value of the largest absolute difference between Expec and Expec_last
    
    while Max > 0.0001:
        #Construction of Risky_V
        Risky_V = Rewards_Risky + np.dot(P3,Expec)

        #Save of the matrix Expec into another variablee before modifing it
        Expec_last = np.array(deepcopy(Expec), dtype = float)

        Dice = [1] * nbr_steps #Reset of the policy dice because of the new iteration
        for i in range(nbr_steps - 1):
            Values_3V = []
            Values_3V.append(Risky_V[i])
            Expec[i] = min(Values_3V) #Calcul of the minimum between 3 different values
            Dice[i] = 3 #Print of the dice we have to play
    
        #We want to find the maximum absolute difference between Expec and Expec_last
        Difference = np.abs(Expec - Expec_last)
        Max = np.amax(Difference)
    
    #Turn the Expec and Dice lists into a vector of type np.ndarray
    Expec = np.array(Expec)
    Dice = np.array(Dice)
    return [Expec.tolist(),Dice.tolist()]

def Purely_Random_Strategy(layout = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], circle = False):
    
    # layout : Vector indicating the location and type of traps
    # circle : Boolean parameter that indicates if the game is in mode circle of not

    nbr_steps = len(layout) #Number of steps/states
    
    P1 = Safe_Matrix(nbr_steps) #Implementation of the result of Safe_Matrix in the variable
    P2 = Normal_Matrix(layout, nbr_steps, circle) #Implementation of the result of Normal_Matrix in the variable
    P3 = Risky_Matrix(layout, nbr_steps, circle) #Implementation of the result of Risky_Matrix in the variable

    #Construction of Rewards vectors which differ according to the use of the chosen dice
    Rewards_Safe = []
    Rewards_Normal = []
    Rewards_Risky = []

    for step in range(len(layout)):
        if step < len(layout)-1: #If it's not the last state
            if layout[step] == 3: #If the state have a prison trap
                Rewards_Safe.append(1) #The safe dice allow us to avoid all the traps
                Rewards_Normal.append(1.5) #The normal dice active the trap with 50% probability, the mean of rewards is 1.5
                Rewards_Risky.append(2) #The risky dice active the trap every time, the rewards is 2

            elif layout[step] != 3:
                Rewards_Safe.append(1)
                Rewards_Normal.append(1)
                Rewards_Risky.append(1)

        if step == len(layout)-1: #If it's the last state the reward is 0 for all dices
            Rewards_Safe.append(0)
            Rewards_Normal.append(0)
            Rewards_Risky.append(0)
    
    Rewards_Safe = np.array(Rewards_Safe, dtype = float)
    Rewards_Normal = np.array(Rewards_Normal, dtype = float)
    Rewards_Risky = np.array(Rewards_Risky, dtype = float)

    Dice = [1] * nbr_steps #Reset of the policy dice because of the new iteration
    for i in range(nbr_steps):
        choice = rd.randrange(1,4) #We choose randomly the dice
        if choice == 1: #if the value of the dice is 1, we use the Safe_Matrix
            Dice[i] = 1 #Print of the dice we have to play
        elif choice == 2: #if the value of the dice is 2, we use the Normal_Matrix
            Dice[i] = 2 #Print of the dice we have to play
        elif choice == 3: #if the value of the dice is 3, we use the Risky_Matrix
            Dice[i] = 3 #Print of the dice we have to play

    #Construction of value iteration
    Expec = [i for i in range(nbr_steps - 1, -1, -1)]
    Max = float("inf") #First implementation of the Max value which will be the value of the largest absolute difference between Expec and Expec_last  
    while Max > 0.0001:
        #Construction of Safe_V
        Safe_V = Rewards_Safe + np.dot(P1,Expec)

        #Construction of Normal_V
        Normal_V = Rewards_Normal + np.dot(P2,Expec)

        #Construction of Risky_V
        Risky_V = Rewards_Risky + np.dot(P3,Expec)

        #Save of the matrix Expec into another variablee before modifing it
        Expec_last = np.array(deepcopy(Expec), dtype = float)

        for i in range(nbr_steps):
            Values_3V = []
            if Dice[i] == 1:
                Values_3V.append(Safe_V[i])
            elif Dice[i] == 2:
                Values_3V.append(Normal_V[i])
            elif Dice[i] == 3:
                Values_3V.append(Risky_V[i])

            Expec[i] = min(Values_3V) #Calcul of the minimum between 3 different values
    
        #We want to find the maximum absolute difference between Expec and Expec_last
        Difference = np.abs(Expec - Expec_last)
        Max = np.amax(Difference)
    
    #Turn the Expec and Dice lists into a vector of type np.ndarray
    Expec = np.array(Expec)
    Dice = np.array(Dice)
    return [Expec.tolist(), Dice.tolist()]

def Mixed_Random_Strategy(layout = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], circle = False, strategie = "Markov", nbr_step_random = 7):
    
    # layout : Vector indicating the location and type of traps
    # circle : Boolean parameter that indicates if the game is in mode circle of not
    # strategie : Type of strategie that we choose to use
    # nbr_step_random : Number of steps where policy is choosen randomly

    nbr_steps = len(layout) #Number of steps/states

    P1 = Safe_Matrix(nbr_steps) #Implementation of the result of Safe_Matrix in the variable
    P2 = Normal_Matrix(layout, nbr_steps, circle) #Implementation of the result of Normal_Matrix in the variable
    P3 = Risky_Matrix(layout, nbr_steps, circle) #Implementation of the result of Risky_Matrix in the variable

    #Construction of Rewards vectors which differ according to the use of the chosen dice
    Rewards_Safe = []
    Rewards_Normal = []
    Rewards_Risky = []

    for step in range(len(layout)):
        if step < len(layout)-1: #If it's not the last state
            if layout[step] == 3: #If the state have a prison trap
                Rewards_Safe.append(1) #The safe dice allow us to avoid all the traps
                Rewards_Normal.append(1.5) #The normal dice active the trap with 50% probability, the mean of rewards is 1.5
                Rewards_Risky.append(2) #The risky dice active the trap every time, the rewards is 2

            elif layout[step] != 3:
                Rewards_Safe.append(1)
                Rewards_Normal.append(1)
                Rewards_Risky.append(1)

        if step == len(layout)-1: #If it's the last state the reward is 0 for all dices
            Rewards_Safe.append(0)
            Rewards_Normal.append(0)
            Rewards_Risky.append(0)
    
    Rewards_Safe = np.array(Rewards_Safe, dtype = float)
    Rewards_Normal = np.array(Rewards_Normal, dtype = float)
    Rewards_Risky = np.array(Rewards_Risky, dtype = float)

    Dice = [1] * nbr_steps #Reset of the policy dice because of the new iteration

    for i in range(nbr_steps):
        if i < nbr_step_random:
            choice = rd.randrange(1,4) #We choose randomly the dice
            if choice == 1: #if the value of the dice is 1, we use the Safe_Matrix
                Dice[i] = 1 #Print of the dice we have to play

            elif choice == 2: #if the value of the dice is 2, we use the Normal_Matrix
                Dice[i] = 2 #Print of the dice we have to play

            elif choice == 3: #if the value of the dice is 3, we use the Risky_Matrix
                Dice[i] = 3 #Print of the dice we have to play
        
        elif i >= nbr_step_random:
            if strategie == "Markov":
                Optimal_values, Optimal_Dice = markovDecision(layout,circle)
                if i <= 13:
                    Dice[i] = Optimal_Dice[i]
                else:
                    continue

            elif strategie == "Dice1":
                    Dice[i] = 1 #Print of the dice we have to play
                
            elif strategie == "Dice2":
                    Dice[i] = 2 #Print of the dice we have to play
                
            elif strategie == "Dice3":
                    Dice[i] = 3 #Print of the dice we have to play

    #Construction of value iteration
    Expec = [i for i in range(nbr_steps - 1, -1, -1)]
    Max = float("inf") #First implementation of the Max value which will be the value of the largest absolute difference between Expec and Expec_last  
    while Max > 0.0001:
        #Construction of Safe_V
        Safe_V = Rewards_Safe + np.dot(P1,Expec)

        #Construction of Normal_V
        Normal_V = Rewards_Normal + np.dot(P2,Expec)

        #Construction of Risky_V
        Risky_V = Rewards_Risky + np.dot(P3,Expec)

        #Save of the matrix Expec into another variablee before modifing it
        Expec_last = np.array(deepcopy(Expec), dtype = float)

        for i in range(nbr_steps):
            Values_3V = []
            if Dice[i] == 1:
                Values_3V.append(Safe_V[i])
            elif Dice[i] == 2:
                Values_3V.append(Normal_V[i])
            elif Dice[i] == 3:
                Values_3V.append(Risky_V[i])

            Expec[i] = min(Values_3V) #Calcul of the minimum between 3 different values
    
        #We want to find the maximum absolute difference between Expec and Expec_last
        Difference = np.abs(Expec - Expec_last)
        Max = np.amax(Difference)
    
    #Turn the Expec and Dice lists into a vector of type np.ndarray
    Expec = np.array(Expec)
    Dice = np.array(Dice)
    return [Expec.tolist(), Dice.tolist()]

#####################################

#Construction of Simulator algorithm 

def Simulator(layout = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],circle = False ,Dice = [3,3,3,3,3,3,3,3,3,3,3,3,3,3],nbr_sim = 50000,start = 0):

    # layout : Vector indicating the location and type of traps
    # circle : Boolean parameter that indicates if the game is in mode circle of not
    # Dice : List of policy we have to use
    # nbr_sim : Number of time we play with the simulator
    # start : Step where we begin

    Cost = 0 #Variable which will represents the cost to finish the game

    for game in range(nbr_sim):
        i = start
        while i < 14: #As long as we are not at the last square, we have to keep playing
            last_step = i
            if Dice[i] == 1: #If the policy is using Safe Dice
                progression = rd.randint(0,1) #Safe Dice allows to do 0 or 1
            
            elif Dice[i] == 2: #If the policy is using Normal Dice
                progression = rd.randint(0,2) #Normal Dice allows to do 0,1 or 2
            
            elif Dice[i] == 3: #If the policy is using Risky Dice
                progression = rd.randint(0,3) #Risky Dice allows to do 0,1,2 or 3

            #------------------------#

            if progression == 0: #We stay at the same step
                i = i

            elif i == 2: #When we are in the step 3, we can take 2 ways (Slow of Fast lane)
                Taking_Fast_Lane = rd.randint(0,1)
                if Taking_Fast_Lane == 0: #In this case, we take the Slow lane
                    i = i + progression
                
                else: #In thise case, we take the Fast lane
                    i = 9 + progression
            
            elif i == 7 and progression == 3: #In this case, we go directly to the last step
                i = 14
            
            elif i == 8 and progression == 2: #in this case, we go directly to the last step
                i = 14

            elif (i == 8 or i == 12) and progression ==3: #Depending on the game mode, it is possible to restart to the square 1
                if not circle: #In this case, we can't go further than square 14
                    i = 14
                
                elif circle: #In thise case, we have force to start again the game from the step 1
                    i = 0

            elif i == 9 or i == 13: #Depending on the game mode, it is possible to restart to the square 1
                if progression == 1: #In thise case , we go to the last step
                    i = 14

                elif not circle:
                    i = 14
                
                elif circle:
                    if progression > 1:
                        i = progression - 2 #If progression == 3, we go to step 2 (i = 1). If progression == 2, we go to step 1 (i = 0)
            
            else: #When we aren't to a special step, we move the number of time indicates by progression
                i = i + progression

            #------------------------#

            if Dice[last_step] == 2: #When we play with the dice 2, trap have a 50% probability to be activated
                if i < 14: #if we are in the last step, we have to stop de game
                    Activating_Trap = rd.randint(0,1) 
                    if Activating_Trap == 1: #Trap will be activated
                        #layout[j] = 1 | Restart trap: Immediately teleports the player back to the first square
                        if layout[i] == 1:
                            i = 0

                        #layout[j] = 2 | Penalty trap: Immediately teleports the player 3 squares backwards
                        elif layout[i] == 2:
                            if i == 1 or i == 2 or i == 10: #If we are in the step 1,2 or 11
                                i = 0
                                
                            elif i == 11 or i == 12: #If we are in the step 12 or 13
                                i = i - 10

                            else: #If we are in the all other cases, we simply go back 3 squares
                                i = i - 3
 
                        #layout[j] = 3 | Prison trap: The player must wait one turn before playing again.
                        elif layout[i] == 3:
                            Cost = Cost + 1

                        #layout[j] = 4 | Gamble trap: Randomly teleports the player anywhere on the board, with equal, uniform, probability 
                        elif layout[i] == 4:
                            Teleporting_Step = rd.randint(0,14)
                            i = Teleporting_Step
                    if Activating_Trap == 0: #Trap will not be activated
                        i = i

                
            elif Dice[last_step] == 3: #When we play with the dice 3, trap have a 100% probability to be activated
                if i < 14: #if we are in the last step, we have to stop de game               
                    #layout[j] = 1 | Restart trap: Immediately teleports the player back to the first square
                    if layout[i] == 1:
                        i = 0

                    #layout[j] = 2 | Penalty trap: Immediately teleports the player 3 squares backwards
                    elif layout[i] == 2:
                        if i == 1 or i == 2 or i == 10: #If we are in the step 1,2 or 11
                            i = 0

                        elif i == 11 or i == 12: #If we are in the step 12 or 13
                            i = i - 10

                        else: #If we are in the all other cases, we simply go back 3 squares
                            i = i - 3


                    #layout[j] = 3 | Prison trap: The player must wait one turn before playing again.
                    elif layout[i] == 3:
                        Cost = Cost + 1

                    #layout[j] = 4 | Gamble trap: Randomly teleports the player anywhere on the board, with equal, uniform, probability 
                    elif layout[i] == 4:
                        Teleporting_Step = rd.randint(0,14)
                        i = Teleporting_Step

            #------------------------#

            Cost = Cost + 1

    Mean_Cost = Cost/nbr_sim #Mean of costs of each game have to be calculated

    return(Mean_Cost)

#####################################

#Parameters
Number_of_Game = 10000
#layout = [0,0,3,0,0,0,0,0,0,3,0,0,0,1,0] #Game without trap
layout = [0,2,1,1,2,1,4,1,1,0,0,3,0,1,0] #Our Game. An other game can be tested
circle = True
#Comparaison between theorical expected costs and empirical average costs of Optimal Strategy (markovDecision)
print("Comparaison Theorical Expected Costs vs Empirical Average Costs : Optimal Strategy (markovDecision)\n")
#Theorical result of optimal strategie
print("=> Theorical expected costs & policy of Optimal strategy :")
print(markovDecision(layout,circle))
print("\n")

Expec, Policy = markovDecision(layout,circle)
Policy.append(1)
Optimal_Results = [] #Construction of vector which will containts Empirical Costs for each policy
for step in range(len(layout)):
    Optimal_Results.append(Simulator(layout,circle,Policy,Number_of_Game,step)) #Simulation of the game

print("=> Empirical average costs (Optimal Strategy) : ")
print(Optimal_Results[:14])
print("\n")

#Graph of comparaison
print("=> Graph of comparaison :")
plt.grid(True)
plt.plot([i for i in range(1,len(layout))], Optimal_Results[:14], marker = "o", label = "Empirical Average Costs")
plt.plot([i for i in range(1,len(layout))], Expec, marker = "*", label = "Theorical Expected Costs")
plt.xlabel("Number of steps")
plt.ylabel("Costs of steps")
plt.legend()
plt.show()
print("\n")

#Comparaison between costs of differents strategies
print("Comparaison between costs of differents strategies\n")
#Theorical results and policy for each strategy are calculated
Expec_Dice1 , Policy_Dice1 = Dice1_strategy(layout,circle)
Expec_Dice2 , Policy_Dice2 = Dice2_strategy(layout,circle)
Expec_Dice3 , Policy_Dice3 = Dice3_strategy(layout,circle)
Expec_Purely_Random , Policy_Purely_Random = Purely_Random_Strategy(layout,circle)
Expec_Mixed_Random_Half_Optimal , Policy_Mixed_Random_Half_Optimal = Mixed_Random_Strategy(layout, circle, "Markov",7) #Half version of theses tests involves 50% of random and 50% of the strategy
Expec_Mixed_Random_Half_Dice1 , Policy_Mixed_Random_Half_Dice1 = Mixed_Random_Strategy(layout, circle, "Dice1",7)
Expec_Mixed_Random_Half_Dice2 , Policy_Mixed_Random_Half_Dice2 = Mixed_Random_Strategy(layout, circle, "Dice2",7)
Expec_Mixed_Random_Half_Dice3 , Policy_Mixed_Random_Half_Dice3 = Mixed_Random_Strategy(layout, circle, "Dice3",7)

#Construction of vectors which will containt Empirical Costs for each policy
Dice1_Results = []
Dice2_Results = []
Dice3_Results = []
Purely_Random_Results = []
Mixed_Random_Half_Optimal_Results = []
Mixed_Random_Half_Dice1_Results = []
Mixed_Random_Half_Dice2_Results = []
Mixed_Random_Half_Dice3_Results = []

#Simulation of theses strategy on the game
for step in range(len(layout)):
    Dice1_Results.append(Simulator(layout,circle,Policy_Dice1,Number_of_Game,step))
    Dice2_Results.append(Simulator(layout,circle,Policy_Dice2,Number_of_Game,step))
    Dice3_Results.append(Simulator(layout,circle,Policy_Dice3,Number_of_Game,step))
    Purely_Random_Results.append(Simulator(layout,circle,Policy_Purely_Random,Number_of_Game,step))
    Mixed_Random_Half_Optimal_Results.append(Simulator(layout,circle,Policy_Mixed_Random_Half_Optimal,Number_of_Game,step))
    Mixed_Random_Half_Dice1_Results.append(Simulator(layout,circle,Policy_Mixed_Random_Half_Dice1,Number_of_Game,step))
    Mixed_Random_Half_Dice2_Results.append(Simulator(layout,circle,Policy_Mixed_Random_Half_Dice2,Number_of_Game,step))
    Mixed_Random_Half_Dice3_Results.append(Simulator(layout,circle,Policy_Mixed_Random_Half_Dice3,Number_of_Game,step))

print("=> Optimal Strategy : ")
print(Optimal_Results[:14])
print("\n")
print("=> Dice1 Strategy : ")
print(Dice1_Results[:14])
print("\n")
print("=> Dice2 Strategy : ")
print(Dice2_Results[:14])
print("\n")
print("=> Dice3 Strategy : ")
print(Dice3_Results[:14])
print("\n")
print("=> Purely Random Strategy : ")
print(Purely_Random_Results[:14])
print("\n")
print("=> Mixed Random Strategy - 50% Optimal : ")
print(Mixed_Random_Half_Optimal_Results[:14])
print("\n")
print("=> Mixed Random Strategy - 50% Dice1 : ")
print(Mixed_Random_Half_Dice1_Results[:14])
print("\n")
print("=> Mixed Random Strategy - 50% Dice2 : ")
print(Mixed_Random_Half_Dice2_Results[:14])
print("\n")
print("=> Mixed Random Strategy - 50% Dice3 : ")
print(Mixed_Random_Half_Dice3_Results[:14])
print("\n")


#Graph of comparaison
print("=> Graph of comparaison :")
plt.grid(True)
plt.plot([i for i in range(1,len(layout))], Optimal_Results[:14], marker = "*", label = "Optimal")
plt.plot([i for i in range(1,len(layout))], Dice1_Results[:14], marker = "o", label = "Dice1")
plt.plot([i for i in range(1,len(layout))], Dice2_Results[:14], marker = "o", label = "Dice2")
plt.plot([i for i in range(1,len(layout))], Dice3_Results[:14], marker = "o", label = "Dice3")
plt.plot([i for i in range(1,len(layout))], Purely_Random_Results[:14], marker = "o", label = "Purely Random")
plt.plot([i for i in range(1,len(layout))], Mixed_Random_Half_Optimal_Results[:14], marker = "o", label = "Mixed Random: 50% Optimal")
plt.plot([i for i in range(1,len(layout))], Mixed_Random_Half_Dice1_Results[:14], marker = "o", label = "Mixed Random: 50% Dice1")
plt.plot([i for i in range(1,len(layout))], Mixed_Random_Half_Dice2_Results[:14], marker = "o", label = "Mixed Random: 50% Dice2")
plt.plot([i for i in range(1,len(layout))], Mixed_Random_Half_Dice3_Results[:14], marker = "o", label = "Mixed Random: 50% Dice3")
plt.xlabel("Number of steps")
plt.ylabel("Costs of steps")
plt.legend()
plt.show()

#####################################
