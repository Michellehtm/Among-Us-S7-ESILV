# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:34:51 2020

@author: Michelle
"""

#%% Step 1
import random
import string

class Node(): 
    
    def __init__(self,player, score=None):
        self.player = player
        self.score = score 
        self.left = None
        self.right = None
        self.height = 1
    
    def __str__(self):
        return "{}, score : {}".format(self.player, self.score)
  
 
class AVLTree(): 
    
    def __init__(self):
        self.root=None       
        
    def insert(self, root, key, player=None): 
      
        # Step 1 - Perform normal BST 
        if not root:
            if player==None:
                str = string.ascii_letters
                a=Node((''.join(random.choice(str) for i in range(5))),key)
            else :
                a = Node(player, key)
            return a
        elif key < root.score:
            if player==None:
                root.left = self.insert(root.left, key) 
            else:
                root.left = self.insert(root.left, key, player)
        else:
            if player==None:
                root.right = self.insert(root.right, key)
            else:
                root.right = self.insert(root.right, key, player)
            
        # Step 2 - Update the height of the ancestor node 
        root.height = 1 + max(self.getHeight(root.left), 
                           self.getHeight(root.right)) 
  
        # Step 3 - Get the balance factor 
        balance = self.getBalance(root)
  
        # Step 4 - If the node is unbalanced, then try out the 4 cases 
        # Case 1 - Left Left 
        if balance > 1 and key < root.left.score: 
            return self.rightRotate(root) 
  
        # Case 2 - Right Right 
        if balance < -1 and key > root.right.score: 
            return self.leftRotate(root) 
  
        # Case 3 - Left Right 
        if balance > 1 and key > root.left.score: 
            root.left = self.leftRotate(root.left) 
            return self.rightRotate(root) 
  
        # Case 4 - Right Left 
        if balance < -1 and key < root.right.score: 
            root.right = self.rightRotate(root.right) 
            return self.leftRotate(root) 
  
        return root 
  
    def leftRotate(self, z): 
  
        if z.right != None:
            y = z.right 
            T2 = y.left 
      
            # Perform rotation 
            y.left = z 
            z.right = T2 
      
            # Update heights 
            z.height = 1 + max(self.getHeight(z.left), 
                             self.getHeight(z.right)) 
            y.height = 1 + max(self.getHeight(y.left), 
                             self.getHeight(y.right)) 
      
            # Return the new root 
            return y 
        return z
  
    def rightRotate(self, z): 
  
        if z.left != None:
            
            y = z.left 
            T3 = y.right 
      
            # Perform rotation 
            y.right = z 
            z.left = T3 
      
            # Update heights 
            z.height = 1 + max(self.getHeight(z.left), 
                            self.getHeight(z.right)) 
            y.height = 1 + max(self.getHeight(y.left), 
                            self.getHeight(y.right)) 
      
            # Return the new root 
            return y 
        return z
  
    def getHeight(self, root): 
        if not root: 
            return 0
  
        return root.height 
  
    def getBalance(self, root): 
        if not root: 
            return 0
  
        return self.getHeight(root.left) - self.getHeight(root.right) 
    
    
    #Generates an AVLTree from a List of nodes
    def avlFromList(self, liste, node=None): 
        self = AVLTree()
        for i in liste:
            node = self.insert(node, i.score, i.player)
        self.root=node
        return self
    
    
    def inorder_print(self, root):
        if root:
            self.inorder_print(root.left)
            print(root)
            self.inorder_print(root.right)
    
    #Similar method to inorder traversal
    #Since inorder traversal returns the nodes in an ascending way, we'll use
    #the same reasoning to fill an ascending list with our nodes
    def fillList(self, root, liste):
        if root:
            self.fillList(root.left, liste)
            liste.append(root)
            self.fillList(root.right, liste)
        return liste
    

## adds a random score between 0 and 12 to the elements of liste
def randScore(liste):
    for i in liste:
        if not i.score:
            i.score=0
        i.score = i.score + random.randint(0,12)
    return liste

## Creates random (if True) games of 10 players or games of 10 players based on ranking if False
def createGames(listplayers, rand=False):
    listgames=[]
    if not rand:
        listgames = [listplayers[x:x+10] for x in range(0, len(listplayers), 10)]
    else:
        l = list(listplayers)
        for j in range(int(len(l)/10)):
            listgames.append(random.sample(l,10))
            for i in listgames[j]:
                if i in l:
                    del l[l.index(i)] #We delete the used Nodes from the copied list to make sure there are no duplicates
    return listgames 

def dropPlayers(listplayers):
    if len(listplayers)>10:
        del listplayers[0:10] #We delete the first 10 Nodes because our list is ascending so first = lowest score
    return listplayers

    
def displayFinals(liste):
    if len(liste)==10:
        print("\nThe finalists are :")
        for i in liste:
            print(i)
    
def podium(liste):
    print("\n3__Bronze medal is attributed to {} with a total of {} points".format(liste[-3].player, liste[-3].score))
    print("\n2__Silver medal is attributed to {} with a total of {} points".format(liste[-2].player, liste[-2].score))
    print("\n1__And the WINNER is {} with a total of {} points!".format(liste[-1].player, liste[-1].score))


##Playing the game
def game(nb, rand=True):
    
    #Creating a set of 100 players with random names and no scores if rand is True
    print("Do you want to write the names of every player (yes) or do you want us to generate random names (no)")
    rep = input()
    if rep=="yes":
        rand=False
    if rand:
        players=[(Node((''.join(random.choice(string.ascii_letters) for i in range(5))))) for j in range(nb)]
    else:
        players=[]
        for i in range(nb):
            print("\nWhat's your name?")
            players.append(Node(input()))
    gameTree = AVLTree()
    while len(players)>10:
        
        # if it's the first game, we create 10 random games of 10 random players bc no scores yet
        # if not, we create games with the remaining ones based on ranking
        if len(players)==nb:
            games = createGames(players, True)
        else:
            games = createGames(players, False)
        
        # We print the composition of every game
        j=1
        for i in games:
            print("\n- Players of game {}".format(j))
            j=j+1
            for k in i:
                print (k)
        print("\nGames are over!")
        
        # The games have ended, we are going to give the players random scores
        # Also, we'll gather everyone in one list
        players = []
        for i in games:
            i = randScore(i)
            players = players + i   
        
        #Now we are going to update our AVL Tree according to our ranking
        gameTree = gameTree.avlFromList(players)
        
        #Here, we will print our ranking and update our players list at the same time
        players = []
        players = gameTree.fillList(gameTree.root, players)
        print("The remaining players are:")
        gameTree.inorder_print(gameTree.root)
        
        #Now we eject the last 10 players and if there are more than 10 players remaining, the while loop starts over
        players = dropPlayers(players)
    
    
    #End of the while loop, which means we have 10 remaining players
    #Let's display our players:
    displayFinals(players)
    
    #Supposing they have played the game, we're going to attribiute new scores
    #Plus, let's update our AVL
    print("Let's play the last game!\n\nOkay, the game is over! Here are the new finalists scores:")
    gameTree = gameTree.avlFromList(randScore(players))
    
    #Now the inorder traversal will give us the final ranking:
    players = []
    players = gameTree.fillList(gameTree.root, players)
    gameTree.inorder_print(gameTree.root)
    
    #We announce the podium
    podium(players)
    
    
game(100)

#%% Step 2

import numpy as np
from collections import defaultdict
from random import shuffle

class Graph(): 
  
    def __init__(self, edges): 
        self.edges = edges 
        self.graph=defaultdict(set)
        #we want to create a list of the node's neighbours
        for start, end in self.edges:       #we go from all the edges
            if start in self.graph:
                self.graph[start].append(end)
                self.graph[start].sort()
            else:
                self.graph[start] = [end]
        print(self.graph)
    
    def get_neighbors(self, node):
        neighbors = []
        for neigh in self.graph[node]:
            if neigh not in neighbors:
                neighbors.append(neigh)
        return neighbors
    
    def imposteur(self, impostor=None):
        set_impostors=[]
        if impostor!=None:
            for i in self.graph:
                if i not in self.get_neighbors(impostor) and i!= impostor :
                    set_impostors.append(i)
            if '0' in set_impostors:
                set_impostors.pop(0)

            print("supposing {} is the impostor, we have {} suspected to be the other impostor".format(impostor, set_impostors))
       
        return set_impostors
    

        
        
if __name__ == '__main__': 
    connections = [("0","1"), ("0","4"), ("0", "5"), ("1","2"), ("1","0"), ("1","6"), ("2","1"), ("2","3"), ("2", "7"), ("3", "4"), ("3", "2"),("3", "8"), 
                   ("4","0"), ("4","3"), ("4","9"), ("5","0"), ("5","7"), ("5","8"), ("6", "1"), ("6","8"), ("6","9"), ("7", "2"), ("7", "5"), ("7", "9"),
                   ("8", "3"), ("8", "5"), ("8", "6"), ("9", "6"), ("9", "7"), ("9", "4")]

    print("We have modeled the situation as a graph such as : ")
    graph=Graph(connections)
    print(" and thanks to this we can define a set of possible impostors by "+ "\n")
    for i in ['1','4','5']:
        result=graph.imposteur(i)
        print("\n")
        
##############################################################################
#                     -- Précision du problème --  
##############################################################################
#we keep only one tuple for each connection 
#because in the fMatAdj(n,LA) we consider that if the player A has seen the player B
#then the player B has seen player A (line 79)
connections2 = [(0,1), (0,4), (0,5), (1,2), (1,6), (2,3), (2, 7), (3, 4),(3, 8), (4,9), (5,7), (5,8), (6,8), (6,9), (7, 9)]  

def fMatAdj(n,LA):
    # n = nb of nodes on the graph 
    # LA is the list of edges between the nodes
    Adj = np.zeros((n,n),dtype=int)     #adjacent matrix
    for k in range(len(LA)):
        i, j = LA[k][0], LA[k][1]
        if len(LA[k]) > 2 or i == j or Adj[i,j] != 0:   #we verify that the list is valid
            return ('The list is no valid !')
        else:
            Adj[i,j], Adj[j,i] = 1, 1
    return Adj


## WELSH-POWELL ALGORITHM
# n = nb of nodes on the graph 
# LA is the list of edges between the nodes
def WP(n,LA):
    M = fMatAdj(n,LA)   #adjacent matrix
    D = []
    for i in range(n):
        d = 0
        for j in range(n):
            if M[i,j] != 0:
                d += 1
        D.append([i,d])
    
    D.sort(key=lambda degre: degre[1])
    shuffle(D)
    
    # coloration
    C = 0
    ColoredVertices = 0
    while ColoredVertices < len(D):
        for i in range(len(D)):
            # the interest is only on the colored nodes
            if len(D[i]) == 2:
                ColPoss = True
                for j in range(i):
                    if len(D[j]) == 3 and D[j][2] == C and M[D[i][0],D[j][0]] == 1:
                        ColPoss = False
                        break
                if ColPoss:
                    D[i].append(C)
                    ColoredVertices += 1
        C +=1
        
    # printing results
    print("Nombre de couleurs utilsées :",C)
    print("Sommets ayant la même couleur :")
    for i in range(C):
        s = "couleur " + str(i) + " : "
        for e in D:
            if e[2] == i:
                s += str(e[0]) + " "
        print(s)
    return C,D
        
      
def freq_impostor():
    nb_game=1000
    
    first_imp=[1,4,5]
    print("we simulated ", nb_game, " games")
    freq=np.zeros((3,10), dtype=int) #color 0 on 1st line, color 1 on line 1 and color 2 on line 3
   
    while(nb_game!=0):
        C,D=WP(10, connections2)
        if C==3:    #we only want to study the 3-colorations
            for e in D: #D is composed of lists in which the first element is the player and the last (e[2]) is the color associated
                #we are looking for the color associated to the 3 first impostors 1, 4 and 5
                if e[0]==1: 
                    c1=e[2]
                if e[0]==4:
                    c2=e[2]
                if e[0]==5:
                    c3=e[2]
            #we want to study the probability only on the color associated to only one of the first impostor
            if c1==c2:
                i=c3
            if c1==c3:
                i=c2
            if c2==c3:
                i=c1
            if c1!=c2!=c3:  #if the 3 first impostors have the same color there is no interest in study the case
                for e in D:
                    if e[2]==i:
                        #we are making differences between the first impostors and the others regarding their probability to be an impostor
                        if e[0] in first_imp:
                            freq[i][e[0]]=freq[i][e[0]]+20
                        else:
                            freq[i][e[0]]=freq[i][e[0]]+5
        nb_game=nb_game-1

    return freq

def guess_impostor():
    #1st step : we keep the max of each first impostor in order to see clearer
    first_imp=[1,4,5]
    maximum=[]
    f=freq_impostor()
    y=0
    for i in first_imp:
        maxi=[]
        for j in range(3):
            maxi.append(f[j][i])
        maximum.append(maxi)
    for i in first_imp:
        for j in range(3):
            if f[j][i]!=max(maximum[y]):
                f[j][i]=0
        y=y+1
    
    #2nd step : we suspect one of the first in particular
    m,m_l,m_i=max(f[0]),0,1
    for j in range(3):
        for i in first_imp:
            if f[j][i]>m:
                m=max(f[j])
                m_l=j
                m_i=i
        #now we know that the first impostor is m_i with color m_l  
    
    #3rd step : we are looking for the second impostor in the same color m_l
    for j in range(len(f[m_l])):
        if j in first_imp:
            f[m_l][j]=0 #since we are looking for the other max on the same line we have to put the first max on 0
            
    m2,m_i2=max(f[m_l]),0
    for i2 in [2,3,6,7,8,9]:
        if f[m_l][i2]==m2:
            m_i2=i2     #the second impostor 

    return m_i, m_i2

def estimate_impostor():
    nb=50
    prob=[0]*10
    n=nb
    
    while(n!=0):
        (i,j)=guess_impostor()
        prob[i]=prob[i]+1
        prob[j]=prob[j]+1
        n=n-1
    print("\nnumber of designated impostor in ", nb, "estimations :", prob)
    
    #choose the final impostors
    imp1_sc=max(prob[1], prob[4], prob[5])
    for i in [1,4,5]:
        if prob[i] == imp1_sc:
            imp1=i
            prob[i]=0
    imp2_sc=max(prob)
    for i in range(10):
        if prob[i] == imp2_sc:
            imp2=i
    return imp1,imp2

#print(WP(10, connections2))
#print("\n",freq_impostor())
#print("\n", guess_impostor())

print("\nFinally, the impostors may be :", estimate_impostor())

print("\nFor sure you should vote for the player", estimate_impostor()[0], "! But watch out, the second impostor may be the player", estimate_impostor()[1], "...")            


#%% Step 3

import pandas 
INF  = float("inf")

labels_crew = ["Upper E" ,"Lower E","Reactor","Security","Medbay","Electrical","Cafetaria","Storage","Admin","O2","Com","Weapons","Navigation","Shield"] 
labels_imp = ["Upper E" ,"Lower E","Reactor","Security","Medbay","Electrical","Cafetaria","Storage","Admin","O2","Com","Weapons","Navigation","Shield","Added Cel"] 

crew = [[0,10,9,8,10,INF,13,INF,INF,INF,INF,INF,INF,INF],
         [10,0,9,8,INF,11,INF,14,INF,INF,INF,INF,INF,INF],
         [9,9,0,7,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF],
         [8,8,7,0,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF],
         [10,INF,INF,INF,0,INF,11,INF,INF,INF,INF,INF,INF,INF],
         [INF,11,INF,INF,INF,0,INF,9,INF,INF,INF,INF,INF,INF],
         [13,INF,INF,INF,11,INF,0,11,10,INF,INF,8,INF,INF],
         [INF,14,INF,INF,INF,9,11,0,9,INF,9,INF,INF,9],
         [INF,INF,INF,INF,INF,INF,10,9,0,INF,INF,INF,INF,INF],
         [INF,INF,INF,INF,INF,INF,INF,INF,INF,0,INF,6,9,13],
         [INF,INF,INF,INF,INF,INF,INF,9,INF,INF,0,INF,INF,6],
         [INF,INF,INF,INF,INF,INF,8,INF,INF,6,INF,0,11,12],
         [INF,INF,INF,INF,INF,INF,INF,INF,INF,9,INF,11,0,12],
         [INF,INF,INF,INF,INF,INF,INF,9,INF,13,6,12,12,0]]

imp = [[0,10,0,8,10,INF,13,INF,INF,INF,INF,INF,INF,INF,INF],
         [10,0,0,8,INF,11,INF,14,INF,INF,INF,INF,INF,INF,INF],
         [0,0,0,7,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF,INF],
         [8,8,7,0,0,0,INF,INF,INF,INF,INF,INF,INF,INF,INF],
         [10,INF,INF,0,0,0,11,INF,INF,INF,INF,INF,INF,INF,INF],
         [INF,11,INF,0,0,0,INF,9,INF,INF,INF,INF,INF,INF,INF],
         [13,INF,INF,INF,11,INF,0,11,0,INF,INF,8,INF,INF,0],
         [INF,14,INF,INF,INF,9,11,0,9,INF,9,INF,INF,9,INF],
         [INF,INF,INF,INF,INF,INF,0,9,0,INF,INF,INF,INF,INF,0],
         [INF,INF,INF,INF,INF,INF,INF,INF,INF,0,INF,6,9,13,8],
         [INF,INF,INF,INF,INF,INF,INF,9,INF,INF,0,INF,INF,6,INF],
         [INF,INF,INF,INF,INF,INF,8,INF,INF,6,INF,0,0,12,10],
         [INF,INF,INF,INF,INF,INF,INF,INF,INF,9,INF,0,0,0,7],
         [INF,INF,INF,INF,INF,INF,INF,9,INF,13,6,12,0,0,5],
         [INF,INF,INF,INF,INF,INF,0,INF,0,8,INF,10,7,5,0]]


def shortestroute(graph): 
    dist = graph
    
    for k in range(len(graph)): 
        for i in range(len(graph)):  
            for j in range(len(graph)): 
                dist[i][j] = min(dist[i][j] , (dist[i][k]+ dist[k][j]))
    return dist



#This method has been created to compare crewmatames' traveling time to impostors'
def compare(graph1,graph2):
    a = 0
    b = 0
    
    while(a not in labels_imp or b not in labels_imp):
        print("\nFrom where are you leaving? Check your spelling, uppercases mather")
        a=input()
        print("\nWhere do you wanna go?")
        b=input()
    
    rep_imp = graph2[labels_imp.index(a)][labels_imp.index(b)]
    
    if (a=="Added Cel" or b=="Added Cel"):
        return "\nA crewmate cannot use this route but it will take %d seconds for an impostor to travel between these rooms"%(rep_imp)
    
    else:
        rep_crew=graph1[labels_imp.index(a)][labels_imp.index(b)]
        return "\nTraveling between these rooms will take %d seconds for a crewmate vs %d for an impostor"%(rep_crew,rep_imp)
    

def interface():
    liste=["1","2"]
    a="0"
    print("Welcome to the shortest route calculator \n")
    print("1_Crewmates graph")
    print(*crew, sep="\n")
    print("\n\n2_Impostors graph")
    print(*imp, sep="\n")
    while a not in liste:
        print("\nChoose the number of the graph you want to apply Floyd-Warshall to (1 or 2")
        a = input()
    print("\n Here you go!\n")
    
    #Here we are creating the solution DataFrame for each model
    x1=pandas.DataFrame((shortestroute(crew)), columns=labels_crew, index=labels_crew)
    x2=pandas.DataFrame((shortestroute(imp)), columns=labels_imp, index=labels_imp)
    
    if a=="1":
        print(x1)
    if a=="2":
        print(x2)    
    print("\nYou can check the full graph on Excel")
    
    print("\nDo you want to see the other graph completed? yes or no")
    b=input()
    if (b=="yes"):
        if a=="1":
            print(x2)
        if a=="2":
            print(x1)
    
    #Checking the itinerary answers the last question of the step, it will help
    #comparing the time to travel from a room to another for crewmates and impostors
    print("\nDo you want to check an itinerary? yes or no")
    rep=input()
    while (rep=="yes"):
        print(compare(shortestroute(crew),shortestroute(imp)))
        print("\nDo you want to check something else? yes or no")
        rep=input()
        
        
interface()

#%% Step 4

INF = float("inf")
mapdgr = [[INF,1.5,INF,INF,INF,INF,INF,2.5,INF,INF,INF,INF,INF,INF],
         [1.5,INF,1.5,3,4.5,INF,INF,4,INF,INF,INF,INF,INF,INF],
         [INF,1.5,INF,1.5,3,INF,INF,INF,INF,INF,INF,INF,INF,INF],
         [INF,3,1.5,INF,4.5,INF,INF,INF,INF,INF,INF,INF,INF,INF],
         [INF,4.5,3,4.5,INF,7,INF,INF,INF,INF,INF,INF,INF,INF],
         [INF,INF,INF,INF,7,INF,4.5,6.5,INF,INF,INF,INF,6,5],
         [INF,INF,INF,INF,INF,4.5,INF,3,INF,INF,INF,INF,INF,INF],
         [2.5,4,INF,INF,INF,6.5,3,INF,2.5,5,INF,INF,INF,INF],
         [INF,INF,INF,INF,INF,INF,INF,2.5,INF,2.5,INF,INF,INF,INF],
         [INF,INF,INF,INF,INF,INF,INF,5,2.5,INF,6,2.5,4.5,INF],
         [INF,INF,INF,INF,INF,INF,INF,INF,INF,6,INF,4.5,4,INF],
         [INF,INF,INF,INF,INF,INF,INF,INF,INF,2.5,2,INF,2.5,INF],
         [INF,INF,INF,INF,INF,6,INF,INF,INF,4.5,5.5,2,INF,3],
         [INF,INF,INF,INF,INF,5,INF,INF,INF,INF,INF,INF,3,INF]]

def findroute(graph):
    final = []
    route = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    for i in route:
        index=[i]
        cout=[]
        l=list(route)
        l.remove(i)
        for a in range(len(l)):
            a=index[-1]
            mini = INF
            for j in l:
                if graph[a][j]<=mini:
                    mini = graph[a][j]
                    b = j
            index.append(b)
            cout.append(mini)
            l.remove(b)
        
        final.append((index,sum(cout)))
     
    return final
    
    

def display():
    print("\nThese are the rooms corresponding to every number")
    print("\n0 : Com\n1 : Shield\n2 : Navigation\n3 : O2\n4 : Weapons\n5 : Cafetaria\n6 : Admin\n7 : Storage\n8 : Electrical\n9 : Lower E\n10 : Security\n11 : Reactor\n12 : Upper E\n13 : Medbay")
    
    a = findroute(mapdgr)
    print("\nIn the following you will see the possible routes and their scores based on every rooms cost")
    for i in a:
        print("\n",i)
    
    result = 0
    for i in a:
        if i[1] == min(k[1] for k in a):
            result = i
            
    print("\nBut the best route is {} with a cost of {}".format(result[0],result[1]))
    

display()