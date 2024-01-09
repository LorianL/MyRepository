from random import randint

def Find_Max(list):
    maxIndex=0               #Definiert die Variable x als int oder float
    maxNum = list[maxIndex]    #Die Variable maxNum,bekommt als Startwert die erste Zahl aus einer gegebenen Liste, diese wird dann immer durch die nächst grössere Zahl ersetzt, solange bis keine grössere Zahl mehr in der Liste ist.

    for i in range(0,len(list)): #iteriert durch alle Indexe der gegebenen Liste durch
        if (list[i] > maxNum):   #Vergleicht ob das Element mit dem Index [i] aus einer gegebenen Liste grösser ist als die bislang grösste gefundene Zahl
            maxNum = list[i]    #Falls eine grössere Zahl als maxNum gefunden wird, so wird maxNum durch diese Zahl ersetzt
            maxIndex = i        #ebenfalls wird der Index dieser neuen grössten Zahl aktualisiert

    return [maxNum, maxIndex]   #gibt eine Liste zurück mit der Form [maxNum, maxIndex]


def IndexesNumbers(l): #Diese Funktion gibt die Indexe aller MaxZahlen wieder, erforder das vorhanden sein von der Funktion Find_Max()
    ListOfIndexes = []
    LastIndex = 0
    while True:
        try:
            index = l.index(Find_Max(l)[0],LastIndex)
            LastIndex = index +1
            ListOfIndexes.append(index)
        except:
            break
    return ListOfIndexes


def SortIndexes_MaxToMin(list): #Erstellt eine neue Liste mit den Indexen der Ausgangsliste sortiert von dem Grössten Wert bei [0] und dem kleinsten am Ende
    SortedIndex = [0] #Liste in die Indexe nach ihren Werten absteigen einsortiert werden, der Wert von Index [0] der Liste dient als Referenz und Start für die anderen Werte
    for index in range(1, len(list)): #Aufgrund des Startindexes [0] kann dieser Übersprungen werden
        for element in SortedIndex: #Iterier durch die Liste der Sortierten Liste, von dem grössten Wert zum kleinsten, solange bis ein Wert gefunden wurde der kleiner ist als der bereits Sortierte, zu diesem Zeitpunk wird diese Schleife abgebrochen
            if list[index]>list[element]:#Überprüft ob der Wert vom Index grösser ist als der von Element. Ist dies True so wird folgende Operation ausgelöst
                SortedIndex.insert(SortedIndex.index(element),index) #Zum einen wird der Index von Element gesucht um anschliessend den gegebenen Index vor den kleineren zu schieben
                break # anschliessend wird die Scheilfe abgebrochen, da Index jetzt in der SortedIndex List vorhanden ist
        if list[index]<list[SortedIndex[-1]]: #falls der Wert kleiner ist als alle Werte von SortedIndex, so wird Index an die letzte stelle der Liste SortedIndex eingefügt
            SortedIndex.append(index)
    return SortedIndex #gibt die vollständig sortierte Liste zurück

def isNegatif(num):
    if num >= 0:
        return False
    elif num < 0:
        return True


def Find_Pairs(list): #diese Funktion soll aus der Liste jeweils Paare finde, welche sich jedoch nicht wieder holen sollen, für weiter Informationen kucke Herleitung; diese Funktion solle eine Liste zurück geben in der jeweils wieder Lsiten mit der Länge sind, welche die Indexe enthalten
    EndList = []
    for i in range(0,len(list)-1): #zum Verständnis, schaue die Herleitung, vereinfacht gesagt zieht man ebenfalls ein ab zum ausgleich von i+1
        for j in range(i+1,len(list)):
            EndList.append([i,j])
    return EndList #Der zurückgegebene Wert ist eine Matrix aus Indexen

#-----------------------------------------------------------------------------------------------------------------------------

class Figure:
    def __init__(self):
        self.position = -1

    def goToA(self):
        self.position = 0

    def goBackToB(self):
        self.position = -1

    def move_forward(self, distance):
        self.position += distance

#------------------------------------------------------------------------------------------------------------------------------

class Player:
    def __init__(self,dice_numbers,name):
        self.dice_numbers = dice_numbers
        self.fList = [Figure(),Figure(),Figure(),Figure()]
        self.name = name


    def random_dice(self):
        randomIndex = randint(0,len(self.dice_numbers)-1) #Züfälliger Index aus der Länge der Liste des Würfels. Da die Methode len() bei 1 anfängt zu zählen und die Indexe bei 0 anfangen muss man minus 1 rechnen um diesen Unterschied auszugleichen
        return self.dice_numbers[randomIndex]

    def control_Movment(self,randomNum): #entscheidet ob eine Figure auf das A Feld geht, oder ob eine Figur vom A Feld weiterläuft

        if Player.CheckFieldA(self)[0] and Player.CheckMovement(self,Player.CheckFieldA(self)[1],randomNum):
            self.fList[Player.CheckFieldA(self)[1]].move_forward(randomNum) #Führt die Funktion move_forward() von der Figur aus, welche auf dem Feld A steht; mit self.fList, greift man auf die Liste der Figuren zu und mit CheckFieldA(self)[1] bekommt man den Index der Figur auf der Position A wieder
            #print("hat sich von Feld A entfernt")

        elif (randomNum == 6 and Player.CheckFieldA(self)[0] != True and Player.CheckFieldB(self)[0]):
            #print(f"Player{self.name} hat eine Figur aus dem Haus")
            self.fList[Player.CheckFieldB(self)[1]].goToA()

        else:
            Player.move_Figure(self,randomNum)




    def CheckFieldA(self): #"""Note"""
        for i in range(0,len(self.fList)): #Kontolliert ob eine Figur auf dem Feld 0 steht
            if self.fList[i].position == 0:
                return [True,i] #gibt zum einen zurück ob eine Figur auf B ist und zum anderen dden Index von fList dieser Figur
        return [False, None]
    def CheckFieldB(self): #"""Note"""
        for i in range(0,len(self.fList)): #Kontolliert ob eine Figur auf dem Feld -1 (auch die Felder B genannt) steht
            if self.fList[i].position == -1:
                return [True,i] #gibt zum einen zurück ob eine Figur auf B ist und zum anderen dden Index von fList dieser Figur
        return [False, None]

    def CheckMovement(self,fListIndex,movement_range): #""""NOTE"""#Diese Funktion soll überprüfen, ob der Spieler bei einem Zug auf ein Feld kommt auf dem bereits einer seiner Figuren steht
        if fListIndex == None:
            return False
        deltaPosition = self.fList[fListIndex].position + movement_range #Die Variable deltaPosition gibt an wo sich die Figur nach dem Zug befinden würde
        for f in self.fList:
    #        print(f.position)
            if f.position in range(0,44) and deltaPosition == f.position: #Überprüft als erstes ob f.position zwische 0 und 43 liegt, anschliessend wird überprüft, ob deltaPosition den Gleichen Wert wie f.position hat
                    return False #gibt den Wert False zurück, was soviel bedeutet, wie der Zug ist nicht möglich
        return True

    def move_Figure(self, randomNum): #"""Note"""
        positionList = [] #als erstes Bestimmen wir welche Figur am weitesten vorne liegt
        for figure in self.fList:
            positionList.append(figure.position)
        sortedPostition = SortIndexes_MaxToMin(positionList) #Gibt eine List mit den Sortierte Indexen zurücke, welche nach ihren Postionen absteigend sortiert sind
        for positionIndex in sortedPostition:
            if (self.fList[positionIndex].position + randomNum) in range(1,44) and self.fList[positionIndex].position in range(1,44) and Player.CheckMovement(self,positionIndex,randomNum): #überpruft ob die Figur nach dem Zug noch auf dem Spielfeld ist
                self.fList[positionIndex].move_forward(randomNum) #bewegt die Figur nach vorne
                break

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

class GameField:
    def __init__(self,Player1,Player2):
        self.pList = [Player1,Player2]
        self.lastMove = []
        self.control_Bool = True
        self.Result = None
        self.TurnCounter = 0

    def Decide_FirstPlay(self): # dieses Program entscheidet wer anfängt, dabei gilt, wert die grösse Zahl hat darf anfangen
        FirstPlay_control_Bool = True
        while(FirstPlay_control_Bool): #Diese Schleife dient dazu, falls beide Würfel den gleichen Wert haben, dass so lange gewürfel wird bis unterschieliche Wert herauskommen
            p1_startnumber = self.pList[0].random_dice()
            p2_startnumber = self.pList[1].random_dice()
            if p1_startnumber > p2_startnumber:
                FirstPlay_control_Bool = False
                GameField.GameControl(self,0,1)
            elif p1_startnumber < p2_startnumber:
                FirstPlay_control_Bool = False
                GameField.GameControl(self,1,0)


    def GameControl(self,first,second):
        while(self.control_Bool):
            while True:
                Firstplayer_Dice = self.pList[first].random_dice()
                GameField.PlayerController(self,first,Firstplayer_Dice)

                if GameField.control_Winner(self)[0] or Firstplayer_Dice != 6:
                    break
            if GameField.control_Winner(self)[0]:
                break
            #-----------------------------------------------------
            while True:
                Secondplayer_Dice = self.pList[second].random_dice()
                GameField.PlayerController(self,second,Secondplayer_Dice)
                if GameField.control_Winner(self)[0] or Secondplayer_Dice != 6:
                    break
            if GameField.control_Winner(self)[0]:
                break
            #----------------------------------------------------------
            if self.TurnCounter >= 1000:
                self.control_Bool = False
                self.Result = None #Falls es zu keinem Ergebnis kommen sollte und die Runde länger als 1000 Runden für jeden dauern sollte, so wird der Wert None zurück gegeben; Dies kommt nur vor, wenn keiner der beiden Würfel, ins Ziel kommt
                break
            #print(self.TurnCounter)
            self.TurnCounter += 1
        self.Result = GameField.control_Winner(self)[1]

    def PlayerController(self,status,dice): #mit Status ist gemein ab es first oder second ist und mit dice ist die zufällige Zahl gemeint
        self.pList[status].control_Movment(dice)
        GameField.control_LastMove(self,status)


    def control_LastMove(self, lastplayedPlayer): # diese Funktion überprüft ob der lastplayedPlayer auf dem Feld einer gegnerischen Figur ist #die Variable lastplayedPlayer gibt den Index des letzteb gespielten Spieler
        if lastplayedPlayer == 0: #überprüft ob der Player1 gespielt hat
            FiguresPositions_OfPlayer1 = [] #in diese Variable werden die Positionen der Figuren von Player1 gespeichert
            for i in self.pList[0].fList: #iteriert durch die Positionen aller Figuren von Player1 um sie der Liste beizufügen
                if i.position in range(0,40):
                    enemyposition = i.position -20 #diese Variable beschreibt die Position aus der sicht des Gegners, da das Spielfeld 40 Felder Lang ist und die Spieler sich gegenüber stehen. für genauere Info Erklärungsdokument auf Goodnotes
                    if isNegatif(enemyposition): #falls die enemyposition negatif ist, so wirdd 40 drauf addiert damit die Zahl wieder positif ist
                        enemyposition = enemyposition +40
                    FiguresPositions_OfPlayer1.append(enemyposition) #es werden nur Positionen hinzugefügt, welche auf dem Spielfeld stehen, bedeutet dass die Figuren sich zwischen 0 und 39 befinden
            for figure in self.pList[1].fList: #iteriert durch alle gegnerischen Figuren durch
                if figure.position in FiguresPositions_OfPlayer1: #falls die Position der Figur mit einer der Positionen der des Player1 übereinstimmt, so muss die Figur wieder zurück auf -1
                    figure.goBackToB()
        elif lastplayedPlayer == 1: #überprüft ob der Player2 gespielt hat
            FiguresPositions_OfPlayer2 = [] #in diese Variable werden die Positionen der Figuren von Player2 gespeichert
            for i in self.pList[1].fList: #iteriert durch die Positionen aller Figuren von Player2 um sie der Liste beizufügen
                if i.position in range(0,40):
                    enemyposition = i.position -20 #diese Variable beschreibt die Position aus der sicht des Gegners, da das Spielfeld 40 Felder Lang ist und die Spieler sich gegenüber stehen. für genauere Info Erklärungsdokument auf Goodnotes
                    if isNegatif(enemyposition): #falls die enemyposition negatif ist, so wirdd 40 drauf addiert damit die Zahl wieder positif ist
                        enemyposition = enemyposition +40
                    FiguresPositions_OfPlayer2.append(enemyposition) #es werden nur Positionen hinzugefügt, welche auf dem Spielfeld stehen, bedeutet dass die Figuren sich zwischen 0 und 39 befinden
            for figure in self.pList[0].fList: #iteriert durch alle gegnerischen Figuren durch
                if figure.position in FiguresPositions_OfPlayer2: #falls die Position der Figur mit einer der Positionen der des Player2 übereinstimmt, so muss die Figur wieder zurück auf -1
                    figure.goBackToB()



    def control_Winner(self):
        if (self.pList[0].fList[0].position in range(40,44) and self.pList[0].fList[1].position in range(40,44) and self.pList[0].fList[2].position in range(40,44) and self.pList[0].fList[3].position in range(40,44)):
            self.control_Bool = False #Kontolliert ob ein Spieler gewonnen hat, in dem geschaut wird op jede Figur im Ziel ist
            #print(f"Player {self.pList[0].name} hat gewonnen mit dem Würfel", self.pList[0].dice_numbers)
            return [True, self.pList[0].name]
        elif self.pList[1].fList[0].position in range(40,44) and self.pList[1].fList[1].position in range(40,44) and self.pList[1].fList[2].position in range(40,44) and self.pList[1].fList[3].position in range(40,44):
            self.control_Bool = False
            #print(f"Player {self.pList[1].name} hat gewonnen mit dem Würfel", self.pList[1].dice_numbers)
            return [True, self.pList[1].name] #bestätigt zuerst, dass es einen Sieger gibt und sagt anschliessend wer es ist
        return [False, None] #gibt zurück, dass es noch keinen Sieger gibt

#---------------------------------------------------------------------------------------------------------------------------------------------------------

def Simulation(dice1, dice2,name1,name2): #es ist wichtig zu beachten, dass ein Würfel immer mindestens zwei verschiedene Zahlen hat, sonst könnte es zu Problemen kommen, zum Beispiel beim Würfel [6,6,6,6,6,6] entsteht eine Endlosschleife und das Program kommt zu keinem Ergebniss
    p1 =Player(dice1,name1)
    p2 = Player(dice2,name2)
    Game = GameField(p1,p2)
    x=Game.Decide_FirstPlay()
    return Game.Result

def GameIteration(dice1,name1,dice2,name2,iterations): #iterations muss eine natürliche Zahl sein
    WinnerList = [0,0] # in dieser Liste wird gezählt welcher Würfel wie oft gewinnt
    for i in range(0,iterations):
        SimulationResult = Simulation(dice1,dice2,0,1) #speichert das Ergebniss der Simulation
        if SimulationResult != None:
            WinnerList[SimulationResult] += 1
    print("...")#print(WinnerList)
    if WinnerList[0] > WinnerList[1]: #Überprüft welcher Würfel insgesamt mehr Siege hat
        return name1
    elif WinnerList[0] < WinnerList[1]:
        return name2
    elif WinnerList[0] == WinnerList[1]:
        print("Equal")
        return None

def GameInitialisator(dicesList,iterations):
    pairList = Find_Pairs(dicesList)
    DiceWinsCounter = [] #in dieser Liste werden die Wins gezähl um am Ende zubestimmen, welcher Würfel der beste ist.
    for j in range(0,len(dicesList)):#setzt alle Startwerte für den Wins Counter auf 0
        DiceWinsCounter.append(0)

    for i in pairList: #i besitzt folgende Strucktur [index des 1 Würfels, index des 2 Würfels]
        winner = GameIteration(dicesList[i[0]],i[0],dicesList[i[1]],i[1],iterations) #als name benutzen wir den Index der jeweiligen Würfel, da man sie danach diesen wieder zuordnen kann
        if winner != None: #überprüft ob winner kein None ist. None steht für gleichstand und würde eine Fehlermeldung verursachen; bei einem Gleichstand hat keiner gewonnen, deswegen bekommt keiner ein Punkt
            DiceWinsCounter[winner] +=1 #erhöht den counter für den Würfel der gewonnen hat
    return DiceWinsCounter

def Transform_File(filename):
    lines = open(filename, "r").readlines()
    dicesList = [] #in diese Liste werden die einzelnen Würfel eingefügt
    for i in range(1, len(lines)): #iteriert durch die Zeilen, die erste wird übersprungen, da sie für uns nicht von relevanz ist
        line = lines[i].strip() #entfernt das \n und fügt den Korrekten String in die Variable line
        dice_numbers = line.split(" ") #erstellt eine Liste, diese besteht aus dem Sting line, welcher immer bei " " gespalten worden ist
        for j in range(0,len(dice_numbers)): #iteriert durch die Liste dice_numbers mit hilfe der Indexe
            dice_numbers[j] = int(dice_numbers[j]) #wandelt alle Element der Liste dice_numbers von Strings in int() um
        del dice_numbers[0] #die erste Zahl aus der Zeile n gibt die Anzahl der Seiten des Würfels an, was bei unsere Funktion irrelevant ist und deshalb entfernt wird
        dicesList.append(dice_numbers) #fügt jeden Würfel zur Liste dicesList hinzu
    return dicesList

def FindGlobalWinner(results,f,filename):
    x = Find_Max(resultList) #gibt [maxNum, maxIndex] zurück
    if results.count(x[0]) == 1: #Überprüft ob es nicht zwei Sieger gibt
        print(f"In der Würfelliste \"{filename.upper()}\" ist der Würfel {f[x[1]]} der Beste mit {x[0]} gewonnen Duellen")
    elif x[0] == 0: #überprüft ob alle Würfel immer Gleichstand hatten """NOTE""" dieser Teil des Codes wurde noch nicht getestet und könnte zu Problemen führen
        print(f"keiner der Würfel aus \"{filename.upper()}\" ist in der Lage ein Spiel zu gewinnen")
    elif results.count(x[0]) > 1: #überpruft ob mehrer Sieger gewonnen haben und ehrt alle
        indexList = IndexesNumbers(results)
        dicesList = []
        for i in indexList:
            dicesList.append(f[i])
        print(f"In der Würfelliste \"{filename.upper()}\" sind die Würfel {dicesList} die Besten, alle haben {x[0]} Duelle gewonnen")


if __name__ == '__main__':
    filename = ["wuerfel0.txt","wuerfel1.txt","wuerfel2.txt","wuerfel3.txt"]
    for fName in filename:
        f = Transform_File(fName)
        resultList = GameInitialisator(f,100)
        winnerDice = FindGlobalWinner(resultList,f,fName)


"""
ZUSÄTZLICHE REGEL, MAXIMAL 1000 ZÜGE FÜR JEDEN SPIELER
"""




















"""
GameIteration(dicesList[0],0,dicesList[1],1,100)
dice1 = [1,2,3,3,2,1]
dice2 = [1,2,3,4,5,6]
dicesList = [[2,3,4,5,6,7],[3,4,5,6,7,8]]
GameIteration(dicesList[0],0,dicesList[1],1,1)

Simulation(dice1,dice2,0,1)
Stochastik_Analysation


 and f.position == self.fList[fListIndex].position #kaputter Code aus Zeile 101
"""
