def Find_Min(list):
    minIndex=0               #Definiert die Variable x als int oder float
    minNum = list[minIndex]    #Die Variable minNum,bekommt als Startwert die erste Zahl aus einer gegebenen Liste, diese wird dann immer durch die nächst kleiner Zahl ersetz, solange bis keine kleinere Zahl mehr in der Liste ist.

    for i in range(0,len(list)): #iteriert durch alle Indexe der gegebenen Liste durch
        if (list[i] < minNum):   #Vergleicht ob das Element mit dem Index [i] aus einer gegebenen Liste kliner ist als die bislang kleinste gefundene Zahl
            minNum = list[i]    #Falls eine kleinere Zahl als minNum gefunden wird, so wird minNum durch diese Zahl ersetzt
            minIndex = i        #ebenfalls wird der Index dieser neuen kliensten Zahl aktualisiert

    return [minNum, minIndex]   #gibt eine Liste zurück mit der Form [minNum, minIndex]


def transform_Files(filename):
    file = open(filename, "r").readlines() #speicher die Zeilen der Datei "filename" in eine Liste.
    linesList = []          #Definiert die Variable als Array
    housesCoordinates = []
    windmillCoordinates = []
    for line in file:           # in dieser for Schleife wird das \n aus den Zeilen entfernt und in die neue Variable linesList gespeichert
        linesList.append(line.strip())
    nEnd = int(linesList[0].split(" ")[0])+1                    #int(linesList[0].split(" ")[0])+1 -> Diese Aussage gibt an, in welcher Zeile die letzte N Koordinaten liegen. Dies geschieht in dem die Variable N aus der Datei isoliert wird und anschließend noch eins zu N hinzuaddiert, da die erste Zeile übersprungen wird
    for i in range(1,nEnd):                                     #"range(1,nEnd)" -> dies besagt, dass die Loop von der zweiten Zeile an bis zum letzten N-Koordinatenpaar durch iterieren soll
        coordinates = linesList[i].split()                       #Die Zeile wird bei " " gespalten so dass eine Liste entsteht bei der die x-Koordinate auf [0] ist und die y-Koordinate auf [1]
        for k in range(0,2):                                    #lässte k die Werte 0 und 1 annehmen
            coordinates[k] = int(coordinates[k])                #ändert die Variable coordinates[k] von einem Sting in ein int
        housesCoordinates.append(coordinates)                    #Diese Liste wird in die Liste housesCoordinates eingefügt, so dass eine Matrix entsteht, mit den Verschiedenen Hauskoordinaten
    for i in range(nEnd,len(linesList)):                         #diese for Schleife zählt von n bis zum Ende durch, dies wird dadurch erreicht, dass die range() den Anfangswert N hat und den
        coordinates = linesList[i].split()                      #Die Zeile wird bei " " gespalten so dass eine Liste entsteht bei der die x-Koordinate auf [0] ist und die y-Koordinate auf [1]
        for k in range(0,2):                                    #lässte k die Werte 0 und 1 annehmen
            coordinates[k] = int(coordinates[k])                #ändert die Variable coordinates[k] von einem Sting in ein int
        windmillCoordinates.append(coordinates)                 #Diese Liste wird in die Liste windmillCoordinates eingefügt, so dass eine Matrix entsteht, mit den Verschiedenen Windräderkoordinaten
    return housesCoordinates, windmillCoordinates               #gibt die beiden Matrizen zurück, zuerst die der Häuser und anschließend die der Windräder

def Windmills_MaxHeight(nList, mList): #nList steht für die Liste der Koordinaten der Häuser und mList für die der Windräder; diese Funktion berechnet die maximale Höhe für die jeweiligen Windräder
    hList = []           #in diese Liste kommen die maxmal Höhen der Windräder stehen
    for j in range(0,len(mList)):     #iteriert durch alle Koordinaten der Windräder durch
        elementListofN = [] #in dieser Liste werden die zwischen Ergebnisse abgespeichert
        for i in range(0,len(nList)): #iteriert für jedes Windrad einmal durch alle Koordinaten der Häuser durch
            a = nList[i][0]-mList[j][0]        #berechnet die Differenz der einzelnen x-Koordinaten
            b = nList[i][1]-mList[j][1]        #berechnet die Differenz der einzelnen y-Koordinaten
            c = ((a**2) + (b**2))**0.5         #die beiden Differenzen a und b stehen sich in einem 90 Grad winkel gegenüber womit man den Satz des Pythagoras anwenden kann um die Entfernung des Windrades zum jeweiligen Haus zu berechenen
            elementListofN.append(c)           #fügt die Entfernung zwischen dem jeweiligen Haus und dem Windrad in die Variable elementListofN
        cMin = Find_Min(elementListofN)[0]     #Finde die kleinste Zahl auf der Liste elementListofN mithilfe der Funktion Find_Min()
        hMax = cMin / 10                       #Berechne die maximale Höhe des Windrades unter Anwendung der 10h Regel
        hList.append(hMax)                     #Füge die maximale Höhe für das Windrad j in die Liste hList
        print(hMax,mList[j])
    return hList                               #Gibt die Liste mit den Werten der Maximalen Windradhöhen zurück


if __name__ == "__main__":
    FileNameList = ["landkreis1.txt","landkreis2.txt","landkreis3.txt","landkreis4.txt"]

    for filename in FileNameList:
        print("\n\n\n")
        housesList, windmillList = transform_Files(filename)
        #print(housesList, windmillList)
        Windmills_MaxHeight(housesList,windmillList)
