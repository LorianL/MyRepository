def Find_Min(list):
    minIndex=0               #Definiert die Variable x als int oder float
    minNum = list[minIndex]    #Die Variable minNum,bekommt als Startwert die erste Zahl aus einer gegebenen Liste, diese wird dann immer durch die nächst kleiner Zahl ersetz, solange bis keine kleinere Zahl mehr in der Liste ist.

    for i in range(0,len(list)): #iteriert durch alle Indexe der gegebenen Liste durch
        if (list[i] < minNum):   #Vergleicht ob das Element mit dem Index [i] aus einer gegebenen Liste kliner ist als die bislang kleinste gefundene Zahl
            minNum = list[i]    #Falls eine kleinere Zahl als minNum gefunden wird, so wird minNum durch diese Zahl ersetzt
            minIndex = i        #ebenfalls wird der Index dieser neuen kliensten Zahl aktualisiert

    return [minNum, minIndex]   #gibt eine Liste zurück mit der Form [minNum, minIndex]

def NearestWeight(mass, weightsList):
    mass = (mass**2)**0.5
    valueList = []
    for i in range(0,len(weightsList)):
        valueList.append(((mass- weightsList[i])**2)**0.5)
    return Find_Min(valueList)


def CheckIfItFit(weightsList):
    ResultList_Bool = []
    for mass in range(10,10010,10):
        weights = []
        weights.extend(tuple(weightsList))
        deltaMass = 0
        while True:
            difference = mass - deltaMass
            index_deltaWeights = NearestWeight(difference, weights)[1]

            if deltaMass < mass:
                deltaMass += weights[index_deltaWeights]
            elif deltaMass > mass:
                deltaMass -= weights[index_deltaWeights]
            if deltaMass == mass:
                ResultList_Bool.append(True)
                break
            elif len(weights) == 1:
                ResultList_Bool.append(False)
                break

            weights.pop(index_deltaWeights)
    return ResultList_Bool

def Transform_File(FileName):
    file = open(FileName, "r").readlines()
    weightsList = []
    for line in file[1:]:
        line=line.strip()
        lineElements = line.split(" ")
        for i in range(0,int(lineElements[1])):
            weightsList.append(int(lineElements[0]))
    return weightsList

def printResult(List, FileName):
    counter = 0
    for i in List:
        if i:
            counter += 1
    print(f"Mit Hilfe von den Gewichten aus \"{FileName.upper()}\", kann man {counter} der {len(List)} möglich Gewichte nachstellen, das entspricht einer Quote von {round(counter/len(List)*100,2)}%")

if __name__ == '__main__':
    exampleFiles=["gewichtsstuecke0.txt","gewichtsstuecke1.txt","gewichtsstuecke2.txt","gewichtsstuecke3.txt","gewichtsstuecke4.txt",]
    for FileName in exampleFiles:
        weightsList = Transform_File(FileName)
        BoolList = CheckIfItFit(weightsList)   #return List with Bool Values of all tested Inputs (Intputs are between 10 and 10000 with distance of 10)
        printResult(BoolList, FileName)
