
def Find_Max(list):
    maxIndex=0               #Definiert die Variable x als int oder float
    maxNum = list[maxIndex]    #Die Variable maxNum,bekommt als Startwert die erste Zahl aus einer gegebenen Liste, diese wird dann immer durch die nächst grössere Zahl ersetzt, solange bis keine grössere Zahl mehr in der Liste ist.

    for i in range(0,len(list)): #iteriert durch alle Indexe der gegebenen Liste durch
        if (list[i] > maxNum):   #Vergleicht ob das Element mit dem Index [i] aus einer gegebenen Liste grösser ist als die bislang grösste gefundene Zahl
            maxNum = list[i]    #Falls eine grössere Zahl als maxNum gefunden wird, so wird maxNum durch diese Zahl ersetzt
            maxIndex = i        #ebenfalls wird der Index dieser neuen grössten Zahl aktualisiert

    return [maxNum, maxIndex]   #gibt eine Liste zurück mit der Form [maxNum, maxIndex]

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


class Hotel:
    def __init__(self,distance,rating,index):
        self.distance = distance
        self.rating = rating
        self.index = index
        self.deltaDistance = distance #fEhler anfällig nicht verwenden


    def get_deltaDistance(self, lastDistance):
        self.deltaDistance = self.distance - lastDistance
        return self.deltaDistance

class GetPathRating:
    def __init__(self,HotelsList):
        self.HotelsList = HotelsList
        self.HotelsNum = len(self.HotelsList)
        self.averageRating = GetPathRating.get_AverageRating(self)

    def get_AverageRating(self):
        Sum = 0
        for i in range(0,self.HotelsNum):
            Sum += self.HotelsList[i].rating
        return Sum / self.HotelsNum

class BestPathRating:
    def __init__(self, HotelsList, entireDistance,maxIterations=30):
        self.hList = HotelsList
        self.entireDistance = entireDistance
        self.maxIterations = maxIterations
        self.mode = 0
        self.running_Variable = [0,BestPathRating.adjust_maxIterations(self)]
        self.CatchErrorsPath = GetPathRating([Hotel(0,0,0)]) #dieser Path soll Errors verhindern, welche auftreten könnten, falls eine abzweigung keine mögliche Lösung zurück gibt.
        self.bestRatedPath = BestPathRating.Day1(self)
        BestPathRating.Control_bestRatedPath(self)


    def Control_bestRatedPath(self):
        if self.bestRatedPath == self.CatchErrorsPath:
            self.mode = 1
            self.running_Variable[0] = 0
            PathRating = BestPathRating.Day1(self)
            self.bestRatedPath = PathRating
            if PathRating == self.CatchErrorsPath:
                print("\nes existiert keine Reiseroute mit ihren Eingaben\n")



    def Day1(self):
        Day1HotelsList = BestPathRating.getList(self,None) #erstellt eine Liste mit allen Hotels, welche für Tag 1 infrage kommen und speichert diese in Day1HotelsList
        Day1HotelsList = BestPathRating.optimizeList(self, Day1HotelsList)
        PathRatingsFromDay1 = []
        for h in Day1HotelsList:
            print("\n",30*"-",f"\n {(Day1HotelsList.index(h)+1)} / {len(Day1HotelsList)}","\n",30*"-","\n")
            previousHotels = [] #erstellt eine Liste mit den Vorherigen Hotels, diese soll im Laufe der Schleife immer wieder angepasst und erweitert werden
            previousHotels.append(h) #fügt als erstes das Ursprungs Hotel h hinzu
            if BestPathRating.TomorrowAtDestination(self,None) == False: #überprüft ob wir morgen bereits am Ziel ankommen, sollte das nicht der Fall sein:
                PathRating = BestPathRating.Day2(self,previousHotels)
                PathRatingsFromDay1.append(PathRating) #so werden die besten Hotels für den nächsten Tag gesucht, und die jeweils besten in die Liste PathRatingsFromDay1 hinzugefügt
            else:
                PathRating = GetPathRating(previousHotels)
                PathRatingsFromDay1.append(PathRating)
        return  BestPathRating.Find_BestRatingInList(self,PathRatingsFromDay1)


    def Day2(self,previousHotels):
        Day2HotelsList = BestPathRating.getList(self,previousHotels[-1]) #erstellt eine Liste mit allen Hotels, welche für Tag 2 infrage kommen previousHotels[-1] ist äquivalent mit dem letzten Hotel
        Day2HotelsList = BestPathRating.optimizeList(self, Day2HotelsList)
        PathRatingsFromDay2 = [self.CatchErrorsPath]
        for h in Day2HotelsList: #diese Funktion soll die besten Hotels für PathRatingsFromDay2 finden
            preHotels = []
            preHotels.extend(tuple(previousHotels))
            preHotels.append(h) #fügt zusätzlich das Hotel h hinzu
            if BestPathRating.TomorrowAtDestination(self,h) == False: #überprüft ob wir morgen bereits am Ziel ankommen, sollte das nicht der Fall sein: #lastHotel ist in diesem Fall h
                PathRatingsFromDay2.append(BestPathRating.Day3(self,preHotels)) #so werden die besten Hotels für den nächsten Tag gesucht und zur Liste der PathRatingsFromDay2 hinzugefügt
            else:
                PathRating = GetPathRating(preHotels)
                PathRatingsFromDay2.append(PathRating)
            print(f"{round(self.running_Variable[0] / self.running_Variable[1] *100,2)}% | {self.running_Variable[0]} / {self.running_Variable[1]}")
            self.running_Variable[0] += 1

        return BestPathRating.Find_BestRatingInList(self,PathRatingsFromDay2) #gibt den Weg mit dem besten Rating an die Funktion Day1 zurück



    def Day3(self,previousHotels):
        Day3HotelsList = BestPathRating.getList(self,previousHotels[-1]) #erstellt eine Liste mit allen Hotels, welche für Tag 3 infrage kommen previousHotels[-1] ist äquivalent mit dem letzten Hotel
        Day3HotelsList = BestPathRating.optimizeList(self, Day3HotelsList)
        PathRatingsFromDay3 = [self.CatchErrorsPath]
        for h in Day3HotelsList: #diese Funktion soll die besten Hotels für PathRatingsFromDay3 finden
            preHotels = []
            preHotels.extend(tuple(previousHotels))
            preHotels.append(h) #fügt zusätzlich das Hotel h hinzu
            if BestPathRating.TomorrowAtDestination(self,h) == False: #überprüft ob wir morgen bereits am Ziel ankommen, sollte das nicht der Fall sein: #lastHotel ist in diesem Fall h
                PathRatingsFromDay3.append(BestPathRating.Day4(self,preHotels)) #so werden die besten Hotels für den nächsten Tag gesucht und zur Liste der PathRatingsFromDay3 hinzugefügt
            else:
                PathRating = GetPathRating(preHotels)
                PathRatingsFromDay3.append(PathRating)

        return BestPathRating.Find_BestRatingInList(self,PathRatingsFromDay3) #gibt den Weg mit dem besten Rating an die Funktion Day2 zurück


    def Day4(self,previousHotels):
        Day4HotelsList = BestPathRating.getList(self,previousHotels[-1]) #erstellt eine Liste mit allen Hotels, welche für Tag 4 infrage kommen previousHotels[-1] ist äquivalent mit dem letzten Hotel
        Day4HotelsList = BestPathRating.optimizeList(self, Day4HotelsList)
        PathRatingsFromDay4 = [self.CatchErrorsPath]
        for h in Day4HotelsList: #diese Funktion soll die besten Hotels für PathRatingsFromDay4 finden
            preHotels = []
            preHotels.extend(tuple(previousHotels))
            preHotels.append(h) #fügt zusätzlich das Hotel h hinzu
            if BestPathRating.TomorrowAtDestination(self,h): #überprüft ob wir morgen am Ziel ankommen, sollte das  der Fall sein: #lastHotel ist in diesem Fall h
                PathRating = GetPathRating(preHotels)
                PathRatingsFromDay4.append(PathRating) #so wird der Weg zur Liste PathRatingsFromDay4 hinzugefügt
            else: #Falls wir am nächsten Tag nich da wären, so würden wir die Deadline sprengen
                continue

        return BestPathRating.Find_BestRatingInList(self,PathRatingsFromDay4) #gibt den Weg mit dem besten Rating an die Funktion Day3 zurück



    def getList(self,lastHotel): #gibt eine Liste zurück mit allen infrage kommenden Hotels
        returnList = []
        if lastHotel == None: #Falls lastHotel == None ist, so wird lastHotelDistance auf 0 gesetzt, dies wird dazu gebraucht, da es an Tag1 nach kein letztesHotel gibt
            lastHotelDistance=0
        else:
            lastHotelDistance = lastHotel.distance
        for hotel in self.hList: #iteriert durch alle Hotels aus der Liste hList durch
            if hotel.distance > lastHotelDistance and hotel.get_deltaDistance(lastHotelDistance) in range(1,361): #falls das Hotel weiter entfernt ist als das letzte Hotel und wenn das Hotel zwischen 1 und 360 Minuten vom letzten entfernt ist dann:
                returnList.append(hotel) #wird es zur Liste hinzugefügt
        return returnList

    def optimizeList(self, hotelsList):
        if self.mode == 0: #mode 0 ist der Normale Modus welcher beim ersten Durchlauf verwendet wird
            inputList = []
            for hotel in hotelsList:
                inputList.append(hotel.rating) #erstellt eine Liste mit den Ratings
            indexList = SortIndexes_MaxToMin(inputList) #Sortiere Liste nach ihren ratings. Gibt nur die Indexe sortiert zurück
            returnList = []
            for index in indexList: #Wandelt die Indexe wieder in Objekte des Types Hotel um
                returnList.append(hotelsList[index])
            if self.maxIterations != None:
                returnList = returnList[:self.maxIterations] #falls maxIterations nicht None ist so gebe nur eine Liste mit der länge maxIterations zurück
                return returnList
            return returnList #wenn maxIterations == None ist so gebe die vollständige Liste zurück
        elif self.mode == 1 and self.maxIterations != None:
            inputList = []
            for hotel in hotelsList:
                inputList.append(hotel.distance) #erstellt eine Liste mit den Ratings
            indexList = SortIndexes_MaxToMin(inputList) #Sortiere Liste nach ihren ratings. Gibt nur die Indexe sortiert zurück
            returnList = []
            for index in indexList: #Wandelt die Indexe wieder in Objekte des Types Hotel um
                returnList.append(hotelsList[index])
            returnList = returnList[:self.maxIterations] #falls maxIterations nicht None ist so gebe nur eine Liste mit der länge maxIterations zurück
            return returnList

        elif self.mode == 1 and self.maxIterations == None: #falls maxIterations == None ist und trotzdem in den Modus 1 kommt, so existiert keine Lösung
            print("es existiert keine Reiseroute mit ihren Eingaben")
        else:
            print("IMPORTANT ERROR WITH self.mode")

    def TomorrowAtDestination(self,lastHotel):
        if lastHotel == None: #Falls lastHotel == None ist, so wird lastHotelDistance auf 0 gesetzt, dies wird dazu gebraucht, da es an Tag1 nach kein letztesHotel gibt
            lastHotelDistance=0
        else:
            lastHotelDistance = lastHotel.distance

        if lastHotelDistance + 360 >= self.entireDistance:
            return True
        else:
            return False


    def Find_BestRatingInList(self,RatedHotelList):
        PathList_ratings = []
        for i in RatedHotelList: #erstellt die Liste PathList_ratings in dem es nur die averageRatings einfügt
            PathList_ratings.append(i.averageRating)
        bestPath_rating = Find_Max(PathList_ratings) #gibt einmal das höchste Rating wieder und zweitens den Index davon wieder von der Liste RatedHotelList [HighestRating, IndexOfHighest Rating]
        return RatedHotelList[bestPath_rating[1]] #gibt den besten Path zurück

    def adjust_maxIterations(self):
        x = len(BestPathRating.getList(self,None))
        print("Only a approximation")
        if self.maxIterations > x :
            return x**2
        return self.maxIterations**2


def File_Transform(filename):
    f = open(filename, "r").readlines()
    lines = []
    for line in f:
        line = line.strip() #entfernt das "\n" hinter jeder Zeile
        lineList = line.split(" ") #erstellt eine Liste in dem es Line bei jedem " " aufteilt. Anschliessend wird die Liste zu lineList hinzu hinzugefügt
        for Num in range(0,len(lineList)):
            lineList[Num] = float(lineList[Num]) #Wandelt jeden String in einen Float um
        lines.append(lineList) # Fügt lineList zu lines hinzu

    entireDistance = lines[1][0] # in der Zweiten Zeile steht die gesamte Fahrzeit; Fahrzeit = Distanz

    HotelsList = []
    for i in range(2,len(lines)): #die ersten beiden Zeilen werden übersprungen, da in denen nur die Anzahl der Hotels und die GesamtFahrzeit steht
        HotelsList.append(Hotel(lines[i][0],lines[i][1],i))
    return HotelsList, entireDistance

def print_results(result,filename):
    resultHotels = result.HotelsList
    print(3*"\n")
    print(f"Für die Reise von \"{filename.upper()}\" sind folgende Hotels am besten geeignet, mit einer Durchschnittsbewertung von {round(result.averageRating,2)}/5:")
    for i in resultHotels:
        print(f"Am {resultHotels.index(i)+1}.Tag: Das {i.index}.Hotel, Distanz: {int(i.distance)}, Bewertung {i.rating}/5")
    print(3*"\n")




if __name__ == '__main__':
    filenames = ["hotels1.txt","hotels2.txt","hotels3.txt","hotels4.txt","hotels5.txt"]
    for i in filenames:
        hotelsList,entireDistance = File_Transform(i)
        print_results(BestPathRating(hotelsList,entireDistance,10).bestRatedPath,i)
