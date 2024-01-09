class VerticalCar:
    def __init__(self,parkID,parkRange, ListOf_HorizontalCars):#die parkRange ist eine Liste mit dem kleinsten ord Wert und dem grössten der VerticalCars und die ListHorizontalCars ist eine Liste mit jeweils einer Liste welche die Informationen zu den Horzontal Cars het mit dem Format [ID,Postition]
        self.parkID = parkID #die parkID ist der Standplatz
        self.carID = parkID + 65
        self.parkRange = parkRange
        self.horizontalCars = VerticalCar.createHorizontalCars(self, ListOf_HorizontalCars)
        self.counterList = [0,0] #[0] steht für right und [1] für left
        self.controlBool = [True,True]#[0] steht für right und [1] für left, diese Variable wird auf False gesetzt sobald eine Bewegung nicht mehr mölich ist
        self.notFreeDrive = VerticalCar.checkExit(self)[0] #diese Variable gibt an ob das Auto am Anfang freie fagrt hatte
        VerticalCar.controller(self)

    def resetCars(self):
        for car in self.horizontalCars:
            car.position = car.startPosition

    def checkExit(self):
        for car in self.horizontalCars:
            if car.position[0] == self.parkID or car.position[1] == self.parkID:
                return [True,car] # "True" besagt, dass die Ausfahrt versperrt ist und das zweite Element gibt zurück von was
        return [False,None] #"False" besagt, dass die Ausfahrt nicht verspert ist, und gibt ebenfalls None zurück, das kein Auto die Ausfahrt versperrt


    def controller(self): #Diese Funktion soll später alle anderen Ansteuern
        if self.notFreeDrive and VerticalCar.checkExit(self)[1].checkBorder(-1): #Falls keine Freifahrt ist, wird versucht, das Auto nach links zu schieben
            for _ in range(2):
                VerticalCar.moveScript(self,VerticalCar.checkExit(self)[1],-1)
                if self.controlBool == False: #Falls controlBool False ist, so wird die schleife beendet und der Counter zurückgesetzt
                    self.counterList[0] = -1 #
                    break
                elif VerticalCar.checkExit(self)[0] == False:
                    break
            if VerticalCar.checkExit(self)[0] == True:
                self.counterList[0] = -1

        elif self.notFreeDrive and VerticalCar.checkExit(self)[1].checkBorder(-1) == False:
            self.counterList[0] = -1

        VerticalCar.resetCars(self)
        if self.notFreeDrive and VerticalCar.checkExit(self)[1].checkBorder(1): #Falls keine Freifahrt ist, wird versucht, das Auto nach rechts zu schieben
            for _ in range(2):
                VerticalCar.moveScript(self,VerticalCar.checkExit(self)[1],1)
                if self.controlBool == False: #Falls controlBool False ist, so wird die schleife beendet und der Counter zurückgesetzt
                    self.counterList[1] = -1 #
                    break
                elif VerticalCar.checkExit(self)[0] == False:
                    break
            if VerticalCar.checkExit(self)[0] == True:
                self.counterList[1] = -1
        elif self.notFreeDrive and VerticalCar.checkExit(self)[1].checkBorder(1) == False:
            self.counterList[1] = -1

    def moveScript(self,car,movementDircection):
        if VerticalCar.checkCollision(self,car,movementDircection) and car.checkBorder(movementDircection):
            car.move(movementDircection) #Beweget das "car"
            if movementDircection == -1: #Dieses If-Statment soll alle Schritte zählen
                self.counterList[0] +=1
            elif movementDircection == 1:
                self.counterList[1] +=1
            return True
        else:
            if movementDircection == -1: #Dieses If-Statment soll alle Schritte zählen
                self.controlBool[0] = False
            elif movementDircection == 1:
                self.controlBool[1] == False
            return False


    def checkCollision(self,car,movementDircection): #Überprüft, ob bei dem nächsten Schritt von Car, das Car gegen ein anderes Auto renn und verschiebt das betroffene Auto wenn möglich
        for i in self.horizontalCars:
            if movementDircection == -1 and (car.position[0]-1) in i.position: #überprüft ob car in das Auto "i" gestoßen ist.
                if i.checkBorder(-1): #überprüft ob das Auto im nächsten Schritt gegen die Begrenzung fährt. Bedeutet False: würde gegen die Begrenzung fahren; True: Würde nicht gegen die Begrenzung fahren
                    return VerticalCar.moveScript(self,i,-1) #Falls beides stimmt, so gebe Folgende Methode zurück "Rekursive Funktion"
                else:
                    return False #Gibt False zurück falls das Auto gegen eine Wand fährt
            elif movementDircection == 1 and (car.position[1]+1) in i.position: #genau das gleiche wie vorhin nur für die andere Richtung
                if i.checkBorder(1):
                    return VerticalCar.moveScript(self,i,1)
                else:
                    return False #Gibt False zurück falls das Auto gegen eine Wand fährt
        return True #Gibt zurück, dass keine Kollision stattgefundenhat.

    def createHorizontalCars(self, ListOf_HorizontalCars):
        horizontalCarsList = []
        for car in ListOf_HorizontalCars:
            horizontalCarsList.append(HorizontalCar(car[0],car[1],self.parkRange))
        return horizontalCarsList


    def getInformation(self): #diese Methode dient nur als Hilfe um die Resultate zu printen, ist für die Klasse unwichtig
        if self.counterList[0] ==-1 and self.counterList[1] == -1: #Falls keine gültige Richtung gefunden wird
            return [None,None] #dann wird nur eine Liste mit None Werten zurück gegeben.
        elif self.notFreeDrive == False: #überprüft ob das Auto freie Fahr hatte
            return [None,0] #falls dies zutrifft, so wird eine Liste mit einem None-Wert und einem 0 Wert zurückgegeben
        elif self.counterList[0] == -1:#überprüft,ob counterList[0] ungültig ist
            return [1,1] #Falls dies stimmt so gebe eine Liste zurück mit dem Format [movementIndex, movementDircection], movementIndex=> 0 is movementDircection left and 1 is movementDircection right
        elif self.counterList[1] == -1:#überprüft,ob counterList[1] ungültig ist
            return [0,-1]
        elif self.counterList[0] <= self.counterList[1]: #Überprüft ob in die linke Richtung weniger oder gleich viele Bewegungen gemacht wurden
            return [0,-1] #auch bei gleichstand wird trotzdem links zurückgegeben
        elif self.counterList[0] > self.counterList[1]:
            return [1,1]

class HorizontalCar:
    def __init__(self,carID,PostitionA,parkRange): #die Car ID ist der (Ord Wert - 65) seines Buchstabens
        self.carID = carID
        self.startPosition = [PostitionA, PostitionA+1] #[erste Koordinate, zweite Koordinate]
        self.movementCounter = [0,0] #Die linke Spalte ist für die Bewegungen nach links und die rechte für die Bewegungen nach rechts
        self.position = []
        self.position.extend(tuple(self.startPosition))
        self.parkRange = parkRange

    def move(self,movementDircection): #Diese Funktion, soll das Auto bewegen und die Bewegungen in jede Richtung zählen
        self.position[0] += movementDircection
        self.position[1] += movementDircection
        if movementDircection == -1:
            self.movementCounter[0] += 1
        elif movementDircection == 1:
            self.movementCounter[1] += 1

    def checkBorder(self,movementDircection): #Diese Funktion überprüft ob der NÄCHSTE Schritt noch in der Deffinitionsmenge ist, diese lautet [0,self.parkRange]
        if self.position[0] <= 0 and movementDircection == -1: #Falls jetzt noch eine Bewegung ausgeführt werden sollte, so rennt das Auto in die Wand
            return False
        elif self.position[1] >= self.parkRange and movementDircection == 1:
            return False
        else:
            return True #Gibt True zurück falls der nächste Schritt möglich ist und False falls nicht

#-------------------------------------------------------------------
#Transform File & intialisate Objects
def lines_transform(filename):
    f = open(filename,"r").readlines()
    lines = []
    for line in f:
        line = line.strip()
        line = line.split()
        lines.append(line)
    return lines

def file_transform(filename):
    lines = lines_transform(filename)
    parkRange = ord(lines[0][1]) - 65 #ord von "A" ist immer 65
    ListOf_HorizontalCars = []
    for index in range(2,2+int(lines[1][0])):
        ListOf_HorizontalCars.append([ord(lines[index][0]),int((lines[index][1]))])
    return parkRange, ListOf_HorizontalCars

def initalClasses(parkRange, ListOf_HorizontalCars):
    CarsList = []
    for i in range(0, parkRange+1):#die letzte Zahl wird nicht mit gezählt deswegen +1
        CarsList.append(VerticalCar(i,parkRange,ListOf_HorizontalCars))
    return CarsList

def printCarInstructions(CarsList):
    for car in CarsList:
        movementList = car.getInformation()
        if movementList == [None,None]:
            print(f"{chr(car.carID)}: not possible")
        elif movementList == [None,0]:
            print(f"{chr(car.carID)}:")
        elif movementList == [0,-1]:
            output = f"{chr(car.carID)}: " #auf den Output, sollen später die Anweisungen aufaddiert werden
            for i in range(0,len(car.horizontalCars)): #jeder index soll um eins auf addiert werden
                if car.horizontalCars[i].movementCounter[0] > 0:
                    output += f"{chr(car.horizontalCars[i].carID)} {car.horizontalCars[i].movementCounter[0]} links,\t" #die Anweisungen werden entgegen der Zählrichtung drauf addiert,dadurch wird automatisch die richtige Reihenfolge für Fahrzeuge die in die linke Richtung fahren erzeugt
            print(output.strip(",\t")) #der String endet mit ,\t; dies wird aus estetischen Gründen entfernt
        elif movementList == [1,1]:
            output = f"{chr(car.carID)}: " #auf den Output, sollen später die Anweisungen aufaddiert werden
            for i in range(1,len(car.horizontalCars)+1): #index wird normal wiedergegeben
                if car.horizontalCars[-i].movementCounter[1] > 0:
                    output += f"{chr(car.horizontalCars[-i].carID)} {car.horizontalCars[-i].movementCounter[1]} rechts,\t" #die Anweisungen werden in Richtung der Zählrichtung drauf addiert, dadurch wird automatisch die richtige Reihenfolge für Fahrzeuge die in die rechte Richtung fahren erzeugt
            print(output.strip(",\t")) #der String endet mit ,\t; dies wird aus estetischen Gründen entfernt


if __name__ == '__main__':
    filename = ["parkplatz0.txt","parkplatz1.txt","parkplatz2.txt","parkplatz3.txt","parkplatz4.txt"]
    for fname in filename:
        print("\n",10*"#",fname.upper(),10*"#","\n")
        parkRange, ListOf_HorizontalCars = file_transform(fname)
        CarsList = initalClasses(parkRange,ListOf_HorizontalCars)
        printCarInstructions(CarsList)
