def Find_Min(list):
    minIndex=0               #Definiert die Variable x als int oder float
    minNum = list[minIndex]    #Die Variable minNum,bekommt als Startwert die erste Zahl aus einer gegebenen Liste, diese wird dann immer durch die nächst kleiner Zahl ersetz, solange bis keine kleinere Zahl mehr in der Liste ist.

    for i in range(0,len(list)): #iteriert durch alle Indexe der gegebenen Liste durch
        if (list[i] < minNum):   #Vergleicht ob das Element mit dem Index [i] aus einer gegebenen Liste kliner ist als die bislang kleinste gefundene Zahl
            minNum = list[i]    #Falls eine kleinere Zahl als minNum gefunden wird, so wird minNum durch diese Zahl ersetzt
            minIndex = i        #ebenfalls wird der Index dieser neuen kliensten Zahl aktualisiert

    return [minNum, minIndex]   #gibt eine Liste zurück mit der Form [minNum, minIndex]

#------------------------------------------------------------------------------------------------

class Person:
    def __init__(self, name, verfuegbarkeitsliste):
        self.name = name
        self.verfuegbarkeitsliste = verfuegbarkeitsliste

    def find_avaliableDate(self, appointment): #Diese Funktion soll den besten Ersatztag finden. Das Attribut des gegebenen Tag muss mitgegeben werden, dies ist der Index welcher Termin beriets gegeben ist
        appointmentList = []
        appointmentList.extend(tuple(self.verfuegbarkeitsliste))
        appointmentList[appointment] = 3 #In dem den Termin zu 3 setzt wird er nicht weiter gewertet.
        if self.verfuegbarkeitsliste[appointment] == 0: #Überprüft ob nicht bereits ein Perfekter Termin Vorliegt
            return [self.name, appointment,None] #falls kein Besserer Tag verfügbar ist, so wird None zurückgegeben
        elif appointmentList.count(0)>=1: #Überprüft ob mindestens 1 perfekter Tag in der Liste ist
            replaceDate = appointmentList.index(0)
            return [self.name, appointment,replaceDate]#gibt eine Liste zurück mit der Struktur [Name der Person, gegebener Termin, Ersatztermin]
        elif appointmentList.count(1)>=1 and self.verfuegbarkeitsliste[appointment] != 1:
            replaceDate = appointmentList.index(1)
            return [self.name, appointment,replaceDate]#gibt eine Liste zurück mit der Struktur [Name der Person, gegebener Termin, Ersatztermin]
        else:
            return [self.name, appointment,None] #falls kein Besserer Tag verfügbar ist, so wird None zurückgegeben


def find_bestAppointment(amountAppoinments, personsList):#Diese Funktion soll den Termin mit den besten Verfügbarkeiten finden
    SumsOfAppointments = [] #in diese Liste soll die Summe aller Verfügbarkeiten der Personen an jedem Termin gespeichert werden
    for a in range(0,amountAppoinments):
        SumsOfAppointments.append(0)
        for p in personsList:
            x = p.verfuegbarkeitsliste[a] #gibt entspricht dem Wert der Verfügbarkeit von Person p am Termin a
            SumsOfAppointments[a] += x #Der Wert x wird zu SumsOfAppointments zum Termin a hinzu addiert
    return Find_Min(SumsOfAppointments)[1] #gibt den Termin mit den besten Verfügbarkeiten wieder

def get_bestAppointment(bestAppointment,personsList): #soll alle Verfügbarkeiten von bestAppointment wiedergeben
    bestAppointmentList = []
    for p in personsList:
        bestAppointmentList.append(p.verfuegbarkeitsliste[bestAppointment])
    return bestAppointmentList

def get_perfectAppointment(bestAppointment, personsList):
    perfectAppointList = []
    print(f"Der ideale Termin wäre der {bestAppointment}.Termin \nDazu müsste man jedoch folgende Termine umverlegen:")
    for p in personsList: #iteriert durch alle Personen.
        c = p.find_avaliableDate(bestAppointment)#die Variable c ist eine Zwischenvariable in der alle Werte aus der Methode Person.find_avaliableDate() zwischengespeichert werden
        if c[2] == None:
            perfectAppointList.append(p.verfuegbarkeitsliste[bestAppointment])
        else:
            print(f"\tDie {c[0]}. Person müsste sein {c[2]}.Termin mit dem {bestAppointment}.Termin tauschen")
            perfectAppointList.append(p.verfuegbarkeitsliste[c[2]])
    print(f"\nWenn man alle Änderungen vornimmt, dann sehen die Verfügbarkeiten des {bestAppointment}.Termins wie folgt aus:")
    for i in range(0, len(perfectAppointList)):
        a = perfectAppointList[i]
        if a == 0:
            print(f"\tFür die {i}.Person passt der {bestAppointment}.Termin sehr gut.")
        elif a == 1:
            print(f"\tFür die {i}.Person passt der {bestAppointment}.Termin mäßig.")
        elif a == 2:
            print(f"\tFür die {i}.Person passt der {bestAppointment}.Termin überhaupt nicht.")

#----------------------------------------------------------------------------------------------------

def transform_file(filename):
    f = open(filename).readlines()
    amountAppoinments = int(f[0].strip().split()[1]) #Speichert die Anzahl an m Termine in der variable amountAppoinments
    f.pop(0) #die erste Linie soll ignoriert werden, deswegen entfernt man sie
    personsList = []
    for i in range(0,len(f)): #iteriert durch alle Zeilen von 1 bis zum Ende durch
        line = f[i]
        line = line.strip()
        datesList = line.split() #erstellt eine Liste mit den Verfügbarkeits Informationen
        for j in range(0,len(datesList)): #diese Schleife soll alle Elemente der Liste dateslist von einem str in ein int umwandeln
            datesList[j] = int(datesList[j])
        personsList.append(Person(i, datesList)) #Fügt ein Objekt des Types Person zur Liste hinzu mit den benötigten Attributen
    return amountAppoinments,personsList

if __name__ == '__main__':
    filenames = ["praeferenzen0.txt","praeferenzen1.txt","praeferenzen2.txt","praeferenzen3.txt","praeferenzen4.txt","praeferenzen5.txt"]
    for fname in filenames:
        print("\n",21*"-",f"\"{fname.upper()}\"",21*"-", "\n")
        amountAppoinments,personsList = transform_file(fname)
        bestAppointment = find_bestAppointment(amountAppoinments, personsList)
        get_perfectAppointment(bestAppointment, personsList)
