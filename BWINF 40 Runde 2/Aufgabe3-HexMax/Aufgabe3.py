import numpy as np
import copy

hexList = [[np.matrix("1 1 1 1 1 1 0"),"0"],[np.matrix("0 1 1 0 0 0 0"),"1"], [np.matrix("1 1 0 1 1 0 1"),"2"], [np.matrix("1 1 1 1 0 0 1"),"3"], [np.matrix("0 1 1 0 0 1 1"),"4"],
[np.matrix("1 0 1 1 0 1 1"),"5"], [np.matrix("0 0 1 1 1 1 1"),"6"], [np.matrix("1 1 1 0 0 0 0"),"7"], [np.matrix("1 1 1 1 1 1 1"),"8"], [np.matrix("1 1 1 1 0 1 1"),"9"],
[np.matrix("1 1 1 0 1 1 1"),"A"], [np.matrix("0 0 1 1 1 1 1"),"B"], [np.matrix("1 0 0 1 1 1 0"),"C"], [np.matrix("0 1 1 1 1 0 1"),"D"], [np.matrix("1 0 0 1 1 1 1"),"E"], [np.matrix("1 0 0 0 1 1 1"),"F"]
]

available_nums = [[-1,-2,-3,-4,-5],[1,2,3,4,5]]

def get_compensations(balance_num,available_nums = available_nums):
    #Berechne die einfachen Kombinationen
    balance_num = balance_num * (-1) #Wandelt aus einer positiven Zahl eine neagtiv und umgedreht Grundlage=> balance_num + get_compensations(available_nums,balance_num) = 0 <=> get_compensations(available_nums,balance_num) = - balance_num
    combination_list = []
    combination_list.extend(get_combinations(available_nums[1],balance_num))
    #Berechne max_value
    max_opposite = abs(balance_num) -2
    if max_opposite < 0:
        max_opposite = 0
    #Berechne die komplexen Kombinationen
    for i in range(1,max_opposite+1):
        opposite_combinations= get_combinations(available_nums[1],i*(abs(balance_num)/balance_num)*(-1))
        regular_combinations= get_combinations(available_nums[1],balance_num+i*(abs(balance_num)/balance_num)) #abs(balance_num)/balance_num -> falls balance_num negaitv ist so kommt -1 raus sonst 1
        for combi in opposite_combinations:
            x = combination_filter(copy.deepcopy(combi),copy.deepcopy(regular_combinations))
            for item in x:
                item.extend(combi)
                combination_list.append(item)
    return combination_list
def combination_filter(reference_combi,combinations):
    sum = 0 #setzt den Zähler auf 0
    for i in reference_combi: #iteriert durch jede Zahl von reference_combi
        i = i * (-1) #Da reference_combi das entgegen gesetzt Vorzeichen zu den combinations hat, wir das Vorzeichen angepasst um besser vergleichen zu könnnen
        sum = sum + i
    indexes_to_remove = []
    for i,combi in enumerate(combinations): #iteriert durch alle möglichen Kombinationen ->combi stellt eine Kobination aus der vorherigen liste regular_combinations dar
        for element in reference_combi: #iteriert durch alle elemente der reference_combi
            element = element * (-1) #In diesem Schritt wird das Vorzeichen von Element getauscht
            if element in combi or sum in combi or abs(combi.count(abs(element)/element)) >= abs(element): #Die if-Abfrage, überprüft 3 Sachen: 1.Ist element(Vorzeichen geändert) bereits in combi, 2. Ist die Summe(Vorzeichen angepasst) des opposite_combination in combi(-> Falls die zutrifft, so hätte man in der Kompination ein eigenständiges paar, was verhindert werden soll), 3. ist die summe der 1sen in combi größer oder gleich element(dies würde wieder zu einem eigenständigem paar führen)
                indexes_to_remove.append(i) #Falls eine der drei bedingungen Zutrifft, so wird der index dieser Combi in eine Liste eingefügt und später entfernt

    indexes_to_remove = list(set(indexes_to_remove)) #Falls mehrere male der gleiche Index vorhanden ist, so werden alle außer einer gelöscht
    indexes_to_remove.sort(reverse=True) #sortiert die Indexe von groß nach klein, so dass sich die Indexe beim löschen nicht verschieben
    for i in indexes_to_remove:
        combinations.pop(i)
    return combinations
def get_combinations(list_num,goal): #soll alle kombinationsmöglichkeiten zurück geben, welche alle die gleichen
    target = abs(goal)
    combinations = []
    origin_combination = [] #Auf dieser Kombination baut die ganze Funktion auf
    for _ in range(int(abs(target))): #Definiert die Ausgangskombination,welche ebenfalls eine gültige kombination ist
        origin_combination.append(1)
    combinations.append(origin_combination)
    memory = [origin_combination] #In dieser Variable, werden die neuen Kombinationen zwischen gespeichert um überprüft zu werden. Die die bereits überprüft worden sind werden wieder entfernt
    while len(memory)>0:#Die while-Schleife soll solange laufen, bis der "Tree" fertig ausgebaut wurde und in memory keine noch zu überprüfenden Wege mehr da sind
        for i in list_num[1:]: #iteriert durch alle zur Verfügung stehenden Zahlen außer 1
            if memory[0] == []:#Falls memory leer sein sollt,so wird diese schleife abgebrochen
                break
            if memory[0].count(1) >= i: #Falls die Anzahl an 1sen ausreicht um i zuerstellen,dann
                new_combination = memory[0] #speichere in new_combination den Inhalt von memory[0]
                new_combination = new_combination[:(i*(-1))] #Entferne i 1sen
                new_combination.append(i) #ersetze sie durch 1
                new_combination.sort(reverse=True) #Nun wird noch mal alles von Groß nach klein sortiert
                if new_combination not in combinations: #Falls es die Kombination nicht bereits geben sollte, dann
                    memory.append(new_combination) #füge sie zu memory hinzu um sie überprüfen zu lassen
                    combinations.append(new_combination) #füge die Kombination hinzu
        memory.pop(0) #Anschließend wir die gerade überprüfte Kombination wieder gelöscht
    for i,list in enumerate(combinations): #Sortiert wieder alles von klein nach groß
        list.sort()
        combinations[i] = list
    if float(abs(goal)/goal) == 1: #überprüft op goal postiv ist Ansatz: abs(goal)/goal = 1 -> bedeutet postiv;   abs(goal)/goal = -1 -> bedeutet negativ
        return combinations
    else: #Falls die Zahl negativ sein sollte, so wird jedes element mit -1 multipliziert
        for i,combi in enumerate(combinations):
            for j,item in enumerate(combi):
                combinations[i][j] = item * (-1)
        return combinations
def Find_Max(list):#input: [[index,value]]
    maxIndex=0               #Definiert die Variable x als int oder float
    maxNum = list[maxIndex][1]    #Die Variable maxNum,bekommt als Startwert die erste Zahl aus einer gegebenen Liste, diese wird dann immer durch die nächst grössere Zahl ersetzt, solange bis keine grössere Zahl mehr in der Liste ist.
    for i,b in enumerate(list): #iteriert durch alle Elemente der gegebenen Liste durch
        if maxNum is None:#Falls die aktull größte Zahl null sein sollte, so wird sie ersetzt
            maxNum = b[1]    #Falls eine grössere Zahl als maxNum gefunden wird, so wird maxNum durch diese Zahl ersetzt
            maxIndex = i       #ebenfalls wird der Index dieser neuen grössten Zahl aktualisiert
            continue
        elif b[1] is None: #Falls b[1] None sein sollte so wird sie übersprungen
            continue
        elif (b[1] > maxNum):   #Vergleicht ob das Element grösser ist als die bislang grösste gefundene Zahl
            maxNum = b[1]    #Falls eine grössere Zahl als maxNum gefunden wird, so wird maxNum durch diese Zahl ersetzt
            maxIndex = i        #ebenfalls wird der Index dieser neuen grössten Zahl aktualisiert
    return maxIndex   #gibt maxIndex zurück
def SortIndexes_MinToMax(list): #input: [[index,value][...]]#Erstellt eine neue Liste mit den Indexen der Ausgangsliste sortiert von dem kleinsten Wert bei [0] und dem größtem am Ende
    index = Find_Max(list)
    maxValue = list[index][1]
    sortedIndexs=[list[index][0]] #setzt einen ersten ReferenzWert ein um fest
    sortedValues = [list[index][1]]
    sorted_list = [list[index]]
    list.pop(index)
    none_indexes = []
    none_values = []
    none_list = []
    for item in list: #Da der Refernz Wert den Index 0 hatte so wird dieser hier übersprungen
        if item[1] is None:
            none_indexes.append(item[0])
            none_values.append(item[1])
            none_list.append(item)
            continue
        elif item[1] == maxValue:
            sortedIndexs.append(item[0])
            sortedValues.append(item[1])
            sorted_list.append(item)
        for i,value in enumerate(sortedValues):
            if item[1] < value: #Falls der Wert kleiner ist alls value, so wird die Zahl vor Value eingefügt
                sortedIndexs.insert(i,item[0])
                sortedValues.insert(i,item[1])
                sorted_list.insert(i,item)
                break
    sortedIndexs.extend(none_indexes)
    sortedValues.extend(none_values)
    sorted_list.extend(none_list)
    return sorted_list



#----------------------------------------------------------------------------

class Num_Max():
    def __init__(self,num,changes,meta_list):
        self.num = num
        self.changes = changes
        self.meta_list = meta_list
        self.adjusted_meta_list = copy.deepcopy(self.meta_list)
        self.maximum_num = Num_Max.main(self)

    def update_meta_data(self,set_with_position):#Updatet die Werte mit dem neuen Set, ein set_with_position ist eine Liste mit der Form [sum,changes,Position]
        position = set_with_position[2]
        meta_data = self.adjusted_meta_list[position]
        for i,meta_num in enumerate(meta_data):
            self.adjusted_meta_list[position][i][2][0] = meta_num[2][0] - set_with_position[0]
            self.adjusted_meta_list[position][i][2][1] = meta_num[2][1] - set_with_position[1]

    def main(self):
        max_num = [] #In diese Variable sollen die neuen Maximalen Zahlen gespechert werden
        for i in range(len(meta_list)): #ruft den Code so oft auf, wie es Positionen gibt. Da die Funktion Num_Max.transform(self) die Länge von der adjusted_meta_list immer um 1 verringert, wird diese am Ende eine leere Liste sein
            print(f"{round((i+1)/len(meta_list)*100)}%\t {i+1}/{len(meta_list)}")
            max_num.append(Num_Max.transform(self)[0])
        return max_num

    def transform(self): #In diese Funktion kommt das Auswahlverfahren, für adjusted_meta_list[0]
        for num in list(reversed(self.adjusted_meta_list[0])):
            if num[2][1] == 0: #Diese IF-Abfrage,soll verhindern, dass die Zahl kleiner wird
                self.adjusted_meta_list.pop(0)
                return num
            elif self.changes < num[2][1]: #überprüft ob man mehr Umlegungen benötigt als zur Verfügung stehen um die Zahl zu transformieren
                continue
            self.changes = self.changes -num[2][1]
            is_possible,valid_combination = Num_Max.combination_filter(self,num[2][0])

            if is_possible and valid_combination == None:#überprüft, ob der Fall zutriff, in welchem man keine anderen Wert updaten muss, Grund: wenn num[2][0] == 0 ist, so ist diese Zahl bereits ausgeglichen
                self.adjusted_meta_list.pop(0)
                return num
            elif is_possible:
                self.changes = self.changes - valid_combination[1]
                for set_with_position in valid_combination[0]:
                    Num_Max.update_meta_data(self,set_with_position)
                self.adjusted_meta_list.pop(0)
                return num
            else:
                self.changes = self.changes + num[2][1] #Falls die Umformung nicht möglich ist, so addiere die verbrauchten Umlegungen(welche wir vorher abgezogen hatte) wieder drauf, da wir sie nicht benötigt haben

        return True #Falls, ein passendes Set gefunden wurde, dann wird True zurückgegebene


    def get_best_positions(self,values,combination): #values is a list
        values = copy.deepcopy(values)
        combination = copy.deepcopy(combination)
        delta_meta_list = []
        for pos in copy.deepcopy(self.adjusted_meta_list[1:]): #Die Code erstellt eine Kopie von adjusted_meta_list ab dem index[1], da index[0] die Position ist die wir gerade versuchen zu optimieren. Es werden zum einen nur die benötigten werte kompiert und zum anderen wird aus der for [hex_Num,dez_Num,[sum,changes]] -> [sum,changes]
            pos_list = []
            for item in pos:
                item = item[2]
                for val in values:
                    if val == item[0]:
                        pos_list.append(item)
                        break
            if pos_list != []:
                delta_meta_list.append(pos_list)


        for i,pos in enumerate(delta_meta_list): #sortiert die Liste so, dass für die jeweilige position immer bei pos[0] das Element mit dem niedrigsten change steht, eben falls, falls ein Value mehrmals vorhanden sein sollte, so bleibt nur der mit dem niedrigsten changes erhalten
            vals = copy.deepcopy(values) #erstelle eine Kopie der Values
            pos = SortIndexes_MinToMax(pos)
            for j,item in enumerate(pos): #iteriert durch die neue sortierte Liste von vorne nach hinten
                if item[0] in vals: #überprüft ob es den Value bereits vorher und damit mit niedrigerem Change gibt,falls dies zutriff,so wird das item entfernt
                    vals.remove(item[0])
                else:
                    pos[j] = []
            while [] in pos:
                pos.remove([])
            delta_meta_list[i] = pos

        list_possible_values = []
        for v in values:#erstellt die Strucktur für die Liste list_possible_values
            list_possible_values.append([])

        for i,pos in enumerate(delta_meta_list): #Verwandelt die Kopie von adjusted_meta_list welche nach den Positionen strukturier ist, in eine neue liste,welche nach den verschiedenen Values struktueriert ist
            for item in pos:
                item.append(i)
                for j,v in enumerate(values):
                    if item[0] == v:
                        list_possible_values[j].append(item)
        for i,list in enumerate(list_possible_values):# sortiert die Liste so, dass für list_possible_values[i][0] gilt, dass es der values[i] mit dem insgesamt niedrigstem chang ist
            if list == []: #überprüft, ob die Liste nicht leer ist. Falls sie leer sein sollte, dann bedeutet das, dass der Benötigte wert in keiner der Postitionen zu finden ist
                return [None,None] #gibt eine Liste mit [None,None] zurück.
            list_possible_values[i] = SortIndexes_MinToMax(list)
        new_list = []
        for i,item in enumerate(list_possible_values): #Erstellt eine neue Liste von Listen mit der Form [values[i],len(list[i]),list[i]] -> Diese neue Liste wird erstellt, damit man die Funktion SortIndexes_MinToMax() benutzen kann
            new_list.append([values[i],len(item),item])
        if new_list == []:
            return [None,None] #gibt eine Liste mit [None,None] zurück.
        list_possible_values = SortIndexes_MinToMax(new_list) #Die neu optimierte Liste sieht wie folgt aus, [[value,lenght,list],...] -> sortiert nach der länge, damit wir autmatisch mit dem Value anfangen, welcher in den wenigsten Positionen vorkommt

        used_positions = [] #Hier sollen, die namen der bereits benutzten Position stehen, damit eine Position nicht mehrmals verwendet wird
        result = []
        sum_changes = 0

        for item in list_possible_values: #iteriert durch die neu struktuerierte
            for _ in range(combination.count(item[0])): #Zählt wie oft der ausgewählt wert in der Variable combinations steht und lässt den folgenden Code so oft ausführen
                control_bool = False #Diese Variable soll später helfen zu überprüfen ob eine lösung gefunden wurde
                for data in item[2]:
                    if data[2] not in used_positions: #Kontrolliert zum einen ob die Position von data nicht bereits verwendet wurde
                        used_positions.append(data[2]) #Füge die gerade benutzte Position zu der Variable used_positions
                        result.append(data)
                        sum_changes += data[1] #summiert die anzahl der Changes auf
                        control_bool = True
                        break
                if control_bool == False: #Falls kein passender data_set in item gefunden wurde, so gibt None zurück, was bedeutet, dass es keine Lösung gibt
                    return [None,None] #gibt eine Liste mit [None,None] zurück. Grund: Da es weder ein Valides Result noch eine Summe von Changes
            return [result,sum_changes] #gibt das Resultat sowie die Summe der benötigten änderungen


        return values, list_possible_values

    def combination_filter(self,balance_num): #Diese Methode soll die unnötigen Kombinationen raus werfen und die Kombinationen der länge nach sortieren
        if balance_num == 0:
            return True,None
        combination_list = get_compensations(balance_num)
        valid_combinations = []
        for combi in combination_list:
            combination = Num_Max.get_best_positions(self,list(set(combi)),combi)
            valid_combinations.append(combination)
        valid_combinations = SortIndexes_MinToMax(valid_combinations) #sortiert alle möglichkeiten,so dass die mit den insgesamt wenigsten Changes beim Index 0 steht
        if valid_combinations[0] == None or valid_combinations[0][1] == None:
            return False,None
        elif valid_combinations[0][1] <= self.changes:
            valid_combinations[0][0][0][2] += 1 #Der Name(Index der Position), wird um 1 erhöht, da wir vorher 0 übersprung hatten(Wir hatten 0 übersprungen, da es adjusted_meta_list[0] und wir diese Optimieren wollen)
            return True,valid_combinations[0]
        else:
            return False,None

def get_meta_nums(list,changes): #Diese Funktion soll eine Liste zurückgeben, in welcher alle Metadaten, für eine Mögliche umwandlung enthalten sind
    meta_list = [] #In diese Liste sollen später alle in jeweils eigenen Listen die Metadaten für die einzelnen Positionen stehen, die Position entspricht dem Index
    for num_vec in list: #Iteriert durch die Vektoren der gegebenen Zahlen
        meta_data = [] #Diese Variable dient als zwischen speicher
        for i,hex_vec in enumerate(hexList): #Iteriert durch alle möglichkeiten zum umwandeln
            dif_vec = np.subtract(hex_vec[0],num_vec) #subtrahiert beide Matrizen, das Resultat ist eine Matrix, in welcher eine 1 bedeutet, dass ein Stäbchen entfernt werden muss und ein -1 bedeutet, dass dort eins fehlt
            a = analyze_vector(dif_vec)
            meta_data.append([hex_vec[1],i,a]) #Die Liste hexVec wird erweitert und hat nun die Form [Hex Verion der Zahl, Dezimal Version, [Summe,Benötigte Umlegungen]]
        meta_list.append(meta_data)
    return meta_list

def analyze_vector(vec):
    abs_sum = 0#Berechnet die Summe aller Qudrate der Elemente im gegebenen Vektor, Ziel: die Berechnung aller Verschiebungen
    for i in range(0,7):
        abs_sum += abs(vec.item(i))
    abs_sum = abs_sum / 2 #Teil das Resultat durch zwei, weil sqrsum jede einzelne Aktion angibt, als aufheben und ablegen sind jeweils einzelne Aktionen, deswegen wird das Resultat durch zwei geteilt
    sum = vec.sum()
    return [sum,abs_sum] #Diese Funktion analysiert den Differenzen Vektor und gibt die bnötigten Metadaten zurück
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Output():
    def __init__(self,final_num,vec_start):
        self.num = final_num
        self.vec_max_list = transform_num(self.num) #speichert die Zahlen in Form von Vektoren in einer Liste ab
        self.max_num_matrix = Output.vstack(self,self.vec_max_list) #fügt die Vektoren zu einer großen Matrix zusammen
        self.max_num_matrix = self.max_num_matrix.getT() #passt die Ausrichtung der Matrix an
        self.vec_start = vec_start
        self.start_matrix = Output.vstack(self,self.vec_start)
        self.start_matrix = self.start_matrix.getT()
        self.delta_matrix = np.subtract(self.max_num_matrix,self.start_matrix) #subtrahiert beide Matrixen (die Start und die End Matrix), dadurch kann festgestellt werden, welche änderungen vorgenommen wurden. -1 bedeutet ein Stäbchen wurde entfernt und 1 bedeutet ein Stäbchen wurde hinzugefügt
        self.negatif,self.positif = Output.get_list(self) #erstellt die Liste mit den enötigten Informationen
        self.positif = Output.convert(self,self.positif) #Wandelt die Informationen in Wörter um
        self.negatif = Output.convert(self,self.negatif)
        self.counter = 0
        Output.print_output(self)


    def print_output(self): #Gibt die extrahierten Ergebnisse strukturiert aus
        for i in range(len(self.positif)):
            print(f"{self.counter}. Umlegung: Das Stäbchen {self.negatif[i][1]} von der {self.negatif[i][0]}.Zahl wird zur {self.positif[i][0]}.Zahl, {self.positif[i][1]} umgelegt")
            self.counter += 1

    def convert(self,list): #Diese Funktion wandelt die Zahlen in wörter um
        for i,tuple in enumerate(list):
            if tuple[1] == 0:
                list[i] = (tuple[0],"oben")
            elif tuple[1] == 1:
                list[i] = (tuple[0],"rechts oben")
            elif tuple[1] == 2:
                list[i] = (tuple[0],"rechts unten")
            elif tuple[1] == 3:
                list[i] = (tuple[0],"unten")
            elif tuple[1] == 4:
                list[i] = (tuple[0],"links unten")
            elif tuple[1] == 5:
                list[i] = (tuple[0],"links oben")
            elif tuple[1] == 6:
                list[i] = (tuple[0],"in der Mitte")
        return list

    def vstack(self,vec_list):
        matrix = np.vstack((vec_list[0],vec_list[1]))
        for vec in vec_list[2:]:
            matrix = np.vstack((matrix,vec))
        return matrix

    def get_list(self): #Diese Methode speichert alle -1sen und 1sen in eine Variable in der Form eines Tupels,welcher ihre Koordinaten enthält
        positif = []
        negatif = []
        for i in range(self.delta_matrix.shape[1]): #Die Anzahl der Zahlen wird mit self.delta_matrix.shape[1] herausgefunden
            for j in range(7):
                num = self.delta_matrix.item((j,i))
                if num == -1:
                    negatif.append((i,j))
                elif num == 1:
                    positif.append((i,j))
        return negatif,positif
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def transform_file(FileName): #Diese Funktion wandelt den Inhalt des Dokuments in die Variabeln num,changes um
    file = open(FileName, "r").readlines()
    num = file[0].strip()
    num = num.strip("")
    changes = file[-1].strip()
    changes = int(changes)
    return num,changes

def transform_num(num): #Diese Funktion wandelt die einzelnen Zahlen, in Matrizen um
    num = list(num)
    list_hex = []
    for j in num:
        for i in hexList:
            if i[1] == j:
                list_hex.append(i[0])
    return list_hex

def print_result(final_num,start_vec,file_name,num,changes):
    print(f"\n--- {file_name.upper()} ---\n")
    print(f"Die maximale Zahl, welche man mit {changes} Umlegungen und der Ausgangszahl \"{num}\" erreichen kann ist:")
    string = ""
    for i in final_num:
        string = string + i
    print(string)
    print()
    if file_name in ["hexmax0.txt","hexmax1.txt","hexmax2.txt"]:
        Output(final_num,start_vec)
    print()

filenames = ["hexmax0.txt","hexmax1.txt","hexmax2.txt","hexmax3.txt","hexmax4.txt","hexmax5.txt"] #,"hexmax0","hexmax1.txt","hexmax2.txt","hexmax3.txt","hexmax4.txt","hexmax5.txt"
for filename in filenames:
    num,changes = transform_file(filename)
    num_transform = transform_num(num)
    meta_list = get_meta_nums(num_transform,changes)
    """for i in meta_list:
        print(f"\n{i}")"""
    Max =Num_Max(num,changes,meta_list)
    print_result(Max.maximum_num,num_transform,filename,num,changes)
