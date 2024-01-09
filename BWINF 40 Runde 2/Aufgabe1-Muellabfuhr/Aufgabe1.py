import numpy as np
import copy
def Find_Max_Basic(list):
    maxIndex=0               #Definiert die Variable x als int oder float
    maxNum = list[maxIndex]    #Die Variable maxNum,bekommt als Startwert die erste Zahl aus einer gegebenen Liste, diese wird dann immer durch die nächst grössere Zahl ersetzt, solange bis keine grössere Zahl mehr in der Liste ist.

    for i in range(0,len(list)): #iteriert durch alle Indexe der gegebenen Liste durch
        if (list[i] > maxNum):   #Vergleicht ob das Element mit dem Index [i] aus einer gegebenen Liste grösser ist als die bislang grösste gefundene Zahl
            maxNum = list[i]    #Falls eine grössere Zahl als maxNum gefunden wird, so wird maxNum durch diese Zahl ersetzt
            maxIndex = i        #ebenfalls wird der Index dieser neuen grössten Zahl aktualisiert

    return [maxNum, maxIndex]   #gibt eine Liste zurück mit der Form [maxNum, maxIndex]

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
    return sortedIndexs,sorted_list

def get_short_path(startpoint,matrix):
    gone_nodes = []#Diese Variable gibt an welche knoten bereits überprüft worden sind
    waiting_nodes = [startpoint] #Diese Variable speichert alle Knoten welche noch überprüft werden müssen
    memory = [[startpoint]] #Diese Variable specihert die Wege der noch zu überprüfenden Knoten
    grads = Graph.get_grad(matrix)[0]
    while True and len(waiting_nodes) != 0:
        if grads[waiting_nodes[0]]%2 == 1 and waiting_nodes[0] != startpoint: #überprüft ob der Knoten ungerade ist
            return memory[0] #Falls die zutrifft, so wird der Weg diese Graphen zurück gegeben
        new_nodes = get_connections(waiting_nodes[0],matrix,gone_nodes) #In der Variable new_Nodes werden die nächst möglichen Knoten gespeichert(aufsegehend von waiting_nodes[0])
        gone_nodes.append(waiting_nodes[0])
        for n in new_nodes:
            if n not in waiting_nodes:
                waiting_nodes.append(n)
                path = tuple(memory[0])
                path = list(path)
                path.append(n)
                memory.append(path)
        waiting_nodes.pop(0)
        memory.pop(0)

def get_connections(point,matrix,gone_nodes): #die Parameter sind zum einen die aktuelste Verion der angepassten Matrix und zum anderen der Bereits zurückgelete weg
    row = matrix[:,point] #gibt die Spalte des Knoten auf dem wir uns gerade Befinden wieder
    row.tolist()
    possible_connections = []
    for i,value in enumerate(row):
        if value != 0  and i not in gone_nodes:
            possible_connections.append(i)
    return possible_connections
#---------------------------------------------------------------------------------------------------
class Graph():
    def __init__(self,data,nodes,edges):
        self.nodes = nodes
        self.edges = edges
        self.matrix = Graph.create_matrix(self,data)
        self.weight_matrix = Graph.create_weight_matrix(self,data)
        self.adjusted_matrix = Graph.create_adjusted_matrix(self,data)

    def create_matrix(self,data):
        zero_matrix = np.zeros([self.nodes,self.nodes]) #erstellt eine Null-Matrix mit der Form nodes x nodes
        for i in data: #iteriert durch alle Daten
            zero_matrix.itemset((i[0],i[1]),1) #bei einer Adjenzmatrix werden normalerweise die 1sen nach den richtungen der Edges gesetzt zum Beispiel A -> B aber da unser Graph ungerichtet ist müssen wir sowohl bei A->B als auch bei B->A eine 1 einfügen
            zero_matrix.itemset((i[1],i[0]),1)
        return zero_matrix

    def create_weight_matrix(self,data):
        zero_matrix = np.zeros([self.nodes,self.nodes]) #erstellt eine Null-Matrix mit der Form nodes x nodes
        for i in data: #iteriert durch alle Daten
            zero_matrix.itemset((i[0],i[1]),i[2]) #Diese Matrix ist von der Form her eine Adjenz-Matrix nur, dass statt den 1sen die Gewichte da stehen
            zero_matrix.itemset((i[1],i[0]),i[2])
        return zero_matrix

    def create_adjusted_matrix(self,data):
        adjusted_matrix = Graph.create_matrix(self,data)
        odd_index,control_bool = Graph.get_grad(adjusted_matrix)[1:]
        while control_bool:
            adjusted_matrix = Graph.adjust_grad(odd_index[0],adjusted_matrix,self.weight_matrix)
            odd_index,control_bool = Graph.get_grad(adjusted_matrix)[1:]
        return adjusted_matrix

    def adjust_grad(point,matrix,weight_matrix): #Dies Methode soll die einzelnen Elemente in der Matrix anpassen
        #path = Graph.get_short_path(point,matrix)
        odd_list = Graph.get_grad(matrix)[1]
        odd_list.remove(point)
        path = Daijkstra(point,odd_list,weight_matrix).path
        for i in range(len(path)-1):
            actual_num = matrix.item((path[i],path[i+1]))
            matrix.itemset((path[i],path[i+1]),actual_num+1)#Erhöht die Anzahl der Straßen um eins sowohl bei A->B als auch bei B->A
            matrix.itemset((path[i+1],path[i]),actual_num+1)
        return matrix

    def get_grad(matrix):#soll eine Liste mit den Grade der einzelnen Kreuzungen wiedergeben
        list = matrix.sum(axis=0).tolist() #summjert alle Zahlen von jeder Spalte auf und fügt es in ein Numpy Array, dieser Array wird anschließend in eine Liste convertiert
        odd_list = []
        control_bool = False
        for i,b in enumerate(list):
            if b%2 == 1:
                odd_list.append([i,b])
                control_bool = True
        if len(odd_list) > 0:
            odd_list = SortIndexes_MinToMax(odd_list)[0]
        return list,odd_list,control_bool #[liste mit den jeweiligen Grade, liste mit den Sortierten Indexen,Kontrolle(Falls False, so sind keine ungeraden Grade mehr da)]
#----------------------------------------------------------------------------------------------------------
class Fleury():
    def __init__(self,nodes,adjusted_matrix):
        self.nodes = nodes
        self.maxCounter = adjusted_matrix.sum()/2
        self.euler_circuit = Fleury.create_fleury_matrix(self,adjusted_matrix)

    def create_fleury_matrix(self,matrix):
        startpoint = 0
        current_matrix = copy.deepcopy(matrix)
        euler_circuit = [0] #Dies ist die Liste in welche nach her die besuchten Knoten der Reihenfolge nach eingefügt werden. 0 ist bereits gegeben, da es immer bei 0 startet
        counter = 0
        while True:
            if counter % 10 == 0:
                print(f"{round(counter/self.maxCounter *100,1)}%")
            counter +=1
            if euler_circuit[-1] == 0: #Falls wir uns auf dem Startpunkt befinden, so gibt es keine möglichen "Cut edges"
                delta_matrix = np.zeros([self.nodes,self.nodes]) #erstellt eine Null-Matrix mit der Form nodes x nodes
            else:
                delta_matrix = Fleury.get_delta_matrix(self,euler_circuit[-1],current_matrix) #speichert die delta_matrix
            adapted_matrix = np.subtract(current_matrix,delta_matrix) #Diese Matrix zeigt alle möglichen Wege, ohne dass man ein Teil der original Matrix abschneidet
            next_node = Fleury.choose_nodes(self,euler_circuit[-1],adapted_matrix) #gibt den nächsten Knoten an ohne das Risiko eine "Bridge" zu entfernen
            if next_node != None: #Überprüft ob es ein Resultat gibt
                euler_circuit.append(next_node)
            else:
                next_node = Fleury.choose_nodes(self,euler_circuit[-1],current_matrix) #Falls der einzige verbleibende Knoten über eine "Cut Edge" führt so wird diese genommen
                euler_circuit.append(next_node)
            actual_num = current_matrix.item((euler_circuit[-1],euler_circuit[-2])) #Entnimmt die Anzahl der Straßen, zwischen den beiden letzten gegangenen Knoten
            current_matrix.itemset((euler_circuit[-1],euler_circuit[-2]),actual_num-1)#Verringer die Anzahl der Straßen um eins sowohl bei A->B als auch bei B->A
            current_matrix.itemset((euler_circuit[-2],euler_circuit[-1]),actual_num-1)
            remain_edges = current_matrix.sum()/2 #Diese Funktion gibt die Anzahl der Verbleibenden Straßen an. Die Funktioniert indem man die Elemente der aktuellen Matrix aufsummiert und anschließend wird das ganz noch durch 2 geteilt, da bei ungerichteten Adjenzmatrixen, eine Ecke immer 2 mal dagestellt wird A->B, B->A
            if next_node == 0 and remain_edges == 0: #Falls wir wieder am Start
                return euler_circuit

    def get_delta_matrix(self,prev_node,matrix): #Diese Matrix soll garantieren, dass keine Bridge abgeschnitten wird
        zero_matrix = np.zeros([self.nodes,self.nodes]) #erstellt eine Null-Matrix mit der Form nodes x nodes
        path =get_short_path(prev_node,matrix)
        for i in range(len(path)-1):
            zero_matrix.itemset((path[i],path[i+1]),1)#Gibt die Edges an, welche nicht abgeschnitten werden dürfen
            zero_matrix.itemset((path[i+1],path[i]),1)
        return zero_matrix

    def choose_nodes(self, prev_node, matrix): #Diese Methode soll den nächsten Knoten finden
        possible_nodes = matrix[:,prev_node]#gibt die Spalte des Knoten auf dem wir uns gerade Befinden wieder
        for node,b in enumerate(possible_nodes):
            if b > 0:
                return node #Gibt den ersten Knoten zurück zu dem mindestens eine Ecke führt
        return None #Falls kein Knoten in frage kommt, so gib None zurück
#----------------------------------------------------------------------------------------------
class Daijkstra():
    def __init__(self,startpoint,endpoint,matrix):
        self.startpoint = startpoint
        self.endpoint = Daijkstra.get_endpoint(endpoint)
        self.weight_matrix = matrix
        self.visited = []
        self.visited_nodes = [] #Hier werden die nur die Namen aller bereits besuchten Knoten eingefügt
        self.unvisited = Daijkstra.get_unvisited_list(self)
        self.path = []
        Daijkstra.main(self)

    def get_endpoint(endpoint):#Dies dient dazu, dass man sowohl, eine Liste von möglichen Endzielen angeben kann oder nur ein einzelner Wert oder None
        result = []
        if isinstance(endpoint,list):
            result.extend(endpoint)
        else:
            result.append(endpoint)
        return result

    def main(self):
        for _ in range(len(self.unvisited)):
            current_set = self.unvisited[0]
            x = self.unvisited.pop(0)
            self.visited.append(x)
            self.visited_nodes.append(x[0])
            Daijkstra.update_unvisited(self,current_set)
            if self.endpoint[0] == None:
                pass
            elif len(self.endpoint) == 1 and current_set[0] == self.endpoint[0]: #Überprüft, ob man den Algorithmus bereits beenden kann
                self.path.extend(Daijkstra.get_path(self,self.endpoint[0]))
                break
            elif len(self.endpoint) > 1 and current_set[0] in self.endpoint:#Überprüft, ob man den Algorithmus bereits beenden kann
                self.path.extend(Daijkstra.get_path(self,self.visited[-1][0]))
                break
            if len(self.unvisited) == 1:#Falls nur noch 1 Knoten nicht besucht wurde,so ist der Algorithmus fertig
                if self.unvisited[0] == self.endpoint[0]:
                    Daijkstra.get_path(self)
                x = self.unvisited.pop(0)
                self.visited.append(x)
                self.visited_nodes.append(x[0])
                break

    def update_unvisited(self,current_set):
        neighbors = Daijkstra.get_neighbors(self,current_set[0])
        for i,set in enumerate(self.unvisited): #iteriert durch alle unbesuchten Knoten
            if set in neighbors: #Überprüft ob der Knoten auch ein Nachbar ist
                additional_weight = self.weight_matrix.item((current_set[0],set[0])) #sucht das Gewicht zwischen dem Knoten current_set und set
                possible_new_weight = current_set[1]+additional_weight
                if set[1] is None:
                    self.unvisited[i] = [set[0],possible_new_weight,current_set[0]]
                elif possible_new_weight < set[1]:
                    self.unvisited[i] = [set[0],possible_new_weight,current_set[0]]
            else:
                continue
        self.unvisited = SortIndexes_MinToMax(self.unvisited)[1]

    def get_neighbors(self,node): #Der aktuelle Knoten und die Matrix
        row = self.weight_matrix[:,node] #gibt die Spalte des Knoten auf dem wir uns gerade Befinden wieder
        row.tolist()
        neighbors = []
        for i,value in enumerate(row): #iteriert durch alle Knoten
            if value != 0  and i not in self.visited_nodes: #überprüft ob der Wert größer 0 ist und ob der Knoten nicht bereits besucht wurde
                for set in self.unvisited: #Falls beide bedingungen zutreffen, so wird in unvisited nach dem passenden set gesucht und zu neighbors angefügt
                    if set[0] == i:
                        neighbors.append(set)
        return neighbors

    def get_unvisited_list(self): #Erstell eine Liste, mit der Benötigten Form, [Index,None,None] und setzt den Startpunkt auf 0. [Startpoint,0,None]
        lenght = self.weight_matrix.shape[0] #Gibt mir die Breite der Matrix zurück
        unvisited = []
        for i in range(0,lenght):
            if i == self.startpoint:
                unvisited.append([i,0,None])
            else:
                unvisited.append([i,None,None])
        unvisited = SortIndexes_MinToMax(unvisited)[1] #Sortiert die Liste, sodass das Startelement am Anfang steht
        return unvisited

    def get_path(self,endpoint):
        path = []
        current_set = Daijkstra.search_node(self,endpoint)
        while True:
            if current_set[2] == None:
                path.append(current_set[0])
                break
            path.append(current_set[0])
            current_set = Daijkstra.search_node(self,current_set[2])
        return list(reversed(path)) #Die Liste wird umgedreht, da sie aktuell den Weg vom Endpunkt zum Startpunkt zeigt, sie soll aber den Weg vom Startpunkt zum Endpunkt zeigen

    def get_distance(self,endpoint):
        return Daijkstra.search_node(self,endpoint)[1]

    def search_node(self,node):
        for i in range(len(self.visited_nodes)):
            b = self.visited[i]
            if b[0] == node:
                return b
        print("Not found")

    def merge_adjenz_weight(adjence_matrix,weight_matrix): #
        shape = adjence_matrix.shape[0]
        new_matrix = np.zeros((shape,shape))
        for i in range(shape):
            for j in range(shape):
                actual_num1 = weight_matrix.item((i,j))#Entnimmt der Matrix die Werte an der Stelle (i,j)
                if  adjence_matrix.item((i,j)) > 0: #Falls die Adjenzmatrix größer 1 ist so wird actual_num2 automatisch auf 1 gesetzt
                    actual_num2 = 1
                else: #Falls dies nicht zutrifft,so wird actual_num2 auf 0 gesetzt
                    actual_num2 = 0
                product = actual_num1*actual_num2 #Multipliziert beide Werte
                new_matrix.itemset((i,j),product)
        return new_matrix
#--------------------------------------------------------------------------------------------------
class Split():
    def __init__(self,nodes,adjusted_matrix,weight_matrix,euler_circuit):
        self.nodes = nodes
        self.adjusted_matrix = adjusted_matrix
        self.weight_matrix = weight_matrix
        self.euler_circuit = euler_circuit
        self.total_lenght = Split.complete_lenght(self)
        self.average_lenght = self.total_lenght/5
        self.mst = Daijkstra(0,None,weight_matrix)
        self.day_paths = []
        self.daily_lenght = []
        Split.main(self)
#Therorie: Wir teilen jede Strecke in 3 teile hinfahrt,weg,rückfahrt -> es muss immer ein weg geben, jedoch eine hin und rückfahrt sind nicht zwingen notwendig
    def get_final_paths(self):
        days = []
        for i in self.day_paths:
            path = []
            if i[0] != []:
                i[0].pop(-1)
                for j in i[0]:
                    path.append(j)
            if i[2] != []:
                i[1].pop(-1)
                for j in i[1]:
                    path.append(j)
                for j in i[2]:
                    path.append(j)
            else:
                for j in i[1]:
                    path.append(j)
            days.append(path)
        max_lenght = self.daily_lenght[int(Find_Max_Basic(self.daily_lenght)[1])]
        return days, self.daily_lenght, max_lenght

    def main(self):
        remain_path = self.euler_circuit
        for i in range(5):
            if i == 4:
                Split.calc_best_cutting(self,remain_path,True)
            else:
                remain_path = Split.calc_best_cutting(self,remain_path,False)

    def calc_best_cutting(self,remain_path,isFriday):
        full_daily_path=[[],[],[]] #In diese Variable wird der ganz Weg engespeichert, mit den einzelnen etappen
        path_distance_list = Split.calc_distance_path(self,remain_path) #speichert in path_distance_list die distanzen aller möglichkeiten
        if isFriday == False:
            full_distance_list=[]
            for path in path_distance_list: #erstellt die Variable full_distance_list
                reference_distance = self.mst.get_distance(remain_path[0]) + path[1]+ 2*self.mst.get_distance(path[0])
                full_distance_list.append([path[0],path[1], reference_distance,self.mst.get_distance(path[0])])#Form: [Name des Knoten,Weg_distanz,distanz zum Vergleichen,distanz zum Ursprung]
            main_path = []
            main_path_info = []
            for node in full_distance_list:
                if node[1] <= self.average_lenght//1: #das //1 soll die Zahl abrunden
                    main_path.append(node[0])
                    main_path_info.append(node)
                elif node[2] <= main_path_info[-1][2] and len(main_path)/(self.average_lenght)-1 <= 0.15: #Beschränkung, auf maximal 15% überschreitung des durchschnittes
                    main_path.append(node[0])
                    main_path_info.append(node)
                else:
                    break
            full_daily_path[1].extend(main_path)
            if main_path[-1] != 0:#Überprüft ob eine Rückfahrt nötig ist
                return_path = self.mst.get_path(main_path[-1])#Falls sie nötig sein sollt, so wird die kürzeste gesucht
                return_path = list(reversed(return_path)) #In diesem Fall muss man die Liste noch mal um drehen, weil es in diesem Fall vom einem Bestimmten Punkt zurück zum Ursprung geht und nicht wie normalerweise
                full_daily_path[2].extend(return_path)
            self.daily_lenght.append(self.mst.get_distance(remain_path[0])+main_path_info[-1][1]+self.mst.get_distance(main_path[-1])) #Berechnet die gesamte stecke und fügt es in die Variable
        else:
            full_daily_path[1].extend(remain_path)
            self.daily_lenght.append(self.mst.get_distance(remain_path[0])+path_distance_list[-1][1]) #Berechnet die gesamte Strecke speziell für den Freitag

        if remain_path[0] != 0: #Überprüft ob eine Hinfahrt nötig ist
            outward_path = self.mst.get_path(remain_path[0]) #Falls sie nötig sein sollt, so wird die kürzeste gesucht
            full_daily_path[0].extend(outward_path)

        self.day_paths.append(full_daily_path)
        if isFriday == False:
            remain_path = remain_path[len(main_path)-1:] #Dieser Code soll den schon abgefahrenen Weg entfernen, lässt aber noch den End Knoten da, da er später als Anfangs Knoten dienen wird
        return remain_path

    def calc_distance_path(self,path):
        distance = 0
        memory = [[path[0],0]] #Hier sollen die Distanzen aller Punkte gespeichert werden Form: [Name des Knoten, Entfernung vom Startknoten]
        for node in path[1:]: #Wir iterieren durch alle Knoten im Path außer dem ersten,weil wir den bereits im memory gespeichert haben.
            distance_delta = self.weight_matrix.item((memory[-1][0],node)) #Wir speichern die distanz zwischen dem Letzten Knoten und dem aktuellen in der Variable distance_delta
            distance = distance +distance_delta
            memory.append([node,distance])
        return memory

    def complete_lenght(self):
        shape = self.adjusted_matrix.shape[0]
        total_lenght = 0
        for i in range(shape):
            for j in range(shape):
                actual_num1 = self.weight_matrix.item((i,j))#Entnimmt der Matrix die Werte an der Stelle (i,j)
                actual_num2 = self.adjusted_matrix.item((i,j))
                product = actual_num1*actual_num2 #Multipliziert beide Werte
                total_lenght = total_lenght + product
        return total_lenght/2 #Da bei der Adjenzmatrix jeder Weg doppelt eingetragen ist muss man durch 2 teilen
#--------------------------------------------------------------------------------------------------
def get_information(filename):
    with open(filename) as f:
        lines = f.readlines()
    nodes = int(lines[0].strip().split(" ")[0]) #In der Variable nodes werden die Anzahl an Krezungen gespeichert
    edges = int(lines[0].strip().split(" ")[1]) #In der Variable edges werden die Anzahl an Straßen gespeichert
    edges_list = []
    for l in lines[1:]:
        l = l.strip()
        l = l.split(" ")
        for i in range(3):
            l[i] = int(l[i])
        edges_list.append(l) #Eine Staße wird mit der Form [1.Kreuzung, 2.Kreuzung, Distanz] zur Liste edges_list hinzugefügt
    return edges_list, nodes, edges

def print_results(split_paths,filename):
    days, daily_lenght, max_lenght = split_paths.get_final_paths()
    print(f"\n---{filename.upper()}---\n")
    for i,path in enumerate(days):
        string = ""
        arrow = " -> "
        for item in path:
            if len(string) == 0:
                string = string + str(item)
            else:
                string = string +arrow + str(item)
        print(f"Tag {i+1}: "+string + f", Gesamtlaenge: {int(daily_lenght[i])}")
    print(f"Maximale Lange einer Tagestour: {max_lenght}\n")

if __name__ == '__main__':
    filenames = ["muellabfuhr0.txt","muellabfuhr1.txt","muellabfuhr2.txt","muellabfuhr3.txt","muellabfuhr4.txt","muellabfuhr5.txt","muellabfuhr6.txt","muellabfuhr7.txt","muellabfuhr8.txt",]#"muellabfuhr0.txt","muellabfuhr1.txt","muellabfuhr2.txt","muellabfuhr3.txt","muellabfuhr4.txt","muellabfuhr5.txt","muellabfuhr6.txt","muellabfuhr7.txt","muellabfuhr8.txt",
    for filename in filenames:
        data,nodes,edges = get_information(filename)
        graph = Graph(data,nodes,edges)
        f = Fleury(nodes,graph.adjusted_matrix)
        splited_paths = Split(nodes,graph.adjusted_matrix,graph.weight_matrix,f.euler_circuit)
        print_results(splited_paths,filename)
