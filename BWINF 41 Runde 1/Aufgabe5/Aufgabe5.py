import numpy as np
import copy

def to_int(list): #Diese Funktion verwandelt eine Liste aus String Nummern in ints
    for i,num in enumerate(list):
        list[i] = int(num)
    return list

def map_func(input): #Diese Funktion soll die Indexe wieder um 1 erhöhen, da wir einfachheitshalber alle werte um 1 reduziert haben, weil Python mit dem Index 0 anfäng und die Aufgabenstellung mit dem Index 1
    input += 1
    return str(input) #ebenfalls wird der int() in ein str() umgewandelt um es besser ausgeben zu können


def import_graph(file_name):
    with open(file_name,"r") as file:
        file = file.readlines()
        sizes = to_int(file[0].strip().split()) #Entnimmt aus der 1 Zeile die Anzahl an Knoten und Verbindungen
        edges = [] #Hier werden alle Verbindungen gespeichert in Form von Tupeln (Anfang, Ende)
        for i in file[1:]: #überspringt die 1.Zeile
            edges.append(to_int(i.strip().split()))
    matrix = np.zeros((sizes[0],sizes[0])) #Erstellt die Matrix
    for e in edges:
        matrix.itemset((e[0]-1,e[1]-1),1) #Definiert die einzelnen Verbindungen in einer Matrix. Man zieht jeweils (a-1,b-1), da wir bei 0 anfangen zu Zählen während die Beispiele bei 1 beginnt
    return matrix #Gibt die Matrix zurück

def get_list_to_nodes(nodes_list): #Diese Funktion soll die Werte aus der Adjenz_matrix extrahieren. In der Adjenz_matrix ist eine Verbindung mit einer 1 bei der entsprechenden Koordinate definiert. In diesem Fall haben wir eine Vereinfachte Matrix(1D)
    nodes = [] #Wir wollen in unserer Liste nur die Koordinaten, aus der Liste, bei welcher der Wert == 1 ist.
    for i,n in enumerate(nodes_list):
        if n == 1: #falls die node[i] == 1 ist bedeutet das dass diese note eine Input- bzw. Output_node ist
            nodes.append(i)
    return nodes

def create_nodes(matrix): #Diese Funktion soll aus den Matrixen die Nodes erstellen
    amount = matrix.shape[0] #Die amount gibt an wie viele Nodes es insgesamt gibt
    nodes_list = [] #Die Liste mit den Nodes
    for i in range(amount): #Iteriert durch die Indexe der Nodes
        input_nodes = get_list_to_nodes(matrix[:,i]) #Entnimmt die Input_nodes aus der Adjenz_matrix
        output_nodes = get_list_to_nodes(matrix[i,:]) #Entnimmt die Output_nodes aus der Adjenz_matrix
        n = Node(i,input_nodes,output_nodes,3*amount) #Als visit_limit nehmen wir die Anzahl an Nodes*3
        nodes_list.append(n)
    return nodes_list


class Node:
    def __init__(self,node_index,input_nodes,output_nodes,visit_limit):
        self.node_index = node_index
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.weight_set = set() #Hier werden alle Gewicht drin gespeichert. Dieses Set soll es erleichtern zu überprüfen ob es ein Gewicht bereits gibt und später beim abgleich mit den anderen Nodes
        self.waiting_dic = Node.init_dic(self,self.output_nodes) #In diesem Dic werden die Gewichte gespeichert, welche die Output_nodes noch benötigen PS: die Gewichte wurden bereits angepasst dh. Gewicht +1
        self.visit_counter = 0 #diese Variable soll zählen wie oft die Node besucht worden ist
        self.visit_limit = visit_limit#visit_limit

    def init_dic(self,list): #Erstellt aus einer Liste ein Dic, die Elemente der Liste dienen dabei als Keys und jeder Key wird jeweils als leere Liste definiert
        dic = {}
        for i in list:
            dic[i] = []
        return dic

    def start_node(self): #initialisiert diese Node als Ursprung
        self.weight_set.add(0) #Da der Ursprung das Gewicht 0 hat
        to_do = copy.deepcopy(self.output_nodes)
        for n in self.output_nodes: #Da keine Node bislang einen Wert zugewiesen bekommen hat, bedeutet das dass keine Node bislang das Gewicht 1 hat
            self.waiting_dic[n].append(1)
        return to_do #Gibt die Nodes wieder welche zum Waiting_set hinzugefügt werden müssen

    def update_node(self,nodes_list): #Diese Funktion soll die Node und ihre Variabeln mit den neuen Informationen Updaten
        to_do = set()
        self.visit_counter +=1 #Erhöht den Zähler um 1
        for n in self.input_nodes:
            node = nodes_list[n] #Da self.input_nodes nur den Index der Input_nodes speichert, wird in diesem Schritt die entsprechende Klassse entnommen
            new_weights = node.waiting_dic[self.node_index] #Wir rufen das Waiting_dic von der entsprechenden Node auf und entnehmen die gewichte die zu unserer Node passen
            for w in new_weights: #Nun iterieren wir durch die neuen Gewichte.
                if w not in self.weight_set and self.visit_counter < self.visit_limit: #überprüft ob dieses Gewicht nicht bereits vorhanden ist und ob man nicht bereits das visit_limit überschritten hat
                    self.weight_set.add(w) #Fügt w in weight_set
                    to_do = to_do.union(Node.check_outputs(self,w,nodes_list)) #Fügt die gefundenen Outputs ins to_do set
        return to_do


    def check_outputs(self,weight,nodes_list): #Diese Funktion soll überprüfen welche der Output_nodes nochmal geupdatet werden müssen und gibt ein Set mit diesen zurück(Indexe)
        to_do_set = set()
        for i in self.output_nodes: #iteriert durch alle Output_Nodes
            n = nodes_list[i] #Da die Output_nodes als indexe gespeichert sind, muss man sie noch in die Eigentliche Node um wandeln
            if weight+1 not in n.weight_set and n.visit_counter < n.visit_limit: #überprüft ob das Gewicht+1 noch nicht in n vorhanden ist und ob der visit_counter von n noch nicht das Limit erreicht hat
                to_do_set.add(n.node_index) #Falls dies zutrifft so wird der index von n zur ToDo_list hinzu gefügt
                self.waiting_dic[n.node_index].append(weight+1) #Und das Gewicht+1 wird zum waiting_dic unter dem Index von n eingespecieherrt
        return to_do_set


def main(nodes_list,start_index): #Dies soll die Haupt Funktion darstellen, welche alle anderen aufruft
    waiting_list = nodes_list[start_index].start_node() #Man initiert sowohl die start_node als auch das Waiting_list
    while True:
        next_index = waiting_list.pop(0) #Man entnimmt das [0] Element von waiting_list
        next_node = nodes_list[next_index] #Da es ein Index ist muss dieser noch dem entsprechendem Node zugewiesen werden
        to_do = next_node.update_node(nodes_list) #Diese Funktion gibt die nächsten zuberarbeitenden Nodes zurück
        for i in to_do:
            if i not in waiting_list: #iteriert durch die To_dos und überprüft ob sie nicht bereits in der Warte_liste sind
                waiting_list.append(i) #Falls sie noch nicht in der Waiting_list sind, so werden sie hinzugefügt
        if len(waiting_list) == 0: #Falls keine Elemente mehr im Waiting_list vorhanden sind, so ist der Algorithmus fertig
            return nodes_list

#-----------------------------------------------------------------------------------------

def check_intersection(nodes_Sasha,nodes_Mika): #Diese Funktion soll überprüfen ob sich Mika und Sasha überhaupt einmal auf einem Feld treffen
    for i in range(len(nodes_Sasha)): #Iteriert durch die Indexe von den Knoten von Mika und Sasha. Beide listen sind gleich lang dh. es macht kein unterschied ob wir da Sasha oder Mika einfügen
        intersection = nodes_Sasha[i].weight_set & nodes_Mika[i].weight_set #Der Operator & gibt ein Set zurück welcher die Gemeinsamen Gewicht von nodes_Sasha[i].weight_set und nodes_Mika[i].weight_set enthält
        if len(intersection) > 0: #Falls es eine übereinstimmung gibt bedeutet das dass es eine möglichkeit gibt das Parkour zu lösen
            return True #Falls es einen Treffer gibt, so wird True zurück gegeben
    return False #Ansonsten False

def get_smallest_intersection(nodes_Sasha,nodes_Mika): #Dies soll den Knoten angeben an welchem sich Mika und Sasha mit den wenigesten Sprüngen treffen
    smallest_intersection = None #Hier soll die Node gespeichert werden mit welcher man den Parkour am schnellsten lösen kann. In der Form (Gewicht,Index)
    for i in range(len(nodes_Sasha)): #Iteriert durch die Indexe von den Knoten von Mika und Sasha. Beide listen sind gleich lang dh. es macht kein unterschied ob wir da Sasha oder Mika einfügen
        intersection = nodes_Sasha[i].weight_set & nodes_Mika[i].weight_set #Der Operator & gibt ein Set zurück welcher die Gemeinsamen Gewicht von nodes_Sasha[i].weight_set und nodes_Mika[i].weight_set enthält
        for weight in intersection: #iteriert durch die übereinstimmungen, falls es keine gibt, so wird dieser Teil übersprungen und man beginnt wieder vom Anfang der Schleife
            if smallest_intersection is None: #Falls für smallest_intersection nach kein Tupel existiert,
                smallest_intersection = (weight,i) #So wird unabhängig davon was im Tupel ist, dieser als smallest_intersection definiert.
            elif smallest_intersection[0] > weight: #Falls man nun ein neue Übereinstimmung findet, welches ein kleineres Gewicht als das bereits bekannt,
                smallest_intersection = (weight,i) #so updatet man smallest_intersection mit diesem neuen Tupel
    return smallest_intersection #Gibt smallest_intersection zurück

def track_back(nodes_somebody,smallest_intersection): #Diese Funktion soll den Weg zurück finden, von der kleinsten Übereinstimmung zurück zum Ursprung
    # smallest_intersection => (weight,index)
    path = [smallest_intersection] #Speichert den Weg in Form von Tupeln. Diese wiederum sind aufgebaut wie smallest_intersection
    while True:
        if path[0][0] == 0:
            return path
        past_step = path[0] #Da der letzte Sprung immer vorne eingesetzt wird, erhält man mit path[0] den letzten Sprung
        path.insert(0,search(past_step,nodes_somebody)) #Fügt den neu gefundenen Sprung wieder an den Anfang, die Funktion search() findet den nächsten Sprung

def search(past_step,nodes_somebody): #Sucht nach dem Feld, welches das gesuchte Gewicht besitzt
    past_node = nodes_somebody[past_step[1]] #wandelt den Index in das tatsächliche Objekt um
    for node_index in past_node.input_nodes: #iteriert durch die Input_nodes
        weight_set = nodes_somebody[node_index].weight_set #Entnimmt den Input_nodes die Gewichtsliste
        if past_step[0]-1 in weight_set: #Falls das Gewicht-1 in dem weight_set der betrachteten Node ist,
                return (past_step[0]-1,node_index) #dann gibt man den Tupel zurück und subtrahiert von past_step[0] -1, da dies nun das nächst kleinere Feld ist.

def print_path(path,name):
    for i,(weight,field) in enumerate(path):
        path[i] = field
    path = list(map(map_func, path))
    string = " -> ".join(path)
    print(f"{name} muss auf folgende Felder springen um das Parkour zu absolvieren: {string}\n")


#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    filename_list = ["huepfburg0.txt","huepfburg1.txt","huepfburg2.txt","huepfburg3.txt","huepfburg4.txt"] #"huepfburg0.txt","huepfburg1.txt","huepfburg2.txt","huepfburg3.txt","huepfburg4.txt"
    for file_name in filename_list:
        print(f"\n-----------------{file_name.upper()}------------------\n\n")
        matrix = import_graph(file_name)
        nodes_list = create_nodes(matrix)
        nodes_Sasha = main(copy.deepcopy(nodes_list),0)
        nodes_Mika = main(copy.deepcopy(nodes_list),1)

        if check_intersection(nodes_Sasha,nodes_Mika):
            smallest_intersection = get_smallest_intersection(nodes_Sasha,nodes_Mika)
            sasha_path = track_back(nodes_Sasha,smallest_intersection)
            mika_path = track_back(nodes_Mika,smallest_intersection)
            print_path(sasha_path,"Sasha")
            print_path(mika_path,"Mika")
        else:
            print("Dieser Parkour kann nicht erfolgreich absolviert werden.\n")
