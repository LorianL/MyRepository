import matplotlib.pyplot as plt
from typing import List,Tuple,Set,Dict
import math
import copy

#--------------main part---------------------------
class Point: #Diese Klasse soll die Eingenschaften der Außenstellen(=>Einfachheitshalber als Punkt bezeichnet) speichern
    def __init__(self, id: int, coordinates: tuple) -> None:
        self.id = id
        self.coordinates = coordinates
class Connection:
    def __init__(self,ring_id,point_a_1,point_a,point_b,point_b_1) -> None:
        self.ring_id = ring_id
        self.point_a = point_a
        self.point_b = point_b
        self.point_a_1 = point_a_1
        self.point_b_1 = point_b_1
        self.valid_points = []

    def find_valid_points(self,points_dict) -> None:
        for p in points_dict.values():
            if p.id in (self.point_a_1.id,self.point_a.id,self.point_b.id,self.point_b_1.id):
                 continue
            angle_a = cal_angle(self.point_a_1, self.point_a, p)
            angle_b = cal_angle(self.point_b_1, self.point_b, p)
            if angle_a <= 90 and angle_b <= 90:
                self.valid_points.append(p)

class Onion_Ring:
    def __init__(self, points: Dict[int,Point], ring_id) -> None:
        #Definiere die Benötigten Variabeln:
        self.ring_id: int = ring_id
        self.points: Dict[int,Point] = points
        self.copy_points = copy.deepcopy(points) #Erstellt ein Backup
        self.mean_point: Point = cal_mean_point(self.points)
        self.outermost_point: Point = Onion_Ring.find_outermost_point(self)
        self.points_on_ring: List[Point] = [self.outermost_point]
        self.isValid: bool = None
        self.inValid_points: list = []

        #Erstelle eine Variable, in welcher gespeichert wird, ob der Onion_Ring gültig ist. Also keine ungültigen Winkel hat.
        Onion_Ring.create_ring(self)
        #Wichtig ist, dass remaining_points vor der check_angles() Funktion aufgerufen wird. Da in check_angles() wieder Punkte aus dem Ring entfernt werden, diese Punkte sollen dennoc nicht in remaining Points integriert werden,
        #da sie in keinem Kreis integriert werden können und somit den Algorithus in eine mögliche endlos schleife schickt
        Onion_Ring.find_invalid_point(self)
        self.remaining_points:  Dict[int,Point] = Onion_Ring.find_remaining_points(self)
        if self.isValid:
            Onion_Ring.optimize_ring(self)
            #Diese Funktion dient nur der Visualisierung der Resultate
            plot_ring_of_points(self.points_on_ring,color="black")
        #print(f"Anzahl an Punkten: {len(self.points)}; Anzahl an remaining_points: {len(self.remaining_points)}; Anzahl an Invalid Punkten: {len(self.inValid_points),list(map(map_point_to_id,self.inValid_points))}; Punkte auf dem Ring: {len(self.points_on_ring)}; {self.isValid}")#list(map(map_point_to_id,self.remaining_points.values()))

        #print(self.ring_id,len(self.remaining_points),len(self.inValid_points))

    #Diese Funktion soll den Ring so optimieren, dass er noch mehr Punkte mit einschließt ohne Punkte die sich bereits auf dem Ring befinden zu entfernen und ohne die gültigkeit der Winkel zu verlieren
    def optimize_ring(self) -> None:
        for _ in range(1):
            #Der Ring soll in beide Richtungen Optimiert werden:
            self.points_on_ring.reverse()
            #Diese Funktion soll so lange laufen, bis der gesamte Ring optimiert wurde.
            counter: int = 0
            while True:
                #Da wir uns für die Punkte zwischen zwei Punkten interessieren, paaren wir die Punkte.
                # Man verwende zip() um die Elemente in Paaren zu gruppieren
                # Der letzte Wert wird durch das erste und letzte Element gebildet
                point_pairs = list(zip(self.points_on_ring, self.points_on_ring[1:] + [self.points_on_ring[0]]))
                #wähle ein neues Paar aus, der counter gibt an bei welchem Paar wir uns im moment befinden
                point_A,point_B = point_pairs[counter]
                possible_points = Onion_Ring.find_possible_points(self,point_A, point_B)

                #findet die Indexe von Point_A und B in self.points_on_ring
                index_A = self.points_on_ring.index(point_A)
                index_B = self.points_on_ring.index(point_B)
                #Die folgende Schleife iteriert durch alle möglichen Punkt von kleinster distanz zu größter
                condition_met = False
                for p in possible_points:
                    #überprüft ob der ausgewählte Punkt gültige Winkel hat.
                    condition = [
                    (cal_angle(self.points_on_ring[index_A-1], point_A, p) <= 90),#überprüft den Winkel zwischen dem Punkt vor A, Punkt A und p
                    (cal_angle(point_A, p, point_B) <= 90),#überprüft den Winkel zwischen dem Punkt A, p und Punkt B
                    (cal_angle(p, point_B,self.points_on_ring[index_B+1-len(self.points_on_ring) if index_B+1 >= len(self.points_on_ring) else index_B+1]) <= 90)#überprüft den Winkel zwischen dem p,Punkt_B und Punkt nach B (überprüft ebenfalls on Punkt_B+1 nach im Index-Bereich liegt)
                    ]
                    #Falls alle Konditionen zutreffen:
                    if all(condition):
                        #wir weisen ihm den passenden Winkel zu:
                        p.relative_angle = cal_angle(point_A, p, point_B)
                        #Wir schieben den Punkt zwischen index_A und index_B
                        self.points_on_ring.insert(index_B,p)
                        #Wir entfernen den Punkt von den remaining_points
                        del self.remaining_points[p.id]
                        condition_met = True
                        break
                if condition_met:
                    counter -= 3
                else:
                    counter += 1
                #überprüft ob alle Punkte überprüft wurden
                if counter == len(self.points_on_ring):
                    break

    #Diese Funktion gibt einen ungültigen Punkt zurück und falls es keinen gibt so wird None zurück gegeben
    def find_invalid_point(self) -> Point:
        #sortiere die Punkte nach ihrer Entfernung vom mean_point. Die am weitesten entfernten, kommen als erstes:
        sorted_points: list =  sorted(self.points_on_ring, key=lambda point: cal_distance(point, self.mean_point))
        for p in sorted_points:
            #berechne den Winkel von Punikt p:
            index_p = self.points_on_ring.index(p)
            point_infront = self.points_on_ring[index_p-1] #Der Punkt bevor p
            point_behind = self.points_on_ring[0 if index_p+1 == len(self.points_on_ring) else index_p+1]
            p_angle = cal_angle(point_infront,p,point_behind)
            #Überprüft ob der Winkel zwischen den beiden Graden größer als 90 Grad ist.
            if p_angle > 90:
                #Falls der Punkt ungültig ist:
                self.isValid = False
                self.inValid_points.append(p)
                index = self.points_on_ring.index(p)
                self.points_on_ring.remove(p)
                return p
        #Falls keine ungültigen Punkte gefunden wurden, ist der Ring gültig
        if len(self.inValid_points) == 0:
            self.isValid = True
            return None

    #Diese Funktion soll alle Punkte finden, welche noch nicht in dem Kreis integriert sind.
    def find_remaining_points(self):
        #Kopiere die Punkte in die Variable remaining_points
        remaining_points = copy.deepcopy(self.copy_points)
        #Entferne aus der Liste der übrigen Punkte alle Punkte welche auf dem Ring sind, falls der Ring Valdie ist
        if self.isValid:
            for p in self.points_on_ring:
                del remaining_points[p.id]
        else:
            #Entferne alle Punkte welche inValide sind:
            for p in self.inValid_points:
                del remaining_points[p.id]

        return remaining_points

    #Diese Funktion erstellt einen ersten Ring, ohne auf die Korrektheit der Winkel zu achten
    def create_ring(self) -> None:
        #Finde den zweiten Punkt und füge den Winkel zu diesem hinzu:
        second_point = Onion_Ring.find_next_point(self,self.mean_point,self.outermost_point)
        self.points_on_ring.append(second_point)
        #die While-Schlife soll so lange laufen, bis wir wieder an einem Punkt angekommen sind den wir bereits besucht haben.
        while True:
            #Definiere Punkt_a und Punkt_b
            point_a = self.points_on_ring[-2]
            point_b = self.points_on_ring[-1]
            #Finde den nächsten Punkt:
            next_point = Onion_Ring.find_next_point(self,point_a,point_b)
            #Überprüft ob der Punkt bereits in der Liste ist, falls nicht, so wird er hinzugefügt
            if next_point in self.points_on_ring:
                break
            else:
                self.points_on_ring.append(next_point)

    #Diese Funktion soll den nächten Punkt finden, der in den man in den Ring implementieren will
    #Diese Funktion achtet nicht auf die Korrektheit der Winkel, sondern nimmt immer die Punkte mit dem größten Winkel
    def find_next_point(self,point_a: Point, point_b: Point) -> Point:
        #Erstelle eine Liste in welcher man alle Winkel mit samt des dazu gehörigen Punktes in Form eines tupels speichert
        relative_angles: List[Tuple[float,Point]] = []
        for p in self.points.values():
            #Überprüft, ob p nicht point_a oder point_b ist.
            if p == point_a or p == point_b:
                continue
            #Berechne den Winkel zwischen den beiden Vektoren (point_a,point_b) und (point_b,p)
            angle = cal_angle(point_a,point_b,p)
            #Füge den gefundenen Winkel mit dem dazugehörigen Punkt in einen Tupel und füge ihn zu relative_angles hinzu
            tuple = (angle,p)
            relative_angles.append(tuple)
        #sortiere die Liste relative_angles anhand des ersten Wertes (dem Winkel) von klein nach groß.
        relative_angles.sort(key=sort_first)
        #finde den Tupel mit dem kleinsten Winkel und gib den darin enthaltenen Punkt zurück
        smallest_angle: Point = relative_angles[0][1]
        return smallest_angle

    #Diese Funktion findet den Punkt der am weitesten vom Mittelpunkt entfernt ist.
    def find_outermost_point(self) -> Point:
        point_spacing: List[Tuple[float,Point]] = []
        for p in self.points.values():
            #Berechne die Distanz zwischen dem mean_point und dem Punkt p
            distance: float = cal_distance(self.mean_point,p)
            # Füge die gefundene Distanz mit dem Punkt p in ein Tuple und füge es der Liste hinzu
            tuple = (distance,p)
            point_spacing.append(tuple)

        #sortiere die Liste point_spacing anhand des ersten Wertes (der Distanz) von klein nach groß.
        point_spacing.sort(key=sort_first)
        #finde den Tupel mit der größten Distanz und gib den darin enthaltenen Punkt zurück
        outermost_tuple = point_spacing[-1]
        return outermost_tuple[1]

    #Diese Funktion soll alle Punkte finden, welche theoretisch einen Winkel kleiner als 90 Grad haben, wenn man sie mit Punkt_A und B verbindet
    def find_possible_points(self,point_A: Point, point_B: Point) -> list:
        #Berechnet den Vektor zwischen Punkt A und B
        pair_vector: Tuple[float,float] = (point_B.coordinates[0]-point_A.coordinates[0],point_B.coordinates[1]-point_A.coordinates[1])
        ### Implementiere die hergeleitete Theorie:
        coor_A: Tuple[float,float] = point_A.coordinates
        coor_B: Tuple[float,float] = point_B.coordinates
        help_point_1: Tuple[float,float] = (coor_A[0]+ (pair_vector[0]+pair_vector[1])/2,coor_A[1]+ (pair_vector[1]-pair_vector[0])/2)
        help_point_2: Tuple[float,float] = (coor_A[0]+ (pair_vector[0]-pair_vector[1])/2,coor_A[1]+ (pair_vector[1]+pair_vector[0])/2)
        #Wir suchen alle Punkte, welche eine möglichkeit haben, sich in dem Bereich zu befinden, in welchem es möglich ist, dass sie eine optimierung darstellen
        possible_points: list = []
        for p in self.remaining_points.values():
            p_coor: Tuple[float,float] = p.coordinates
            conditions: List[bool] = [
            sorted([coor_A[0],coor_B[0],p_coor[0]])[1] == p_coor[0] and sorted([coor_A[1],coor_B[1],p_coor[1]])[1] == p_coor[1], #Überprüft die Box zwischen Punkt A und B
            sorted([coor_A[0],help_point_1[0],p_coor[0]])[1] == p_coor[0] and sorted([coor_A[1],help_point_1[1],p_coor[1]])[1] == p_coor[1], #Überprüft die Box zwischen Punkt A und 1
            sorted([coor_A[0],help_point_2[0],p_coor[0]])[1] == p_coor[0] and sorted([coor_A[1],help_point_2[1],p_coor[1]])[1] == p_coor[1], #Überprüft die Box zwischen Punkt A und 2
            sorted([coor_B[0],help_point_1[0],p_coor[0]])[1] == p_coor[0] and sorted([coor_B[1],help_point_1[1],p_coor[1]])[1] == p_coor[1], #Überprüft die Box zwischen Punkt B und 1
            sorted([coor_B[0],help_point_2[0],p_coor[0]])[1] == p_coor[0] and sorted([coor_B[1],help_point_2[1],p_coor[1]])[1] == p_coor[1] #Überprüft die Box zwischen Punkt B und 2
            ]
            #überprüft ob eine der Bedingungen zutrifft und ob Punkt p nicht punkt_A oder B sind
            if any(conditions) and p not in [point_A,point_B]:
                possible_points.append(p)
        #Sortiert die Punkte nach entfernung vom Punkt pair[0]
        length_sorted = sorted(possible_points, key=lambda point: cal_distance(point, point_A))
        #Gibt eine Liste zurück mit allen möglichen Punkten, sortiert von der kleinsten Distanz zu Punkt_A nach der größten
        return length_sorted

class Onion_Algorithm:
    def __init__(self, points_dict: Dict[int,Point]) -> None:
        self.points_dict: Dict[int,Point] = copy.deepcopy(points_dict)
        self.rings_dict: Dict[int,list] = {} #In diesem Dictionary werden alle Ringe gespeichert. Die Ringe sind von außen nach innen hin nummeriert.
        self.inValid_points: list = []
        #print("Winkel: ",cal_angle(self.points_dict[22],self.points_dict[50],self.points_dict[33]))
        Onion_Algorithm.build_onion_rings(self)

    def build_onion_rings(self):
        counter = 0
        #Erstelle einen Dict, in welche die remaining_points für den nächsten Durchlauf gespeichert werden. Als Startwert, werden alle Punkte genommen
        remaining_points: dict = copy.deepcopy(self.points_dict)
        while True:
            #print(f"Anzahl an Invaliden Punkten: {len(self.inValid_points)} => {self.inValid_points}")
            next_Onion_ring = Onion_Ring(remaining_points,counter)
            #Definiert die neue remaining_points Liste
            remaining_points = next_Onion_ring.remaining_points
            #Entnimmt die inValid_points vom Ring und fügt es in die Liste inValid_points
            self.inValid_points.extend(next_Onion_ring.inValid_points)
            #überprüft, ob der Ring überhaupt gültige Punkte besitzt, falls nicht, so wird dieser übersprungen
            if len(next_Onion_ring.points_on_ring) > 0 and next_Onion_ring.isValid == True:
                self.rings_dict[counter] = next_Onion_ring
                #Falls der Ring hinzugefügt wurde geht der counter um 1 hoch
                counter += 1
            #Diese Funktion überprüft, ob es sich um die übrigen Punkte um die letzten handelt. Falls nur noch 4 Punkte übrig bleiben, existiert nur noch eine die möglichkeit einer Form mit 4 Rechtenwinkeln, welche sehr unwahrscheinlich ist.
            #Es lohnt sich in dem Fall mehr direkt einen anderen Algorithus anzuwenden, anstatt die Rechenzeit mit einer Aufgabe aufzublähen die unlösbar ist.
            if len(remaining_points) < 4:
                #fügt die übrig gebliebenen Punkte zu den Invaliden Punkten hinzu
                self.inValid_points.extend(remaining_points.values())
                break
        print(len(self.inValid_points))

#Dieser Algorithmus soll die Verschiedenen Teile vereinen.
class Crossing_Algorithm:
    def __init__(self,Onion_Algorithm,points_dict):
        self.onion_class = Onion_Algorithm
        self.rings: dict = self.onion_class.rings_dict
        self.points_dict = points_dict
        self.connections = Crossing_Algorithm.init_connections(self)
        self.path: list = []

    #Diese Funktion erstellt die Klassen für alle Verbindungen
    def init_connections(self) -> list:
        connections = []
        for r in self.rings.values():
            ring = r.points_on_ring
            ring_id = r.ring_id
            #erstellt die benötigten Paare, jeder Paar besteht aus 4 Punkten
            #Punkt [1] und [2] sind jeweils die Punkte zwischen welchen die Verbindung liegt und Punkt [0] und [3] sind jeweils point_a/b_1
            point_pairs = list(zip(ring,ring[1:]+ring[0:1],ring[2:]+ring[0:2],ring[3:]+ring[0:3]))
            for p in point_pairs:
                con = Connection(ring_id,p[0],p[1],p[2],p[3])
                con.find_valid_points(self.points_dict)
                connections.append(con)
        return connections


    #diese Funktion versucht
    def combine_rings(self):
        counter = 0
        while True:
            #überprüft ob es noch Ringe gibt
            if len(self.rings) == 0:
                break
            #Wählt einen Ring aus und wandelt ihn in Punkt paare um
            next_pick = self.rings.values()[counter]
            next_ring = next_pick.points_on_ring
            point_pairs = list(zip(next_ring, next_ring[1:] + [next_ring[0]]))
            #iteriert durch alle Punktpaare
            for pp in point_pairs:
                #iteriert durch die Ringe
                for target_pick in self.rings.values():
                    #Überprüft ob es sich bei ausgewählten Ring nicht um einen schon benuten handelt
                    if target_ring == next_pick:
                        continue
                    target_ring = next_pick.points_on_ring
                    target_pairs = list(zip(next_ring, next_ring[1:] + [next_ring[0]]))
                    for tp in target_pairs:
                        pass


#----------------additional functions----------------------------
#Diese Funktion berechnet die länge zwischen zwei Punkten. Ein Tupel sieht wie folgt aus: (x-Koordinate,y-Koordinate)
def cal_distance(point_a: Point, point_b: Point) -> float:
    #Wandle die Punkte in Koordinaten um:
    coordinates_a: tuple = point_a.coordinates
    coordinates_b: tuple = point_b.coordinates
    #Nimm die Differenz der Koordinaten, sowohl x-Achse als auch y-Achse
    delta_x = abs(coordinates_a[0]-coordinates_b[0])
    delta_y = abs(coordinates_a[1]-coordinates_b[1])
    #Verwende den Satz des Pythagoras
    distance = (delta_x**2 + delta_y**2)**0.5
    return distance

#Diese Funktion soll den Winkel zwischen beiden Vektoren aus ab und bc berechnen
def cal_angle(a_point: Point, b_point: Point, c_point: Point) -> float:
    #wandle die Punkte in Koordinaten um:
    a_coor = a_point.coordinates
    b_coor = b_point.coordinates
    c_coor = c_point.coordinates
    #Berechne die Vektoren p_a und a_n.
    vector_ab: tuple = (b_coor[0]-a_coor[0],b_coor[1]-a_coor[1])
    vector_bc: tuple = (c_coor[0] - b_coor[0], c_coor[1] - b_coor[1])
    #Berechne das Skalar-Produkt von Vektor p_a und a_n:
    skalar: float = vector_ab[0]*vector_bc[0] + vector_ab[1]*vector_bc[1]
    #Berechne die Länge der Vektoren p_a und a_n
    length_ab = cal_distance(a_point,b_point) #Hierzu verwenden wir eine Funktion, welche wir bereits oben erstellt haben.
    length_bc = cal_distance(b_point,c_point)
    #Berechne den Cosinus-Wert des Winkels:
    cos_value = skalar / (length_ab*length_bc)
    #Fange eventuelle Fehler ab, dies kann eventuelle geschehen, wenn bei einer der vorherigen Rechnungen ungünstig gerundet wird
    if cos_value > 1:
        cos_value = 1
    elif cos_value < -1:
        cos_value = -1
    #Wandle den Cosinus-Wert mit hilfe der arccos Funktion in ein Winkel um:
    angle = math.acos(cos_value)
    #Wandle Bogenmaß in Gradmaß um
    angle_degree: float = math.degrees(angle)
    return angle_degree

#Berechnet den Mittleren Punkt, aus einer Liste von Punkten
def cal_mean_point(points_dict: Dict[int,Point]) -> Point:
    coordinates = list(map(map_point_to_coordinates,list(points_dict.values())))
    sum_x = 0
    sum_y = 0
    for x,y in coordinates:
        sum_x += x
        sum_y += y
    mean_x = sum_x/len(coordinates)
    mean_y = sum_y/len(coordinates)
    mean_point = Point(-1,(mean_x,mean_y))
    return mean_point
#-----------------sort / mapping functions -------------------------
def map_point_to_coordinates(point: Point) -> tuple: #diese Funktion soll eine Liste von Punkten in eine Liste von Koordinaten umwandeln
    coordinates = point.coordinates
    return coordinates
def map_point_to_id(point: Point) -> int: #diese Funktion soll eine Liste von Punkten in eine Liste von id umwandeln
    if type(point) is Point:
        id = point.id
        return id
    return None
def sort_first(input: tuple): #Diese soll eine Liste / Tupel nach dem ersten Wert sortieren PS: von klein nach groß
    return input[0]
def map_pair_up(x,y):
    return (x,y)
#------------------Get information from files---------------------------
#Diese Funktion soll die Informationen einlesen und verarbeiten
def read_files(filename: str) -> Dict[int,Point]:
    with open(f"C:/Users/Lorian/Desktop/BWINF 41 Runde 2/bundeswettbewerb41-runde2/A1-WenigerKrummeTouren/{filename}") as file:
        lines = file.readlines()
    points_dic = dict()

    for i,l in enumerate(lines):
        l = l.strip()
        l = l.split(" ")
        coordinates: tuple = (float(l[0]),float(l[1]))
        point = Point(i,coordinates)
        points_dic[i] = point
    return points_dic

#-----------------plot results---------------------

def label_points(points_dict,fly_map):
    mean_point: Point = cal_mean_point(points_dict)
    plt.scatter(mean_point.coordinates[0],mean_point.coordinates[1],color= "blue")
    rings: dict = fly_map.rings_dict
    for r in rings.values(): #c: Point
        for c in r.points_on_ring:
            #plt.scatter(c.coordinates[0],c.coordinates[1],color="black")
            #plt.annotate(f"{c.id}",xy=c.coordinates)
            continue
    for c in fly_map.inValid_points:
        plt.scatter(c.coordinates[0],c.coordinates[1],color="purple")
        plt.annotate(f"{c.id}",xy=c.coordinates)

def plot_2_points(id_a: int,id_b: int,color: str = "red") -> None: #Diese Funktion soll die Punkte a und b mit einer Linie verbinden
    #Entnehme die Koordinaten der beiden Punkte und füge sie in eine Liste:
    x_coordinates = [original_points_dict[id_a].coordinates[0],original_points_dict[id_b].coordinates[0]]
    y_coordinates = [original_points_dict[id_a].coordinates[1],original_points_dict[id_b].coordinates[1]]
    #Zeichne eine Linie zwischen den Beiden Punkten im Graphen:
    plt.plot(x_coordinates,y_coordinates,color = color)

def plot_ring_of_points(list_points: List[Point],color: str = "red") -> None:
    coordinates = list(map(map_point_to_coordinates,list_points))
    if len(coordinates) == 0:
        return None
    x_coordinates = []
    y_coordinates = []
    for c in coordinates:
        x_coordinates.append(c[0])
        y_coordinates.append(c[1])
    x_coordinates.append(x_coordinates[0])
    y_coordinates.append(y_coordinates[0])
    plt.plot(x_coordinates,y_coordinates,color = color)

#-------------start function----------------------
if __name__ == '__main__':
    for _ in range(10):
        file_number = int(input("int => ")) # Dokumente 1-4 sind konstruiert der rest ist zufällig
        points_dict = read_files(f"wenigerkrumm{file_number}.txt")
        original_points_dict = copy.deepcopy(points_dict) #Diese Variable soll immer das Original besitzen
        fly_map = Onion_Algorithm(points_dict)
        Crossing_Algorithm(fly_map,points_dict)
        label_points(points_dict,fly_map)
        plt.show()






"""#überprüft ob noch immer mindestens 4 Punkte da sind, falls nicht so ist der gesamte Ring ungültig:
if len(self.points_on_ring) < 4:
    #Falls dies der Fall ist, so wir die Liste komlett zu inValid_points hinzugefügt und durch eine Leere ersetzt
    self.inValid_points.extend(self.points_on_ring)
    self.points_on_ring = []
    return
#Nun wo der Punkt weg ist, haben sich ebenfalls die Winkel für p1 = points_on_ring[index] und p2 = points_on_ring[index-1] und  geändert (Sie waren die beiden benachbarten Punkte)
#in einem ersten Schritt werden die Punkte für die neu berechnung der Punkte in einer Liste gespeichert
p1 = [index-1,index,index+1] #in der Mitte (index 1) steht immer der betroffene Punkt
p2 = [index-2,index-1,index]
#überprüft ob es keine ungültigen indexe gibt, dazu iteriert man durch alle Indexe.
for i,(n_1,n_2) in enumerate(zip(p1,p2)):
    #Entpacke den Tupel:
    #Falls der Index n_1/2 größer oder gleich der Länge der Liste ist(und somit außerhalb des Index bereich) wird dieser Index durch n_1-len(self.points_on_ring) ersetzt
    #n_1-len(self.points_on_ring) diese Rechnung macht im Fall, dass der Wert außerhalb des index Bereich liegt nicht anderes als wieder von index 0 an zu zählen
    p1[i] = n_1-len(self.points_on_ring) if n_1 >= len(self.points_on_ring) else n_1
    p2[i] = n_2-len(self.points_on_ring) if n_2 >= len(self.points_on_ring) else n_2
#berechne für die betroffenen Punkte neue Winkel:
self.points_on_ring[p1[1]].relative_angle = cal_angle(self.points_on_ring[p1[0]], self.points_on_ring[p1[1]], self.points_on_ring[p1[2]])
self.points_on_ring[p2[1]].relative_angle = cal_angle(self.points_on_ring[p2[0]], self.points_on_ring[p2[1]], self.points_on_ring[p2[2]])
#Starte die Funktion von vorne:
Onion_Ring.check_angles(self)
#Beendige die jetzige sitzung:
return
def label_points(points_dict: Dict[int,Point]):
    mean_point: Point = cal_mean_point(points_dict)
    plt.scatter(mean_point.coordinates[0],mean_point.coordinates[1],color= "blue")
    for c in points_dict.values(): #c: Point
        plt.scatter(c.coordinates[0],c.coordinates[1],color="black")
        plt.annotate(f"{c.id}",xy=c.coordinates)



"""
