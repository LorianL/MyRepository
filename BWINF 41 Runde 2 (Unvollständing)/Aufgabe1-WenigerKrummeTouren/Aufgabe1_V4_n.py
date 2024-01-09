import matplotlib.pyplot as plt
from typing import List,Tuple,Set,Dict
import math
import copy
import numpy as np

#--------------main part---------------------------
class Point: #Diese Klasse soll die Eingenschaften der Außenstellen(=>Einfachheitshalber als Punkt bezeichnet) speichern
    def __init__(self, id: int, coordinates: tuple) -> None:
        self.id = id
        self.coordinates = coordinates

class Matrix:
    def __init__(self, points_dict) -> None:
        self.points_dict = points_dict
        self.num_of_nodes = len(points_dict) #Speichert die Anzahl an Punkten
        self.matrix = np.zeros((self.num_of_nodes, self.num_of_nodes, self.num_of_nodes)) #in der Variable self.matrix wird eine 3d Matrix Nullmatrix mit den abmaßen n*n*n
        Matrix.set_connections(self)

    def set_connections(self):
        #Entnimmt das Element für die x-Achse
        for x in self.points_dict.values():
            #Entnimmt das Element für die y-Achse
            for y in self.points_dict.values():
                #Entnimmt das Element für die z-Achse
                for z in self.points_dict.values():
                    #Überprüft ob im Tupel keine Duplikate sind und ob der Winkel größer 90 Grad ist.
                    if Matrix.check_doubles(self, (x,y,z)) and cal_angle(x,y,z) < 90:
                        self.matrix.itemset((x.id,y.id,z.id),1)


    def check_doubles(self, position: tuple) -> bool:
        pos_set = set(position) #Wandelt den tuple in ein Set um
         #Überprüft ob die länge dieses Set == 3 ist, falls dies zutrifft bedeutet das dass nicht zweimal die gleichen Punkte im tupel waren.
         #Wäre ein duplikat im Tupel, so würde das set diesen nur einmal speichern wodurch die Länge nicht mehr == 3 ist
        if len(pos_set) == 3:
            return True
        else:
            return False

class Solvable:
    def __init__(self,matrix):
        self.matrix = matrix
        self.size = self.matrix.shape[0] #Hier wird die Anzahl an Knoten gespeichert
        self.bool_list = [False] * self.size
        Solvable.check_grades(self)
        print(all(self.bool_list),self.size)

    #Diese Funkktion soll überprüfen ob der Gesamte Graph miteinadner Verbunden ist, bedeutet, dass nicht zwei separate Graphen vorliegen
    def check_connectivity(self):
        waiting_set = set()
        finished_set = set()

    def check_grades(self):
        print(self.matrix[:,0,:])
        for i in range(self.size):
            sum = np.sum(self.matrix[:,i,:])
            if sum > 0:
                self.bool_list[i] = True
            if sum >= self.size/2:
                print(i,sum,True)
            else:
                print(i,sum,False)

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

def label_points(points_dict):
    mean_point: Point = cal_mean_point(points_dict)
    plt.scatter(mean_point.coordinates[0],mean_point.coordinates[1],color= "blue")
    for c in points_dict.values():
        plt.scatter(c.coordinates[0],c.coordinates[1],color="black")
        plt.annotate(f"{c.id}",xy=c.coordinates)
def plot_2_points(id_a: int,id_b: int,color: str = "red") -> None: #Diese Funktion soll die Punkte a und b mit einer Linie verbinden
    #Entnehme die Koordinaten der beiden Punkte und füge sie in eine Liste:
    x_coordinates = [original_points_dict[id_a].coordinates[0],original_points_dict[id_b].coordinates[0]]
    y_coordinates = [original_points_dict[id_a].coordinates[1],original_points_dict[id_b].coordinates[1]]
    #Zeichne eine Linie zwischen den Beiden Punkten im Graphen:
    plt.plot(x_coordinates,y_coordinates,color = color)

#-------------start function----------------------
if __name__ == '__main__':
    for _ in range(10):
        file_number = int(input("int => ")) # Dokumente 1-4 sind konstruiert der rest ist zufällig
        points_dict = read_files(f"wenigerkrumm{file_number}.txt")
        matrix = Matrix(points_dict).matrix
        Solvable(matrix)
        label_points(points_dict)
        plt.show()
