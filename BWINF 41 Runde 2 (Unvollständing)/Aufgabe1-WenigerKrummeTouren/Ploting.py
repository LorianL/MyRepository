import matplotlib.pyplot as plt
from typing import Tuple
from typing import List
import math

#------------------Get information from files---------------------------
def read_files(filename: str) -> list:
    with open(filename) as file:
        lines = file.readlines()
    x_coordinates = []
    y_coordinates = []

    for l in lines:
        l = l.strip()
        l = l.split(" ")
        x_coordinates.append(float(l[0]))
        y_coordinates.append(float(l[1]))
    return list(zip(x_coordinates,y_coordinates))

#-----------------plot results---------------------

"""def plot_points(x_coordinates: float,y_coordinates: float):
    mean_x,mean_y = focus_point(x_coordinates,y_coordinates)
    plt.scatter(x_coordinates,y_coordinates,color = "black")
    #plt.plot(x_coordinates,y_coordinates,color = "red")
    plt.scatter(mean_x,mean_y,color="blue")
    plt.show()"""

def label_length(coordinates: list):
    print(len(coordinates))
    mean_point: tuple = focus_point(coordinates)
    plt.scatter(mean_point[0],mean_point[1],color= "blue")
    for c in coordinates: #c: tuple[float,float]
        distance = cal_slope_angle(mean_point,c)
        plt.scatter(c[0],c[1],color="black")
        plt.annotate(f"{int(distance)}",xy=c)
    plt.show()


def focus_point(coordinates: List[Tuple[float,float]]) -> Tuple[float,float]:
    sum_x = 0
    sum_y = 0
    for x,y in coordinates:
        sum_x += x
        sum_y += y
    mean_x = sum_x/len(coordinates)
    mean_y = sum_y/len(coordinates)
    return (mean_x,mean_y)

#----------------additional functions----------------------------
def cal_distance(point_a: Tuple[float,float],point_b: Tuple[float,float]) -> float: #Diese Funktion berechnet die länge zwischen zwei Punkten. Ein Tupel sieht wie folgt aus: (x-Koordinate,y-Koordinate)
    delta_x = abs(point_a[0]-point_b[0])
    delta_y = abs(point_a[1]-point_b[1])
    distance = (delta_x**2 + delta_y**2)**0.5
    return distance

def cal_slope_angle(point_a: Tuple[float,float],point_b: Tuple[float,float]) -> float: #der Basic_Winkel bedeutet sowie der Steigungswinkel der Geraden zwischen zwei Punkten
    #Berechne die Längen des Steigungsdreiecks und überprüft ob diese nicht 0 sind. Sollte dies der Fall sein, so wird 0 durch eine Zahl ersetzt die fast 0 ist:
    delta_x =  0.00001 if (point_a[0]-point_b[0]) == 0 else point_a[0]-point_b[0]
    delta_y = 0.00001 if (point_a[1]-point_b[1]) == 0 else point_a[1]-point_b[1]
    #Berechne die Steigung (Gegenkathete/Ankathete):
    slope: float = delta_y/delta_x
    #Berechne den Steigungswinkel mithilfe des Tangens(Gibt den Winkel in Bogenmaß zurück):
    angle_radiant: float = math.atan(slope)
    #Wandle Bogenmaß in Gradmaß um
    angle_degree: float = math.degrees(angle_radiant)
    return angle_degree

def cal_relative_angle(previous_point: tuple, actual_point: tuple, next_point: tuple) -> float: #Diese Funktion soll den Flug-Winkel berechnen.
    #Berechne die Steigungswinkel von dem vorherigen Punkt bis zum Aktuellen und den Steigungswinkel vom Aktuellen bis zum nächsten Punkt
    previous_actual_angle = cal_slope_angle(previous_point,actual_point)
    actual_next_angle = cal_slope_angle(actual_point, next_point)
    #Berechne die Differenz der beiden Winkel
    relative_angle: float = abs(actual_next_angle-previous_actual_angle)
    return relative_angle




#-------------start function----------------------
if __name__ == '__main__':
    for _ in range(10):
        file_number = int(input("int => ")) # Dokumente 1-4 sind konstruiert der rest ist zufällig
        coordinates = read_files(f"wenigerkrumm{file_number}.txt")
        label_length(coordinates)
