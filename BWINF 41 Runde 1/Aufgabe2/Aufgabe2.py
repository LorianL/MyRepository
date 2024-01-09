import numpy as np
import random
from PIL import Image

def get_seeds(num,grow_speed_range,maxdelay,color_range=(100,200)): #Diese Funktion soll die Kristal_Klassen initalisieren; num => gibt die Anzahl an Seeds an
    resolution=(720,1280) #(x-Achse,y-Achse)
    crystals_list = [] #In dieser Liste werden die unterschiedlichen Kristale gespeichert
    seeds_list = [] #Diese Liste hilft zu überprüfen, ob eine Koordinaten Paar bereits benutzt wurde
    delta_color = (color_range[1]-color_range[0])/num #delta_color gibt die benötigte differenz an,
    #um einer num-Anzahl an Pixel eine gleichmäßig aber unterschiedliche Anzahl an Grautöne in einer Farbbreite von color_range[0] bis color_range[1] zu geben
    for i in range(num):
        while True: #wiederholt, so oft den Prozess, bis ein Seed gefunden wurden, welcher noch nicht benutzt wurde. (Ist vor allem bein höheren Werten für num wichtig)
            seed_pos = (random.randint(0,resolution[0]-1),random.randint(0,resolution[1]-1)) #Gibt ein zufälliges Koordinaten Paar, zwischen den Grenzen von x -> {0,720} und y -> {0,1280} wieder
            if seed_pos not in seeds_list: #überprüft ob das Koordinaten-Paar nicht bereits verwendet wurde
                seed_color = i*delta_color+color_range[0] #berechtnet den Grauton für den i.ten Kristal
                grow_speed = random.randint(grow_speed_range[0],grow_speed_range[1]) #Gibt eine zufällige Wachstumsgeschwindigkeit zwischen 1 und einem Wert an.
                delay = random.randint(0,maxdelay) #Gibt eine Zufällige Zahl für die Verzögerung des Kristals
                seed = Crystal(seed_pos,seed_color,grow_speed,delay) #und initalisiert diesen
                crystals_list.append(seed) #Dieser neue Kristal wird zur liste hinzugefügt
                seeds_list.append(seed_pos) #Und zum Schluss ebenfall das Koordinaten Paar
                break
    return crystals_list #Gibt die Liste mit den initalisierten Kristalen zurück


class Crystal: #Diese Klasse representiert jeden Kristal einzel und ermöglicht es uns einem einzelnen Kristal eine Wachstumsgeschwindigkeit zu geben
    def __init__(self,seed_pos,seed_color,grow_speed,delay):
        self.seed_color = seed_color #Gibt den Grau-ton des Pixels an
        self.grow_speed = grow_speed #Grow_speed ist die Wachstumsgeschwindigkeit
        self.delay = delay #Der delay gibt an um wie viele Durchläufe der Kristal verzögert ist
        self.open_pos = [seed_pos] #In der Liste open_pos, werden die Pixel gespeichert, welche noch bearbeitet werden müssen.
        self.counter = 0 #Der Counter zählt die Durchläufe
        self.complete = False #Diese Bool-Variable gibt an ob dieser Kristal fertig berechnet ist. Dies ist der Fall, wenn open_pos leer ist.



    def find_pixel(self,matrix,position): #Diese Funktion soll Pixel finden, welche noch von keinem anderen Kristal besetzt sind
        new_pos = [] #In diese Liste werden die gefunden Pixel gespeichert
        conditions =[(1,0),(-1,0),(0,1),(0,-1)] #gibt an welcher Wert auf welche Achse hinzu gerechnet wird (x,y)
        borders = [(0,720),(0,1280)] # Definiert die Grenzen des Bildes => [(x_min,x_max),(y_min,y_max)]


        for c in conditions:
            x = position[0] + c[0] #Durch das Auf addieren von der Condition für die x-Achse erhalten wir die x-Achse des neu zubetrachtenden Pixels
            y = position[1] + c[1] #Genau das gleiche nur für die y-Achse
            if x < borders[0][0] or x >= borders[0][1] or y < borders[1][0] or y >= borders[1][1]: #Dieses if_statement überprüft, ob der betrachtete Pixel außerhalb des Bildes liegt
                continue #Falls dies der Fall ist, so fährt man sofort mit der nächsten Condition fort
            i = matrix.item((x,y)) #Entnimmt den Wert des neu zubetrachtenden Pixels
            if i == 0: #Falls dieser Wert == 0 sein sollt, bedeutet das dass dieser Pixel noch leer/undefiniert ist
                new_pos.append((x,y))#Somit wird dieser neu gefundene Pixel zur liste hinzugefügt
        return new_pos #Am ende wird diese Liste noch zurückgegeben

    def grow_crystal(self,matrix): #Diese Funktion soll die Kristalle wachsen lassen
        if self.counter < self.delay: #Falls das Limit vom Delay noch nicht ereicht wurde,so
            self.counter += 1 #erhöht man den Zähler
            return matrix #Und beendet den Durchlauf
        expansions = len(self.open_pos) * self.grow_speed #Gibt an wie viele Pixel erweitert werden
        for _ in range(expansions):
            pixel = self.open_pos.pop(random.randint(0,len(self.open_pos)-1)) #Entnimmt einen zufälligen Pixel aus der Liste der noch zu bearbeitenden Pixel
            to_change = Crystal.find_pixel(self,matrix,pixel) #Die gefunden Pixel werden in der Liste to_change abgespeichert.

            for p in to_change: #iteriert durch die gefunden Pixel
                matrix.itemset(p,self.seed_color) #und gibt ihnen den Grauton, des Kristals
                self.open_pos.append(p) #Fügt diesen Pixel in die Liste der zu überprüfenden Pixel

            if len(self.open_pos) == 0: #Überprüft ob die Liste open_pos leer ist
                self.complete = True #Sollte dies der Fall sein, so ist dieser Kristal fertig berechnet und die Variable complete wird auf True gesetzt
                return matrix #beendet die Schleife und die Funktion
        return matrix #Gibt die Matrix mit den neu definierten Pixeln zurück



def running(crystals_list): #Diese Funktion ist die Hauptfunktion, welche die unterfunktionen aufruft und koordiniert
    matrix = np.full([720, 1280],0,dtype=np.uint8) #im ersten Schritt wird unser Matrix erstellt
    while True:
        matrix = crystals_list[0].grow_crystal(matrix) #Startet die Funktion welche den Kristal[0] wachsen lässt
        if crystals_list[0].complete == True: #Falls der betrachtete Kristal bereits fertig ist,
            crystals_list.pop(0) #So wird er aus der Liste entfernt
            if len(crystals_list) == 0: #Überprüft ob die Liste mit den Kristalen leer ist, sollte dies der Fall sein, so ist da Bild vollendet
                return matrix #gibt die fertig angefärbte Matrix wieder
            continue #Und man macht mit dem nächsten Kristal weiter
        else:
            crystals_list.append(crystals_list.pop(0)) #Zum schluss wird der Kristal an der Stelle[0] wieder ganz nach hinten in der Liste verschoben

different_settings = [(100,(1,3),0),(50,(3,12),5),(200,(1,2),3)]#die Settings sind wie folgt aufgebaut, (Anzahl an Kristallen, Zahlenbereicht für den Faktor welcher die Wachstumsgeschwindigkeit bestimmt => (untere Grenze,obere Grenze),Verzögerung)
for i,setting in enumerate(different_settings):
    num,grow_speed_range,max_delay = setting #Entpackt die Werte
    matrix = running(get_seeds(num,grow_speed_range,max_delay)) #Erstellt die Matrix
    print(f"Bild {i+1}: fertig")
    image = Image.fromarray(matrix) #wandelt die Matrix in ein Bild um
    image.show() #Zeigt das Bild an
    image.save(f"Aufgabe5_Bild{i+1}.jpg") #Speichert das Bild
