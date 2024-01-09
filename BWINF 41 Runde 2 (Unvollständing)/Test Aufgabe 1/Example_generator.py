import numpy as np
import random
from PIL import Image


class Crystal:
    def __init__(self,seed_pos,seed_color):
        self.seed_pos = seed_pos
        self.seed_color = seed_color
        self.open_pos = [seed_pos]
        self.complete = False
        self.grow_speed = random.randint(1,5)



    def find_pixel(self,matrix,position):
        new_pos = []
        conditions =[(1,0),(-1,0),(0,1),(0,-1)] #gibt an welcher Wert auf welche Achse hinzu gerechnet wird (x,y)
        borders = [(0,720),(0,1280)] # Definiert die Grenzen des Bildes => [(x_min,x_max),(y_min,y_max)]


        for c in conditions:
            x = position[0] + c[0] #Gibt die neue x_Achsen Position an
            y = position[1] + c[1] #Gibt die neue y_Achsen Position an
            if x < borders[0][0] or x >= borders[0][1] or y < borders[1][0] or y >= borders[1][1]: #Überpfüft ob der nächste Pixel außerhalb des Bildes wäre
                continue #Falls dies zu trifft so macht man mit dem nächste möglichen Pixel weiter
            i = matrix.item((x,y)) #Entnimmt den Wert auf welchem sich der nächste Pixel befinden soll
            if i == 0:
                new_pos.append((x,y))
        return new_pos

    def grow_crystal(self,matrix):
        new_pixels = []
        for i in self.open_pos:
            to_change = Crystal.find_pixel(self,matrix,i)

            for p in to_change:
                matrix.itemset(p,self.seed_color)
                new_pixels.append(p)

        self.open_pos = new_pixels #die alte Liste mit inzwischen überprüften Pixeln wird durch die neue ersetzt
        if len(new_pixels) == 0:
            self.complete = True

        return matrix

    def control_grow(self,matrix):
        for _ in range(self.grow_speed):
            matrix = Crystal.grow_crystal(self,matrix)
        return matrix




points = 10


def get_seeds(num=points,resolution=(720,1280),color_range=(100,200)):
    crystals_list = []
    seeds_list = []
    delta_color = (color_range[1]-color_range[0])/num
    for i in range(num):
        while True:
            seed_pos = (random.randint(0,resolution[0]-1),random.randint(0,resolution[1]-1))
            if seed_pos not in seeds_list:
                seed_color = i*delta_color+color_range[0]
                seed = Crystal(seed_pos,seed_color)
                crystals_list.append(seed)
                break
    return crystals_list



def running(crystals_list):
    matrix = np.full([720, 1280],0,dtype=np.uint8)
    counter = 0
    while True:
        counter += 1
        if len(crystals_list) == 0 or counter == points*3:
            print(counter)
            return matrix
        if crystals_list[0].complete == True:
            crystals_list.pop(0)
            continue
        matrix = crystals_list[0].control_grow(matrix)
        crystals_list.append(crystals_list[0])
        crystals_list.pop(0)


matrix = running(get_seeds())

image = Image.fromarray(matrix) # mode="RGB"
image.show()
image.save(f"Example_{points}Points.jpg")
