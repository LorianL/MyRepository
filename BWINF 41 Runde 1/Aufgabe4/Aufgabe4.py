import copy
#sort keys---------------
def sort_duration(order):
    return order.duration

def sort_input_time(order):
    return order.input_time
#--------------------------------
class Order(): #Diese Klasse soll die Eingenschaften der Aufträge speichern
    def __init__(self,input_time,duration):
        self.input_time = input_time
        self.duration = duration
        self.order_start = None

    def waiting_time(self): #Diese Funktion soll die Wartezeit zurückgeben
        waiting_time = self.order_start - self.input_time #Die Wartezeit ist Definiert als die differenz zwischen der Eingabezeit und dem Auftragsbeginn
        return waiting_time

    def prt_information(self):
        print(f"Eingangszeitpunkt: {self.input_time}\t Dauer: {self.duration}\t Arbeitsbeginn: {self.order_start}, Wartezeit: {Order.wartezeit(self)}")
        pass

#-----------------------------------------------------
class Template:
    def __init__(self,order_list):
        self.order_list = copy.deepcopy(order_list)
        self.incoming_orders = []
        self.finished_orders = []
        self.day_counter = 0 #Diese Variable soll die Arbeitstage zählen
        self.working_day = [] #in working_day sollen die Uhrzeiten der Arbeitstage in Form von (day_start,day_end) hinzugefügt werden
        Template.init_clock(self)

    def next_day(self):
        day_start = 540 + 1440*self.day_counter #Berechnet den Anfang des nächsten Tages
        day_end = 960 + 1440*self.day_counter #und das Ende
        self.working_day.append((day_start,day_end)) #Fügt den neuen Arbeitstag in die Liste
        self.day_counter += 1 #Erhöht den Tageszähler um 1


    def init_clock(self):
        self.order_list.sort(key=sort_input_time) #Sortiert die Liste, so dass die Aufträge welche die kleinste input_time haben vorne stehen
        first_incoming_order = self.order_list.pop(0) #Entnimmt den ersten Auftrag aus der Liste
        Template.next_day(self)
        Template.adjust_clock(self)
        Template.check_incoming_orders(self) #Im letzten Schritt überprüft man (in Fall 1) ob während man auf den Arbeitsbeginn wartet keine neuen Aufträge reingekommen sind.(Fall 2): ob zum first_incoming_order nicht noch parallel eine zweite eingegangen ist



    def adjust_clock(self):
        next_possible_order = self.order_list.pop(0)
        while True: #Die while_Schlife dient dazu, im Falle dass der nächste Auftrag erst nach dem Arbeitstag eingeht, dieser Prozess so lange wiederholt wird bis der passende Arbeitstag gefunden wurde PS: dient zum sonderfall falls mehrere Tage keine neuen Aufträge rein kommen
            if next_possible_order.input_time < self.working_day[-1][0]: # "self.working_day[-1][0]" Dieser Ausdruck schaut auf den Arbeitstag beginn des neusten Tages
            #Die if-Schleife überprüft also, ob der Auftrag vor Arbeitsbeginn reingekommen ist.
                self.incoming_orders.append(next_possible_order) #Falls dies zutriff, so wird der nächste Auftrag in die Wartelist gesetzt
                self.clock = self.working_day[-1][0] #Und die Uhr wird auf den Arbeitstagbeginn gesetzt
                break
            elif next_possible_order.input_time >= self.working_day[-1][0] and next_possible_order.input_time < self.working_day[-1][1]: #"self.working_day[-1][1]" Dieser Ausdruck schaut auf das Arbeitstagsende des neusten Tages
            #Diese if-Schleife überprüft ob der nächste Auftrag während einem Arbeitstag reingekommen ist.
                self.incoming_orders.append(next_possible_order) #Falls dies zutriff, so wird der nächste Auftrag in die Wartelist gesetzt
                self.clock = next_possible_order.input_time #Ebenfalls wird die Uhr auf die input_time des Auftrages nachvorne gedreht.
                break
            Template.next_day(self)

    def check_incoming_orders(self):
        while True:
            if len(self.order_list) != 0 and self.order_list[0].input_time <= self.clock: #Überprüft als erstes ob noch ein Element in order_list ist und als zweites ob das Element[0] eine Input_time hat welche kleiner ist als die aktuelle Zeit.
                new_order = self.order_list.pop(0)  #Falls dies zutrifft so wird Element[0] der order_list entnommen
                self.incoming_orders.append(new_order) #und zur incoming_orders list hinzugefügt
            else: #Falls die obrige bedingung nicht zutrifft so wird die Schleife beednet
                break
        if len(self.incoming_orders) == 0: #Als letztes wird überprüft ob in incoming_orders kein Auftrag ist
            Template.adjust_clock(self) #Falls dies Zutrifft, so wird die Uhr so angepasst, dass es wieder ein Auftrag gibt, denn es muss immer mindestens 1 Auftrag zur verfügung stehen

    def update_clock(self,order):#Diese Funktion soll die Uhr updaten und gegebenden falls einen neuen Tag einleiten
        self.clock += order.duration
        if self.clock > self.working_day[-1][1]: #Falls der Auftrag über die Arbeitszeit hinaus geht, so müssen die überzogenen Minuten dem nächsten Arbeitstag zu gerechenet werden
            delta_duration = self.clock - self.working_day[-1][1] #Nimm die differenz aus der Zeit an welchem der Auftrag beendet wäre und dem Arbeitsende
            Template.next_day(self)
            self.clock = self.working_day[-1][0] + delta_duration #setzt die Uhr auf den Arbeitsbeginn des nächsten Tages und addiert noch die Überzogenen Minuten hinzu, so dass der Auftrag vollendet ist ohne überstunden


class FirstIn_FirstOut(Template):#Diese Klasse simuliert das Verfahren, bei welchem die Aufträge der Reihenfolge nach abgearbeitet werden
    def __init__(self,order_list):
        super().__init__(order_list) #Erbt alles von der Template Klasse
        FirstIn_FirstOut.create_workplan(self) #und startet den Algorithmus

    def create_workplan(self):
        while True:
            FirstIn_FirstOut.check_incoming_orders(self) #Überpfüft on keine neuen Aufträge eingekommen sind
            self.incoming_orders.sort(key=sort_input_time) #Sortiert die Aufträge nach ihrer Dauer
            target_order = self.incoming_orders.pop(0) #Entnimmt den Auftrag mit der geringsten input_time
            target_order.order_start = self.clock #setzt fest wann der Auftrag begonnen wurde
            self.finished_orders.append(target_order)
            Template.update_clock(self,target_order)
            if len(self.order_list) == 0 and len(self.incoming_orders) == 0: #Falls sowohl order_list als auch incoming_orders leer ist, so ist der Algorithmus fertig
                break

class Shortest_First(Template):
    def __init__(self,order_list):
        super().__init__(order_list)
        Shortest_First.create_workplan(self)

    def create_workplan(self):
        while True:
            Shortest_First.check_incoming_orders(self) #Überpfüft on keine neuen Aufträge eingekommen sind
            self.incoming_orders.sort(key=sort_duration) #Sortiert die Aufträge nach ihrer Dauer
            #print(self.clock-list(map(sort_input_time,self.incoming_orders))[0])

            target_order = self.incoming_orders.pop(0) #Entnimmt den Auftrag mit der geringsten input_time
            self.finished_orders.append(target_order)
            target_order.order_start = self.clock
            Template.update_clock(self,target_order)
            if len(self.order_list) == 0 and len(self.incoming_orders) == 0:
                break
class Compromis(Template): #Dies ist mein Lösungsansatz. Es ist ein Kompromis aus den beiden Vorherigen
    def __init__(self,order_list):
        super().__init__(order_list) #Erbt alles von der Template Klasse
        Compromis.create_workplan(self) #und startet den Algorithmus

    def create_workplan(self):
        counter = 0
        while True:
            Compromis.check_incoming_orders(self) #Überpfüft on keine neuen Aufträge eingekommen sind
            sort_keys = {0:sort_input_time,1:sort_duration}
            self.incoming_orders.sort(key=sort_keys[counter%2]) #Sortiert die Aufträge nach ihrer Dauer
            target_order = self.incoming_orders.pop(0) #Entnimmt den Auftrag mit der geringsten input_time
            target_order.order_start = self.clock #setzt fest wann der Auftrag begonnen wurde
            self.finished_orders.append(target_order)
            Template.update_clock(self,target_order)
            if len(self.order_list) == 0 and len(self.incoming_orders) == 0: #Falls sowohl order_list als auch incoming_orders leer ist, so ist der Algorithmus fertig
                break
            counter += 1


def import_orders(file_name):
    with open(file_name) as f:
        lines_list = f.readlines()
    order_list = []
    for i,l in enumerate(lines_list):
        l = l.strip()
        l = l.split()
        if len(l) == 0:
            continue
        order_list.append(Order(int(l[0]), int(l[1])))
    return order_list

def print_statisics(work_plan,procedure_name,file_name): #Diese Funktion soll sowohl die durschnittliche als auch die maximale Wartezeit ausgeben
    total_waiting_time = 0 #Gibt die gesamte Wartezeit aller Aufträge an. Wenn man dies durch die Anzahl an aufträgen teilt erhält man die durchschnittliche Wartezeit
    waiting_time_list = [] #in diese Liste sollen die einzelnen Wartezeiten eingetragen werden um später die größte entnehmen zu können
    for order in work_plan:
        wait = order.waiting_time() #In der Variable wait wird die Wartezeit von der order gespeichert
        total_waiting_time += wait
        waiting_time_list.append(wait)

    average_waiting_time = total_waiting_time / len(waiting_time_list) #Um die durschnittliche Wartezeit zu erhalten teilen wir die gesamte Wartezeit durch die Anzahl an Aufträgen
    maximum_waiting_time = sorted(waiting_time_list)[-1] #Sortiert im erstem Schritt die Liste von klein nach groß und im zweiten schritt entnimmt es die letzte Zahl was somit die längste wartezeit ist

    print(f"{procedure_name}:\tDurchschnittliche Wartezeit: {int(average_waiting_time)}\tMaximale Wartezeit: {maximum_waiting_time}")


if __name__ == '__main__':
    file_names = ["fahrradwerkstatt0.txt","fahrradwerkstatt1.txt","fahrradwerkstatt2.txt","fahrradwerkstatt3.txt","fahrradwerkstatt4.txt"]#"fahrradwerkstatt0.txt","fahrradwerkstatt1.txt","fahrradwerkstatt2.txt","fahrradwerkstatt3.txt","fahrradwerkstatt4.txt"
    for filename in file_names:
        orders = import_orders(filename)
        print(f"\n--------------------{filename}---------------------------\n")
        altes_verfahren = FirstIn_FirstOut(orders).finished_orders
        neues_verfahren = Shortest_First(orders).finished_orders
        mein_verfahren = Compromis(orders).finished_orders
        print_statisics(altes_verfahren,"Altes Verfahren",filename)
        print_statisics(neues_verfahren,"Neues Verfahren",filename)
        print_statisics(mein_verfahren,"Mein  Verfahren",filename)
