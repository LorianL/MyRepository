

def create_dic():
    dic = {}
    for i in [33,38,39,40,41,42,44,45,46,48,49,50,51,52,53,54,55,56,57,58,59,61,63,91,93,95,171,187,65279]:
        dic[i] = None #Als ersatz setzen wir None ein, bedeutet dass wir diesen Ord durch nichts ersetzten und somit löschen
    return dic


dic = create_dic() #Erstellt ein Dic mit allen Ords, welche entfernt werden sollen
text = []
with open("Alice_im_Wunderland.txt",encoding="utf-8") as f: #Öffnet den Text mit encoding="utf-8"
    lines = f.readlines()
    for l in lines: #Diese for schleife liest alle Linien des Textes ein
        l = l.strip().lower() #Entfernt "\n" und macht alle Buchstaben zu lower Cases
        l = l.translate(dic) #Entfernt alle unnötige zeichen, welche keine Buchstaben oder Zahlen sind mithilfe der translate Funktion
        l = l.split() #Spaltet den String in eine Liste aus den einzelnen Wörtern auf
        text.extend(l) #Fügt die Wörter zum rest des Textes hinzu


def get_stoerung(name): #Diese Funktion soll aus der Datei name, die Störung rauslesen
    with open(name,encoding="utf-8") as f:#Dazu liest es das Dokument ein
        line = f.readlines()
        line = line[0].strip() #Und entnimmt nur die erste Zeile und entfernt das "\n"
        line = line.split() #am ende Spaltet es den String noch in seine einzelnen Wörter/Zeichen auf und gibt eine Liste zurück
        return line

def find_words(stoerung): #Diese Funktion soll alle bekannten Wörter in der Störung extrahieren
    list = [] #In dieser Liste sollen die Indexe gepseichert werden auf welchen sich ein Wort befindet
    for i,word in enumerate(stoerung): #Iteriert durch die Störung
        if word != "_": #Falls es sich bei dem Wort nicht um ein gesuchtes Wort handelt, dann
            list.append((i,word)) #fügt man das Wort mitsamt ihrer Position in der Störung zur Liste hinzu
    return list


def check_stoerung(stoerung_list,text,index):
    bool_list = [] #Hier soll ein True/False Wert für jedes bekannte Wort aus der Störung eingefügt werden. True bedeutet dass das Wort aus der Störung mit der Textstelle übereinstimmt, False bedeutet nicht
    for s in stoerung_list: #iteriert durch die bereits bekannten Wörter
        bool_list.append(text[index+s[0]-stoerung_list[0][0]] == s[1]) #index+s[0] Dieser Ausdruck überprüft ob das nächste Wort nach dem ersten Treffer auch übereinstimmt. "-stoerung_list[0][0]" Dieser Ausdruck dient dazu falls das erste Wort bei der Störung eine Unbekannte ist. PS: Im Normalfall ist dieser ausdruck = -0
    return bool_list

def find_stoerung(stoerung):
    stoerung_list = find_words(stoerung)
    possible_results = []
    for i,word in enumerate(text):
        if stoerung_list[0][1] == word and all(check_stoerung(stoerung_list,text,i)): #Überprüft ob das erste Wort der Störung mit der aktuellen Textstelle übereinstimmt. Falls dies Zutrifft, so kontrolliert die Funktion check_stoerung() ob ebenfalls der Rest der Störung mit der aktuellen Textstellen übereinstimmt
            #Falls alle bools der Funktion check_stoerung True sind, so stimmt die Störung mit der Aktuellen Textstelle überein
            possibility = text[i-stoerung_list[0][0]:i+len(stoerung)-stoerung_list[0][0]] #Dann wird die Textstelle als possibility gespeichert. Hier dient die Stelle "-stoerung_list[0][0]" wieder für den Sonderfall, dass die Störung mit einem Unbekannten Wort anfangen sollte
            if possibility not in possible_results: #überprüft ob es keine Duplikate gibt
                possible_results.append(possibility) #fügt die mögliche Lösung zur Liste hinzu
    output(possible_results)

def output(result): #Printet die Gefunde Stelle
    for i,r in enumerate(result):
        string = " ".join(r)
        print(f"{i+1}.Stelle: {string}")

if __name__ == '__main__':
    filenames = ["stoerung0.txt","stoerung1.txt","stoerung2.txt","stoerung3.txt","stoerung4.txt","stoerung5.txt","Sonderfall.txt"]#"stoerung0.txt","stoerung1.txt","stoerung2.txt","stoerung3.txt","stoerung4.txt","stoerung5.txt"
    for name in filenames:
        print(f"\n----------------{name}----------------\n")
        stoerung = get_stoerung(name)
        print(f"Störung: {stoerung}\n")
        find_stoerung(stoerung)
