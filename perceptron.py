import numpy as np

def nothing():
    return ''

#fungsi untuk membaca data latih dan memasukkan nilainya ke dalam array
def readData(txt, array):
    f = open(txt, 'r')
    next = f.read(1)
    i = 0 
    while next != "":
        if (next == '.'):
            array[i] = -1
            i = i+1
        elif (next == '#'):
            array[i] = 1
            i = i+1
        else:
            nothing()
        next = f.read(1)
    f.close()

#fungsi aktivasi, threshold = 0
def activation(v):
    if v >= 0 :
        return 1
    else:
        return -1

#fungsi update bobot
def update_weight(old_w, lr, t, x_input):
    new_w = old_w + (lr * t * x_input)
    return new_w

#match
def match(a, b): 
    index = 0
    for i in range(len(a)):
            if (a[i] != b[i]):
                index +=1
    return index

#fungsi training
def training(array_input, array_target, array_w, b, learning_rate):
    target_update = np.zeros(shape=(7), dtype= np.integer)
    epoch_status = False
    while epoch_status == False:
        for i in range(7): # 7 neuron
            v = np.dot(array_input, array_w[:, i]) + b[i] #xi * wi
            y = activation(v)
            target_update[i] = y
            #print ("Neuron =", i+1)
            #print ("  b  | v  |  y |  t")
            #print (b[i], " | ", v," | ", y," | ", array_target[i])
            if (y != array_target[i]):
                #print ("tidak sesuai")
                for j in range(63):
                    array_w[j][i] = update_weight(array_w[j][i], learning_rate, array_target[i], array_input[j])
                    b[i] = b[i] + (learning_rate * array_target[i])
        if (match(array_target, target_update) != 0):
            epoch_status = False
        else:
            epoch_status = True
    return array_w, b
            
def testing(array_input, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K):
    target_update = np.zeros(shape=(7), dtype= np.integer)
    for i in range(7): # 7 neuron
        v = np.dot(array_input, array_w[:, i]) + b[i]
        y = activation(v)
        target_update[i] = y
        #print ("Neuron =", i+1)
        #print ("  b  | v  |  y")
        #print (b[i], " | ", v," | ", y)
        
    #print("target test ", target_update)
    if (match(target_A, target_update) == 0):
        return ("result : A")
    elif (match(target_B, target_update) == 0):
        return ("result : B")
    elif (match(target_C, target_update) == 0):
        return ("result : C")
    elif (match(target_D, target_update) == 0):
        return ("result : D")
    elif (match(target_E, target_update) == 0):
        return ("result : E")
    elif (match(target_J, target_update) == 0):
        return ("result : J")
    elif (match(target_K, target_update) == 0):
        return ("result : K")
    else:
        return ("no matches")
            
        
#inisialisasi file data latih, array buat menyimpan data latih, membuat array kosong inisiali awal
train_file = ['train/A.txt','train/B.txt','train/C.txt', 'train/D.txt','train/E.txt', 'train/J.txt', 'train/K.txt']
arr_A = np.zeros(shape=(63), dtype=np.integer)
arr_B = np.zeros(shape=(63), dtype=np.integer)
arr_C = np.zeros(shape=(63), dtype=np.integer)
arr_D = np.zeros(shape=(63), dtype=np.integer)
arr_E = np.zeros(shape=(63), dtype=np.integer)
arr_J = np.zeros(shape=(63), dtype=np.integer)
arr_K = np.zeros(shape=(63), dtype=np.integer)
train_data = [arr_A, arr_B, arr_C, arr_D, arr_E, arr_J, arr_K]
   
#inisialisasi target
target_A = np.array([1, -1, -1, -1, -1, -1, -1])
target_B = np.array([-1, 1, -1, -1, -1, -1, -1])
target_C = np.array([-1, -1, 1, -1, -1, -1, -1])
target_D = np.array([-1, -1, -1, 1, -1, -1, -1])
target_E = np.array([-1, -1, -1, -1, 1, -1, -1])
target_J = np.array([-1, -1, -1, -1, -1, 1, -1])
target_K = np.array([-1, -1, -1, -1, -1, -1, 1])
    
#input nilai data latih ke dalam array
for i in range(len(train_data)):
    readData(train_file[i], train_data[i])
    
array_w = np.zeros(shape=(63,7))
b = np.zeros(shape=(7))
learning_rate = 1

#training
array_w, b = training(arr_A, target_A, array_w, b, learning_rate)
array_w, b = training(arr_B, target_B, array_w, b, learning_rate)
array_w, b = training(arr_C, target_C, array_w, b, learning_rate)
array_w, b = training(arr_D, target_D, array_w, b, learning_rate)
array_w, b = training(arr_E, target_E, array_w, b, learning_rate)
array_w, b = training(arr_J, target_J, array_w, b, learning_rate)
array_w, b = training(arr_K, target_K, array_w, b, learning_rate)
#print (array_w)
#print(b)

#input testing
test_file = ['test/A.txt','test/B.txt','test/C.txt', 'test/D.txt','test/E.txt', 'test/J.txt', 'test/K.txt', 'test/A1.txt','test/B1.txt','test/C1.txt', 'test/D1.txt','test/E1.txt', 'test/J1.txt', 'test/K1.txt','test/K2.txt' ]
test_A = np.zeros(shape=(63), dtype=np.integer)
test_B = np.zeros(shape=(63), dtype=np.integer)
test_C = np.zeros(shape=(63), dtype=np.integer)
test_D = np.zeros(shape=(63), dtype=np.integer)
test_E = np.zeros(shape=(63), dtype=np.integer)
test_J = np.zeros(shape=(63), dtype=np.integer)
test_K = np.zeros(shape=(63), dtype=np.integer)
test_A1 = np.zeros(shape=(63), dtype=np.integer)
test_B1 = np.zeros(shape=(63), dtype=np.integer)
test_C1 = np.zeros(shape=(63), dtype=np.integer)
test_D1 = np.zeros(shape=(63), dtype=np.integer)
test_E1 = np.zeros(shape=(63), dtype=np.integer)
test_J1 = np.zeros(shape=(63), dtype=np.integer)
test_K1 = np.zeros(shape=(63), dtype=np.integer)
test_K2 = np.zeros(shape=(63), dtype=np.integer)
test_data = [test_A, test_B, test_C, test_D, test_E, test_J, test_K, test_A1, test_B1, test_C1, test_D1, test_E1, test_J1, test_K1, test_K2]

#input dokumen testing ke dalam array
for i in range(len(test_data)):
    readData(test_file[i], test_data[i])

#testing    
print("Dokumen A,",testing(test_A, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen B,",testing(test_B, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen C,",testing(test_C, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen D,",testing(test_D, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen E,",testing(test_E, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen J,",testing(test_J, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen K,",testing(test_K, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen A1,",testing(test_A1, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen B1,",testing(test_B1, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen C1,",testing(test_C1, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen D1,",testing(test_D1, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen E1,",testing(test_E1, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen J1,",testing(test_J1, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen K1,",testing(test_K1, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
print("Dokumen K2,",testing(test_K2, array_w, b, learning_rate, target_A, target_B, target_C, target_D, target_E, target_J, target_K))
