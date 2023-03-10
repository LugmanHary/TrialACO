import numpy as np
from numpy import inf
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Generate all solution
def findrouteallant(rute, pheromone, visibility):
    rute = np.zeros((m, n))
    temp_visibility = np.array(visibility)
    for i in range(m):
        rute = findroute(rute, i, temp_visibility, pheromone)
        #print(rute)
    return rute

# Generate step by step
def findroute(rute, ant_number, temp_visibility, pheromone):
    i = copy.copy(ant_number)
    p_feature = np.power(pheromone, alpha)
    v_feature = np.power(temp_visibility, beta)
    combine_feature = np.multiply(p_feature, v_feature)
    # print(combine_feature)
    for j in range(n-1):
        #print("Iterasi=",j)
        current_loc = int(rute[i, j])
        #print("Location="current_loc)
        combine_feature[:, current_loc] = 0
        #print(combine_feature)
        # adding axis
        locom_feature = combine_feature[current_loc, :]
        total = np.sum(locom_feature)
        probs = locom_feature/total
        cum_probs = np.cumsum(probs)
        r = np.random.random_sample()
        city = np.nonzero(cum_probs > r)[0][0]
        rute[i, j+1] = city
    rute.astype(int)
    return rute
np.power

# Evaluasi solution
def totaldistancetour(tour):
    jarak = 0
    tour = np.array(tour)
    tour = tour.astype(int)
    for i in range(0, len(tour)-1):
        jarak = jarak+d[tour[i]][tour[i+1]]
    jarak = jarak+d[tour[len(tour)-1]][tour[0]]
    return jarak

def evaluate(ants):
    ants = ants.tolist()
    ddf = pd.DataFrame(pd.Series(ants), columns=["sequence"])
    ddf['evaluation'] = ddf.apply(lambda x: totaldistancetour(x["sequence"]), axis=1)
    evaluation_list = list(ddf["evaluation"])
    return evaluation_list

# Update pheromone
def updatepheromone(pheromone, rute, evaluasi):
    pheromone = (1-e)*pheromone
    for i in range(m):
        for j in range(n-1):
            dt = 1/evaluasi[i]
            pheromone[int(rute[i, j]), int(rute[i, j+1])] = pheromone[int(rute[i, j]), int(rute[i, j+1])] + dt
            #print(pheromone)
    return pheromone

# Main program
def performancerecord(evaluasi, rute, best_performance, solution_array, performance_array, best_performance_array):
    current_performance = min(evaluasi)
    #current_performance = [current_performance]
    current_solution = rute[np.argmin(evaluasi)]
    if current_performance<best_performance:
        best_performance = copy.copy(current_performance)
    solution_array.append(current_solution)
    performance_array.append(current_performance)
    best_performance_array.append(best_performance)
    return solution_array, performance_array, best_performance_array, best_performance

# Plot performance
def PlotP(performance_array, best_performance_array):
    fig, ax = plt.subplots()
    ax.plot(performance_array)
    #ax.plot(best_performance_array)
    ax.set_title("Grafik Performance")
    ax.set_ylabel("Value")
    ax.set_xlabel("Iterasi")
    ax.legend(['Performance', 'Best Performance'])
    plt.show()
        
# Plot rute 
def PlotE(evaluasi):
    fig, ax = plt.subplots()
    rute_semuts = list(range(len(evaluasi)))
    semut_ke_n = ["Ant " + str(i) for i in rute_semuts]
    colors = []
    for i in range(len(evaluasi)):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    ax.bar(semut_ke_n, evaluasi, color=colors)
    for i, v in enumerate(evaluasi):
        ax.text(i, v+1, str(v), ha='center')
        #ax.text(rute_semuts[i], evaluasi[i]+6, str(evaluasi[i]), ha='center')
        #ax.text(rute_semuts[i], evaluasi[i]-11, semut_ke_n[i], ha='center')
        #ax.plot(rute_semuts[i], evaluasi[i], 'o', color='red')
    ax.set_title("Grafik Evaluasi Semut")
    ax.set_ylabel("Evaluasi")
    ax.set_xlabel("Posisi semut")
    ax.set_xticks(range(len(evaluasi)))
    plt.show()

# Plot evaluasi
def PlotV(evaluasi):
    mat_eva = [[0] * len(evaluasi) for i in range (iteration)]
    for i in range(iteration):
        for j in range(len(evaluasi)):
            mat_eva[i][j] = evaluasi[j]
    sux_eva = range(len(mat_eva))
    fig, ax = plt.subplots()
    for i in range(len(mat_eva[0])):
        suy_eva = [row[i] for row in mat_eva]
        ax.plot(sux_eva, suy_eva, label='Ant {}'.format(i))
    min_eva = np.min(mat_eva)
    max_eva = np.max(mat_eva)
    ax.set_ylim([min_eva-100, max_eva+100])
    ax.set_title("Grafik Evaluasi Semut")
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Evaluasi")
    ax.legend()
    plt.show()
    
# Plot pheromone
def PlotR(pheromone):
    #index_phe = np.indices((pheromone.shape[0], pheromone.shape[1]))
    out_phe = np.zeros_like(pheromone)
    for i in range(iteration):
        for j in range(pheromone.shape[0]):
            for k in range(pheromone.shape[1]):
                if pheromone[j][k] == np.max(pheromone[j]):
                    out_phe[j][k] = pheromone[j][k] - k
                elif pheromone[j][k] == np.min(pheromone[:,k]):
                    out_phe[j][k] = pheromone[j][k] + j
                else:
                    out_phe[j][k] = pheromone[j][k] * (i+1)
        #print(f"Iterasi pheromone ke-{i}:")
        #print(out_phe)
        out_phe = out_phe.copy()
        #print()
    for k in range(iteration):
        #print(f"Data baris dan kolom Iterasi ke-{k} :")
        #print()
        for i in range(len(pheromone)):
            for j in range(len(pheromone[0])):
                #print("baris", i, "& kolom", j, pheromone[i][j])
                pheromone[i][j]
            #print()
    fig, ax = plt.subplots()            
    x = np.arange(pheromone.shape[1])
    y = np.arange(pheromone.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = pheromone[Y,X]
    c = ax.pcolormesh(X, Y, Z, cmap='YlGnBu')
    fig.colorbar(c, ax=ax)
    ax.set_title('Grafik Pheromone')
    ax.set_xlabel('Kolom')
    ax.set_ylabel('Baris')
    plt.show()    

# Data informasi matriks
matriks = np.array([[0, 127, 210, 303, 400, 210, 268, 346, 433, 303, 346, 410, 486, 400, 433, 486, 551],
                    [127, 0, 100, 200, 300, 100, 141, 223, 316, 200, 223, 282, 360, 300, 316, 360, 424],
                    [210, 100, 0, 100, 200, 141, 100, 141, 223, 223, 200, 223, 282, 316, 300, 316, 360],
                    [303, 200, 100, 0, 100, 223, 141, 100, 141, 282, 223, 200, 223, 360, 316, 300, 316],
                    [400, 300, 200, 100, 0, 316, 223, 141, 100, 360, 282, 223, 200, 424, 360, 316, 300],
                    [210, 100, 141, 223, 316, 0, 100, 200, 300, 100, 141, 223, 316, 200, 223, 282, 360],
                    [268, 141, 100, 141, 223, 100, 0, 100, 200, 141, 100, 141, 223, 223, 200, 223, 282],
                    [346, 223, 141, 100, 141, 200, 100, 0, 100, 223, 141, 100, 141, 282, 223, 200, 223],
                    [433, 316, 223, 141, 100, 300, 200, 100, 0, 316, 223, 141, 100, 360, 282, 223, 200],
                    [303, 200, 223, 282, 360, 100, 141, 223, 316, 0, 100, 200, 300, 100, 141, 223, 316],
                    [346, 223, 200, 223, 282, 141, 100, 141, 223, 100, 0, 100, 200, 141, 100, 141, 223],
                    [410, 282, 223, 200, 223, 223, 141, 100, 141, 200, 100, 0, 100, 223, 141, 100, 141],
                    [486, 360, 282, 223, 200, 316, 223, 141, 100, 300, 200, 100, 0, 316, 223, 141, 100],
                    [400, 300, 316, 360, 424, 200, 223, 282, 360, 100, 141, 223, 316, 0, 100, 200, 300],
                    [433, 316, 300, 316, 360, 223, 200, 223, 282, 141, 100, 141, 223, 100, 0, 100, 200],
                    [486, 360, 316, 300, 316, 282, 223, 200, 223, 223, 141, 100, 141, 200, 100, 0, 100],
                    [551, 424, 360, 316, 300, 360, 282, 223, 200, 316, 223, 141, 100, 300, 200, 100, 0]])

print("Matriks awal:")
print(matriks)
print()

# Mengambil inputan index matriks terpilih dari user 
valid = False
while not valid:
    inh_0 = str(input("Masukan index ke-0: "))
    inh_1 = str(input("Masukan index ke-1: "))
    inh_2 = str(input("Masukan index ke-2: "))
    inh_3 = str(input("Masukan index ke-3: "))
    inh_4 = str(input("Masukan index ke-4: "))
    #print()
    if len(set([inh_0, inh_1, inh_2, inh_3, inh_4])) == 5:
        if '' not in [inh_0, inh_1, inh_2, inh_3, inh_4]:
            valid = True
        else:
            print("Error: Tidak boleh ada karakter kosong\n")
    else:
        print("Error: Tidak boleh ada karakter yang sama\n")
print()

# Buat dictionary untuk konversi str to num
dict1 = {'Z': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
     'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16} 

hf_0 = inh_0
ak_0 = [dict1[i] for i in hf_0]
hfak_0 = int(str(ak_0).strip('[]'))

hf_1 = inh_1
ak_1 = [dict1[i] for i in hf_1]
hfak_1 = int(str(ak_1).strip('[]'))

hf_2 = inh_2
ak_2 = [dict1[i] for i in hf_2]
hfak_2 = int(str(ak_2).strip('[]'))

hf_3 = inh_3
ak_3 = [dict1[i] for i in hf_3]
hfak_3 = int(str(ak_3).strip('[]'))

hf_4 = inh_4
ak_4 = [dict1[i] for i in hf_4]
hfak_4 = int(str(ak_4).strip('[]'))

out_dict1 = [hfak_0, hfak_1, hfak_2, hfak_3, hfak_4]

for i, str_num in enumerate(out_dict1):
    print(f"Data index ke-{i}: {str_num}")
print()

d = matriks[[hfak_0, hfak_1, hfak_2, hfak_3, hfak_4]][:, [hfak_0, hfak_1, hfak_2, hfak_3, hfak_4]]

print("Matriks terpilih:")
print(d)
print()

# Mengambil input untuk algoritma dari user
falid = False
while not falid:
    in_iter = input("Masukkan jumlah iterasi: ")
    in_ant = input("Masukkan jumlah semut: ")
    in_eva = input("Masukkan pengendali evaporasi: ")
    in_alp = input("Masukkan pengendali pheromone: ")
    in_bet = input("Masukkan pengendali visibilitas: ")
    
    if all(val != '' for val in [in_iter, in_ant, in_eva, in_alp, in_bet]):
        if all(val.replace('.', '').isnumeric() for val in [in_iter, in_ant, in_alp, in_bet]) and in_eva.replace('.', '').isdigit():
            in_iter = int(in_iter)
            in_ant = int(in_ant)
            in_eva = float(in_eva)
            in_alp = float(in_alp)
            in_bet = float(in_bet)
            falid = True
        else:
            print("Error: Terdapat angka yang ambigu\n")
    else:
        print("Error: Tidak boleh ada nilai kosong\n")
print()

iteration = in_iter
n_ants = in_ant
n_city = len(d)
m = n_ants
n = n_city
e = in_eva
alpha = in_alp
beta = in_bet

#rute = np.ones((m, n+1))
#pheromone = .1*np.ones((m,n))
warnings.filterwarnings("ignore", category=RuntimeWarning)
visibility = 1/d
visibility[visibility == inf] = 0
pheromone = .1*np.ones((n,n))
rute = np.zeros((m, n))
best_performance = inf
solution_array = []
performance_array = []
best_performance_array = []

for ite in range(iteration):
    rute = findrouteallant(rute, pheromone, visibility)
    evaluasi = evaluate(rute)
    pheromone = updatepheromone(pheromone, rute, evaluasi)
    solution_array, performance_array, best_performance_array, best_performance = performancerecord(evaluasi, rute, best_performance, solution_array, performance_array, best_performance_array)
    # Record performance solution
    #PlotP(performance_array, best_performance_array)
    # Record route evaluation
    #PlotE(evaluasi)
    #PlotV(evaluasi)
    # Record pheromone 
    #PlotR(pheromone)

# Menghilangkan tanda titik pada matriks rute
data_rute = rute.astype(int)
# Mencari data unique kemunculan setiap baris dalam data matriks
unique, counts = np.unique(data_rute, axis=0, return_counts=True) 
# Mencari rute data matriks dengan jumlah kemunculan terbanyak
most_frequent = unique[np.argmax(counts)]
max_count_index = np.argmax(counts)
max_frequent_rute = unique[max_count_index]
count = counts[max_count_index]
data_semut = [data_rute[i].copy() for i in range(data_rute.shape[0])]

# Mengeluarkan paket semut
#print("Data rute semut:")
#print(data_rute)
#print()

#print("Data pada setiap rute semut:")
#for i in range(len(data_rute)):
#   print("Rute jarak semut ke-{0} : {1} dan urutannya : {2}".format(i, evaluasi[i], data_rute[i]))
#print()

# Mencari rute keluar terbanyak
print("Data rute semut keluar terbanyak:")
print("Pada urutan:",max_frequent_rute, "keluar sebanyak", count, "-kali")
print()

# Mencari total jarak paling minimum
dist_min = evaluasi
dist_min_loc = min(dist_min)
index_min_loc = dist_min.index(dist_min_loc)
min_frequent_rute = data_rute[index_min_loc]

print("Rute jarak yang paling minimum:")
print(f"Rute jarak semut ke-{index_min_loc}:({dist_min_loc}) urutannya:{min_frequent_rute}")
print()

# Membuat rute final dari urutan angka dan huruf
print("Rute final:")
home = max_frequent_rute[0]
data_final = np.append(max_frequent_rute, home)
print(data_final)

out_dict = [data_final[0], data_final[1], data_final[2], data_final[3], data_final[4], data_final[5]]

dict2 = {0: 'Z', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H',
         9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P'}

akhf_0 = None
akhf_1 = None
akhf_2 = None
akhf_3 = None
akhf_4 = None
akhf_5 = None 

for i, str_num in enumerate(out_dict1):
    if out_dict[0] == i:
        akhf_0 = str_num
    if out_dict[1] == i:
        akhf_1 = str_num
    if out_dict[2] == i:
        akhf_2 = str_num
    if out_dict[3] == i:
        akhf_3 = str_num
    if out_dict[4] == i:
        akhf_4 = str_num
    if out_dict[5] == i:
        akhf_5 = str_num 

out_dict2 = [akhf_0, akhf_1, akhf_2, akhf_3, akhf_4, akhf_5]
out_dict_new = []

for i in out_dict2:
    out_dict_new.append(dict2[i])
print(out_dict_new)
print()

print('Data rute final')
for i, num_str in enumerate(out_dict):
    print(f"Data rute ke-{i}: {num_str}")
print()

for i, num_str in enumerate(out_dict_new):
    print(f"Data rute ke-{i}: {num_str}")
    
# Mengeluarkan rutenya ke varibel baru (nanti dikirim ke arduino)
send_a0 = data_final[0]
send_a1 = data_final[1]
send_a2 = data_final[2]
send_a3 = data_final[3]
send_a4 = data_final[4]
send_a5 = data_final[5]

send_h0 = out_dict_new[0]
send_h1 = out_dict_new[1]
send_h2 = out_dict_new[2]
send_h3 = out_dict_new[3]
send_h4 = out_dict_new[4]
send_h5 = out_dict_new[5]