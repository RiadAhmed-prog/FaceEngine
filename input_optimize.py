import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

import numpy as np

from deepface.commons import functions, realtime, distance as dst




person_names = glob.glob("test/*")

print(person_names)
with open('encodings.pickle', 'rb') as handle:
    known_encoding_dict = pickle.load(handle)

with open('known_person_info.pickle', 'rb') as handle:
    known_person_info = pickle.load(handle)

print("encoding successfully done for " + str(len(known_person_info)) + " people")


l = []
for info in known_person_info:
    
    for i in range (0,len(info['encodings'])) :
        dist = 0
        dists =[]
        for j in range(0,len(info['encodings'])):
            if i !=j:
                # target = info['encodings'][i].reshape((1,-1))
                # target_to_compare = info['encoding'][j].reshape((1,-1))
                target = info['encodings'][i]
                target_to_compare = info['encodings'][j]
                # dist = face_recognition.face_distance(target, target_to_compare)[0]
                try:
                    distance = dst.findCosineDistance(target, target_to_compare)
                except Exception as e:
                    print("Exception in deepface cosine function: ", e)
                # dist = dst.findCosineDistance(target, target_to_compare)[0]
                dists.append(distance)

        try:
            if j != 0:
                # print(np.median(dists))
                
                info['av_loss'].append(np.mean(dists))
           
        except:
            # print('1 image can not be compared')
            pass


dist_sorted = sorted(dists, reverse = True)

for i in range(len(person_names)):
    
    fig, ax = plt.subplots(figsize = (18,10))
    ax.scatter(i, known_person_info[i]['av_loss'])
    
    # x-axis label
    ax.set_xlabel(person_names[i])
    
    # y-axis label
    ax.set_ylabel('average distance')
    plt.savefig('mean_'+known_person_info[i]["name"]+'.png')
    # plt.show()

mu = np.mean(dist_sorted)
mu = mu - 0.1
for i, name in enumerate(person_names):
            print("i {} name {}", i, name)
        #del_index = np.argmax(known_person_info[i]['av_loss'])
        #val = known_person_info[i]["av_loss"][del_index]
        # if len(dist_sorted) < 5:
        #     continue
        # else:
            # c = 1
            val  = known_person_info[i]['av_loss']
            print("val", val)
            print("mu", mu)
            
            if val > mu:
                print("hello")
                # c += 1
                del_index = np.where(known_person_info[i]['av_loss'] == val)

                delete_ = del_index[0][0]
                #print("del_index", del_index[0][0])
                os.remove(name + '/'+ known_person_info[i]['files'][delete_])
                print("deleted "+str(name + '/'+ known_person_info[i]['files'][delete_]))
                # if c == 6:
                #     break
            else:
                pass

