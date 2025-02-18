import os
import glob
import numpy as np
import face_recognition
import cv2
import matplotlib.pyplot as plt

import numpy as np


def delete_outliers(model):

    person_names = glob.glob("extra/*")
    faces = []
    known_person_info = []
    print(person_names)


    name_list = []
    for name in person_names:


        only_name = name.split("/")[-1]

        name_list.append(only_name)
        known_encoding_dict = {
            'name': only_name,
            'encoding':[],
            'av_loss':[],
            'files':[]
            }
        for root, dirs, files in os.walk(name):
            temp_images = []
            temp_encoding = []
            for index, file in enumerate(files):
                img = cv2.imread(str(name+'/'+file))
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    print("Skipping ", name+"/"+file)
                temp_images.append(img)
                try:
                   
                    face_locations_m, points = model.predict(img)
                    face_locations = []
                    for bb in face_locations_m[0]:
                        face_locations.append((bb[1], bb[2], bb[3], bb[0]))
                      
                    encod = face_recognition.face_encodings(img, face_locations)[0]
                    
                except:
                    print("No face found", name+"/"+file)
                    
                    continue
                known_encoding_dict['encoding'].append(encod)
                known_encoding_dict['files'].append(file)


            faces.append(temp_images)
            known_person_info.append(known_encoding_dict)
    print("encoding done successfully for "+str(len(known_person_info))+" people" )

   
    l = []
    for info in known_person_info:

        for i in range (0,len(info['encoding'])) :
            dist = 0
            dists =[]
            for j in range(0,len(info['encoding'])):
                if i !=j:
                    target = info['encoding'][i].reshape((1,-1))
                    target_to_compare = info['encoding'][j].reshape((1,-1))
                    dist = face_recognition.face_distance(target, target_to_compare)[0]
                    
                    dists.append(dist)

            try:
                if j != 0:
                    

                    info['av_loss'].append(np.mean(dists))

            except:
                print('1 image can not be compared')


    dist_sorted = sorted(dists, reverse = True)
    


    for i in range(len(person_names)):

        fig, ax = plt.subplots(figsize = (18,10))
        ax.scatter(known_person_info[i]['files'], known_person_info[i]['av_loss'])

        
        ax.set_xlabel(person_names[i])

     
        ax.set_ylabel('average distance')
        plt.savefig('mean_'+known_person_info[i]["name"]+'.png')


    mu = np.mean(dist_sorted)

    for i, name in enumerate(person_names):
            #del_index = np.argmax(known_person_info[i]['av_loss'])
            #val = known_person_info[i]["av_loss"][del_index]
            if len(dist_sorted) < 20:
                continue
            else:
                c = 1
                for val in known_person_info[i]['av_loss']:
                # for val in dist_sorted:
                    #print("val", val)
                    if val > mu:
                        c += 1
                        del_index = np.where(known_person_info[i]['av_loss'] == val)

                        delete_ = del_index[0][0]
                        #print("del_index", del_index[0][0])
                        os.remove(name + '/'+ known_person_info[i]['files'][delete_])
                        print("deleted "+str(name + '/'+ known_person_info[i]['files'][delete_]))
                        if c == 30:
                            break
                    else:
                        pass


