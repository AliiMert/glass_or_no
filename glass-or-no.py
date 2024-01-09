#There are 150 with glass photos and 150 without glass photos. 
#The machine was trained by dividing it into 5 parts with k fold cross validation.
#There are 3 different algorithm and accurarcy scores.
import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#Fotoğrafların bulunduğu klasörlerin yolları
glass_path = 'C:\\Users\\alime\\Desktop\\Yapay Zeka Proje\\glass_or_no\\glass'
no_glass_path = 'C:\\Users\\alime\\Desktop\\Yapay Zeka Proje\\glass_or_no\\no_glass'

#Fotoğrafları yeniden boyutlandırarak çeken fonksiyon
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(cv2.resize(img, (350, 350)))
            labels.append(label)
    return images, labels

#Tüm fotoğrafları etiketliyor
glass_images, glass_labels = load_images_from_folder(glass_path, 0)
no_glass_images, no_glass_labels = load_images_from_folder(no_glass_path, 1)

#Tüm fotoğrafları birleştir
all_images = np.array(glass_images + no_glass_images)
all_labels = np.array(glass_labels + no_glass_labels)

#StratifiedKFold ile 5 parçaya bölme
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=60)

accuracy_scores_svc = []  #SVC için doğruluk skorları listesi

for train_index, test_index in skf.split(all_images, all_labels):
    X_train, X_test = all_images[train_index], all_images[test_index]
    y_train, y_test = all_labels[train_index], all_labels[test_index]

    #SVC algoritması ile makine öğrenmesi modeli ve eğitim
    model = SVC(kernel='linear')
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    
    #Test aşamasının doğruluklarına göre başarı değeri hesaplama
    y_test_pred = model.predict(X_test.reshape(len(X_test), -1))
    accuracy = accuracy_score(y_test, y_test_pred)
    accuracy_scores_svc.append(accuracy)

print("SVC ALGORITHM")
#her splitteki doğruluk değerlerini yazdırır    
print("1. Test Accuarcy: ",accuracy_scores_svc[0])
print("2. Test Accuarcy: ",accuracy_scores_svc[1])
print("3. Test Accuarcy: ",accuracy_scores_svc[2])
print("4. Test Accuarcy: ",accuracy_scores_svc[3])
print("5. Test Accuarcy: ",accuracy_scores_svc[4])
#Ortalama doğruluk skorunu hesaplayıp yazar
mean_accuracy = np.mean(accuracy_scores_svc)
print("SVC Mean Accuarcy:", mean_accuracy)

#StratifiedKFold ile 5 parçaya bölme
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=60)
accuracy_scores_decision_tree = []  # Karar ağacı için doğruluk skorlarını saklamak için liste

for train_index, test_index in skf.split(all_images, all_labels):
    X_train, X_test = all_images[train_index], all_images[test_index]
    y_train, y_test = all_labels[train_index], all_labels[test_index]

     # Karar Ağacı modelini oluştur ve eğit
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train.reshape(len(X_train), -1), y_train)
    
    # Karar Ağacı modelinin performansını değerlendir ve doğruluk skorunu sakla
    y_pred_decision_tree = decision_tree.predict(X_test.reshape(len(X_test), -1))
    accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
    accuracy_scores_decision_tree.append(accuracy_decision_tree)
    
print("Desicion Tree Algorithm")
#her splitteki doğruluk değerlerini yazdırır    
print("1. Test Accuarcy: ",accuracy_scores_decision_tree[0])
print("2. Test Accuarcy: ",accuracy_scores_decision_tree[1])
print("3. Test Accuarcy: ",accuracy_scores_decision_tree[2])
print("4. Test Accuarcy: ",accuracy_scores_decision_tree[3])
print("5. Test Accuarcy: ",accuracy_scores_decision_tree[4])
#Ortalama doğruluk skorunu hesaplayıp yazar/default olarak minkowski uzaklığı ile hesaplama yapılmış
mean_accuracy_decision_tree = np.mean(accuracy_scores_decision_tree)
print("Desicion Tree Test Mean Accuarcy:", mean_accuracy_decision_tree)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=60)
accuracy_scores_knn = []  # KNN için doğruluk skorlarını saklamak için liste

for train_index, test_index in skf.split(all_images, all_labels):
    X_train, X_test = all_images[train_index], all_images[test_index]
    y_train, y_test = all_labels[train_index], all_labels[test_index]

    # K-En Yakın Komşu modelini oluştur ve eğit
    knn = KNeighborsClassifier()
    knn.fit(X_train.reshape(len(X_train), -1), y_train)

    # K-En Yakın Komşu modelinin performansını değerlendir ve doğruluk skorunu sakla
    y_pred_knn = knn.predict(X_test.reshape(len(X_test), -1))
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    accuracy_scores_knn.append(accuracy_knn)

print("KNN ALGORİTHM")
#her splitteki doğruluk değerlerini yazdırır    
print("1. Test Accuarcy: ",accuracy_scores_knn[0])
print("2. Test Accuarcy: ",accuracy_scores_knn[1])
print("3. Test Accuarcy: ",accuracy_scores_knn[2])
print("4. Test Accuarcy: ",accuracy_scores_knn[3])
print("5. Test Accuarcy: ",accuracy_scores_knn[4])
# Ortalama doğruluk skorunu yazdır
mean_accuracy_knn = np.mean(accuracy_scores_knn)
print("KNN Mean Accuarcy:", mean_accuracy_knn)
