import matplotlib.pyplot as plt
file_nb = open("D:\conclusion\hdnb.txt", "r")
data_nb=file_nb.read()
file_nb.close()
nb=data_nb
print (nb)

file_knn = open("D:\conclusion\knnhdknn.txt", "r")
data_knn=file_knn.read()
file_knn.close()
knn= (data_knn)
print (knn)

file_svm = open("D:\conclusion\hdsvm.txt", "r")
data_svm=file_svm.read()
file_svm.close()
svm= data_svm
print (svm)

file_dt = open("D:\conclusion\hddt.txt", "r")
data_dt=file_dt.read()
file_dt.close()
dt= data_dt
print (dt)

file_lr = open("D:\conclusion\logostichdlr.txt", "r")
data_lr=file_lr.read()
file_lr.close()
lr= data_dt
print (lr)


plt.axis([0, 6, 0, 100])
x = [0,1,2,3,4,5]
my_xticks = [0,'NB','KNN','SVM','DT','LR']
plt.xticks(x, my_xticks)
y = [0,float(nb)*100,float(knn)*100,float(svm)*100,float(dt)*100,float(lr)*100]
plt.bar(x, y, align='center')

plt.title('Info')
plt.ylabel('ACCURACY')
plt.xlabel('ALGORITHMS')

plt.show()

