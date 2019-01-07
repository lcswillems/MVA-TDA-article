#first part : build all diags
import os
from sklearn import datasets
import numpy as np
import persim
import matplotlib.pyplot as plt

diags = []

for file in os.listdir("Data_Hand/Precise Masks"):
    basename = file[:-4]
    pts = sio.loadmat("Data_Hand/Boundaries/{}.mat".format(basename))['hand_bounadry']
    img = misc.imread("Data_Hand/Precise Masks/{}.bmp".format(basename))

    Gf = get_fcenter(pts, img)
    persistence_pts = persistence_pts_from_Gf(Gf)
    diags += [persistence_pts]
    
#then, non compulsory, heat map of the euclidean dist of these diags (only the first sample out of the 10)
#(there are 10 samples for each gesture, each person. 10 persons. 10 gestures)

n = np.array(diags).shape[0]
m = 10
p = int(n/m)

dis = np.zeros((p,p))
for i in range(0,n,m):
    #print(i)
    for j in range(0,i,m):
        #print(j)
        
        distance_bottleneck, _ = persim.bottleneck(diags[i], diags[j], matching=True)
        dis[int(i/m),int(j/m)]=distance_bottleneck
        #print(dis[i,j])

plt.imshow(dis, cmap='hot', interpolation='nearest')
plt.colorbar()

#at last, compare each diag with the 10 refs for the gestures (1st person, 1st sample has been chosen for the ref).
#a prediction is then chosen based on the diagram. the argmin of the distances to the refs.
diag_ref = np.array([diags[0],diags[100],diags[200],diags[300],
                     diags[400],diags[500],diags[600],diags[700],
                     diags[800],diags[900]])


res = np.zeros(n)
for i in range(n):
    print(i)
    dists = []
    for j in range(10):
        distance_bottleneck, _ = persim.bottleneck(diags[i], diag_ref[j], matching=True)
        dists += [distance_bottleneck]
    res[i] = np.argmin(np.array(dists))

    
#it appears that it is not a very good solution. 621 wrong guesses out of 1000. Maybe use Wasserstein instead of Bottleneck.
#or using the first sample is not a good idea
ideal = np.array([int(i/100) for i in range(1000)])
np.sum(ideal != res)

#confusion matrix
confus = np.zeros((m,m))
for i in range(n):
    k = int(i*m/n)
    l = int(res[i])
    confus[k,l]+=1
    
fig, ax = plt.subplots(figsize=(8,8))
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(confus, cmap='seismic')

for (i, j), z in np.ndenumerate(confus):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.show()
