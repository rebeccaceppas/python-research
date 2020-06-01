import numpy as np
import random

def distance(p1,p2):
    ''' Finds the distance between p1 and p2.'''
    return (np.sqrt(np.sum(np.power(p2-p1,2))))


def majority_vote(votes):
    ''' Takes list of votes as input, creates dictionary with count for each given vote and then checks which key has largest value. If there is a tie, randomly selects one winner. '''
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)

    return random.choice(winners)

#commenting out because want to use longer one for random attribute
"""

import scipy.stats as ss
def majority_vote(votes):
    ''' Return the most common element in votes. '''
    mode, count = ss.mstats.mode(votes)
    return mode 

"""

def find_nearest_neighbors(p, points, k=5):
    ''' Find k nearest neighbors of point p and return their indices. '''
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p,points[i])
    ind = np.argsort(distances)
    return ind[0:k]


def knn_predict(p,points,outcomes,k=5):
    ''' Finds k nearest neighbors and predicts the class of p based on majority vote. '''
    ind = find_nearest_neighbors(p,points,k)
    return majority_vote(outcomes[ind])

import scipy.stats as ss

def generate_synthetic_data(n=50):
    ''' Creates two sets of points from bivariate normal distrubutions. '''
    points = np.concatenate((ss.norm(0,1).rvs((n,2)),ss.norm(1,1).rvs((n,2))), axis=0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)), axis=0)
    return (points, outcomes)

import matplotlib.pyplot as plt

""" (points,outcomes) = generate_synthetic_data(20)
n=20
plt.figure()
#first n rows in columns 0 and 1
plt.plot(points[:n,0], points[:n,1], 'ro')
#last n rows in columns 0 and 1
plt.plot(points[n:,0], points[n:,1], 'bo')
plt.savefig('bivariatedata.jpg')
plt.show() """

def make_prediction_grid(predictors, outcomes, limits, h, k):
    ''' Classifiy each point on the prediction grid. '''
    (x_min,x_max,y_min,y_max) = limits
    xs = np.arange(x_min,x_max,h)
    ys = np.arange(y_min,y_max,h)
    xx, yy = np.meshgrid(xs,ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k)

    return (xx, yy, prediction_grid)


def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """' Plot KNN predictions for every point on the grid.' """
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)

'''(predictors, outcomes)=generate_synthetic_data(50)

(xx,yy,prediction_grid) = make_prediction_grid(predictors, outcomes, (-3,4,-3,4), 0.1, 50)
plot_prediction_grid(xx,yy,prediction_grid,'knn_synth50.jpg')'''


from sklearn import datasets
iris = datasets.load_iris()
iris['data']

#all of rows but only columns 0 and 1
predictors = iris['data'][:, 0:2]
outcomes = iris['target']

#plotting all predictors with outcome=0 (all rows and only column 0), same thign for column 1 for y-axis
plt.plot(predictors[outcomes==0][:, 0], predictors[outcomes==0][:,1], 'ro')
#same thing for outcome 1
plt.plot(predictors[outcomes==1][:, 0], predictors[outcomes==1][:,1], 'go')
#and for outcome 2
plt.plot(predictors[outcomes==2][:, 0], predictors[outcomes==2][:,1], 'bo')
plt.savefig('iris.jpg')


(xx,yy,prediction_grid) = make_prediction_grid(predictors, outcomes, (4,8,1.5,4.5), 0.1, 5)
plot_prediction_grid(xx,yy,prediction_grid,'iris_grid.jpg')

#using SciKitLearn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors,outcomes)
sk_predictions = knn.predict(predictors)

#using out own code
my_predictions = np.array([knn_predict(p,predictors,outcomes,5) for p in predictors])

#out boolean array telling you for which predictions they agree
sk_predictions == my_predictions
#true is 1 and false is 0
np.mean(sk_predictions == my_predictions) #out 0.96
#how often do our predictions agree with actual outcomes
print(np.mean(sk_predictions == outcomes)) #out 0.83
print(np.mean(outcomes == my_predictions)) #out 0.85