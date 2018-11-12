import numpy as np
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import imageio
from os import listdir
from os import mkdir
import shutil


np.random.seed(1)

image_path = './Asian'
train_num = 50
test_num = 350

D = 250*250 # original dimension
d = 800 # new dimension

ProjectionMatrix = np.random.randn(D, d)

def build_file_list(pattern, train_num, test_num):
    """
    INPUT:
        pattern = 'R.' or 'L.'
        train_num: number of train images
        test_num: number of test images

    OUTPUT:
        2 lists of filenames
    """
    train_file_list = []
    test_file_list = []
    count = 0
    for fi in listdir(image_path):
        if fi.find(pattern) != -1:
            if (count < train_num):                
                train_file_list.append(fi)
            elif (count < (train_num + test_num)):
                test_file_list.append(fi)
            else:
                break
            count += 1

    return (train_file_list, test_file_list)
    

def build_file_list_ext(pattern, file_num):
    """
    INPUT:
        pattern = 'R.' or 'L.'
        file_num: number of images

    OUTPUT:
        1 list of filenames
    """
    file_list = []
    count = 0
    for fi in listdir(image_path):
        if fi.find(pattern) != -1:
            if (count < file_num):                
                file_list.append(fi)
            else:
                break
            count += 1

    return file_list 

############################################################################
# convert RBG image to gray scale
def rgb2gray(rgb):
    #Y' = 0.299*R + 0.587*G + 0.114*B
    return rgb[:,:,0]*.299 + rgb[:,:,1]*.587 + rgb[:,:,2]*.114

# feature extraction
def vectorize_image(filename):
    # load image
    rgb = imageio.imread(filename)
    #convert to gray scale
    gray = rgb2gray(rgb)
    # vectorization each row is a data point
    im_vec = gray.reshape(1, D)
    return im_vec

def build_data_matrix(train_fl):
    total_images = len(train_fl)
    X_full = np.zeros((total_images, D))
    y = np.hstack((np.zeros((total_images//2, )), np.ones((total_images//2), )))

    for i in range(total_images):
        buf = vectorize_image(image_path + '/' + train_fl[i])
        if len(buf) > 0:
            X_full[i, :] = buf

    X = np.dot(X_full, ProjectionMatrix)
    return (X, y)

(left_train_fl, left_test_fl) = build_file_list('L.', train_num, test_num)
(right_train_fl, right_test_fl) = build_file_list('R.', train_num, test_num)

train_fl = left_train_fl + right_train_fl
test_fl = left_test_fl + right_test_fl

(X_train_full, y_train) = build_data_matrix(train_fl)

x_mean = X_train_full.mean(axis = 0)
x_var = X_train_full.var(axis = 0)

def extract_feature(X):
    return (X - x_mean)/x_var

#########################
# try LogisticRegression
X_train = extract_feature(X_train_full)
X_train_full = None ## free this variable

(X_test_full, y_test) = build_data_matrix(test_fl)
X_test = extract_feature(X_test_full)
X_test_full = None

logreg = linear_model.LogisticRegression(C=1e9)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy for Logistic Regression: %.2f %%' %(100 * accuracy_score(y_test, y_pred)))

#########################
# try SVC
clf = SVC(kernel='poly', degree = 3, gamma=1, C = 100)

svr_lin = SVC(C=1e3, kernel='linear')
svr_poly = SVC(C=1e3, kernel='poly', degree=2)
svr_rbf = SVC(C=1e3, kernel='rbf', degree = 3, gamma=1)

svr_lin.fit(X_train, y_train)
y_svc_lin_pred = svr_lin.predict(X_test)
print('Accuracy for linear: %.2f %%' %(100 * accuracy_score(y_test, y_svc_lin_pred)))

svr_poly.fit(X_train, y_train)
y_svr_poly_pred = svr_poly.predict(X_test)
print('Accuracy for poly: %.2f %%' %(100 * accuracy_score(y_test, y_svr_poly_pred)))

svr_rbf.fit(X_train, y_train)
y_svr_rbf_pred = svr_rbf.predict(X_test)
print('Accuracy for rbf: %.2f %%' %(100 * accuracy_score(y_test, y_svr_rbf_pred)))