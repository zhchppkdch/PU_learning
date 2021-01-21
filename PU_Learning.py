
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import f1_score, mean_squared_error as mse
from collections import Counter
from sklearn.model_selection import train_test_split

# -------------------- First PU Learning Method #--------------------
class TwoStepTechnique(BaseEstimator, ABC):

    def __init__(self):
        self.classifier = None

    @abstractmethod
    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the confident negative (and optionally extend positive) examples;
        X: NxM array containing M feature values for N examples;
        s: Nx1 array indicating for N examples whether they are postive (1) or unlabeled (0);
        Returns: n,p. Where n and p have the same dimensions as s. 1 indicates a confident negative and
        respectively positive example. If the set of positive examples is not extended, then p is equal to n.
        Examples that are 0 in both n and p remain unlabeled.
        """

        pass

    @abstractmethod
    def step2(self, X, n, p) -> BaseEstimator:
        """
        Trains a classifier
        X: NxM array containing M feature values for N examples;
        n: N array: This array indicates for N examples whether they are negatively labeled (1) or not (0);
        p: N array: This array indicates for N examples whether they are positively labeled (1) or not (0) ;
        Returns: classifier which inherits sckikit-learn's BaseEstimator.
        """


class PEBL(TwoStepTechnique):
    def __init__(self,
                 nb_strong_negatives=10,
                 classifier_for_step_1=RandomForestClassifier(criterion='entropy', n_jobs=-1, random_state=331),
                 classifier_for_step_2=svm.SVC(kernel='linear', probability=True, random_state=331),
                 seed=331
                 ):
        super().__init__()

        self.classifier_for_step_1 = classifier_for_step_1
        self.classifier_for_step_2 = classifier_for_step_2

        ## ADD YOUR CODE HERE (instantiate the parameters)

    def step1(self, X, s) -> Tuple[np.ndarray, np.ndarray]:
        '''
        In the first step, train a non-traditional classifier and select the top nb_strong_negatives strongest negatives as
        negative set. Note that nb_strong_negatives is a parameter with default value equal to 10. To select the strongest
        negatives, first order the examples according to their output probability and then pick up only the first
        nb_strong_negatives values.
        '''

        # set the seed for experiments reproducibility
        s2 = s
        X2_train = X

        s_size = np.size(s2)
        s_neg = np.zeros(s_size)
        s_pos = np.empty_like(s2)
        s_pos[:] = s2

        self.classifier_for_step_1.fit(X2_train, s2)
        pred = self.classifier_for_step_1.predict_proba(X2_train)[:, 1]

        #sort the pred_prob with lowest and find the 10 negative
        neg_index = np.argsort(pred)[:10]
        for i in neg_index:
            s_neg[i] = 1

        # find the postive feature by caculate the mean of all labeled postive,
        # and if any unlabeled data is greater than that mean, make it postive
        total = 0
        pos_count = 0
        for i in range(len(pred)):
            if (s_pos[i] == 1):
                total += pred[i]
                pos_count += 1
        pos_threshold = total / pos_count

        for j in range(len(s_pos)):
            if (pred[j] >= pos_threshold):
                if (s_pos[j] == 0):
                    s_pos[j] = 1
        # tips: watch out that not all unlabeled examples will be marked as strong negatives but at least 1 ...
        return (s_neg,s_pos)

    def step2(self, X, n, p) -> BaseEstimator:
        # set the seed for experiments reproducibility
        s_size = np.size(n)
        X2_train = X
        s_neg,s_pos = n,p
        s_combined = np.zeros(s_size)

        # take out neg and pos to fit svm, and rest of unlabeled data will be used for predication
        for i in range(s_size):
            if (s_neg[i] == 1):
                s_combined[i] = 0
            elif (s_pos[i] == 1):
                s_combined[i] = 1
            else:
                s_combined[i] = -1

        s_combined = np.array(s_combined)
        final = np.concatenate((X2_train, s_combined[:, None]), axis=1)


        df = pd.DataFrame(data=final[0:, 0:], index=[i for i in range(final.shape[0])],
                          columns=['f' + str(i) for i in range(final.shape[1])])
        name = list(df.columns)[-1]
        unlabeled = df.loc[df[name] == -1]  # unlabeled data used for pred
        labeled_df = df.loc[df[name] != -1]
        labled = labeled_df.iloc[:, -1].values  # postive + strong neg for train to fit y
        labeled_data = labeled_df.iloc[:, :-1].values  # postive + strong neg for train to fit x
        unlabeled = unlabeled.iloc[:, :-1].values

        #loop until there is no negative can be found in unlabeled set
        for i in range(20):
            self.classifier_for_step_2.fit(labeled_data, labled)
            pred_prob = self.classifier_for_step_2.predict_proba(unlabeled)[:, -1]
            pred = self.classifier_for_step_2.predict(unlabeled)

            N = []
            N_label = []
            for i in range(len(pred)):
                if (pred[i] == 0):
                    N.append(i)
            if (len(N) == 0): break

            N2_data = np.take(unlabeled, N, axis=0)

            count = 0
            for i in range(s_size):
                for j in N2_data:
                    comparison = j == X2_train[i]
                    equal_arrays = comparison.all()
                    if (equal_arrays == True):
                        if (s_combined[i] == -1):
                            s_combined[i] = 0

            s_combined = np.array(s_combined)
            final = np.concatenate((X2_train, s_combined[:, None]), axis=1)
            df = pd.DataFrame(data=final[0:, 0:], index=[i for i in range(final.shape[0])],
                              columns=['f' + str(i) for i in range(final.shape[1])])
            name = list(df.columns)[-1]
            unlabeled = df.loc[df[name] == -1]  # unlabeled data used for pred
            labeled_df = df.loc[df[name] != -1]
            labled = labeled_df.iloc[:, -1].values  # postive + strong neg for train to fit y
            labeled_data = labeled_df.iloc[:, :-1].values  # postive + strong neg for train to fit x
            unlabeled = unlabeled.iloc[:, :-1].values
            if (len(unlabeled) == 0): break


        return self.classifier_for_step_2

    def fit(self, X, s):
        n, p = self.step1(X, s)
        classifier_step2 = self.step2(X, n, p)
        self.classifier = classifier_step2

    def predict(self, X):
        y_pred = self.classifier.predict(X)

        return np.array(y_pred)

    def predict_proba(self, X):

        probability = self.classifier.predict_proba(X)
        return probability

#--------------------#--------------------#--------------------#--------------------
#-------------------- Second PU Learning Method #--------------------

class ElkanNotoLabelFrequencyEstimator:
    def __init__(self,
                 classifier=svm.SVC(kernel='rbf', probability=True, random_state=331, max_iter=1.e6),
                 val_ratio=0.1,
                 seed=331):

    ## ADD YOUR CODE HERE
        self.classifier = classifier

    def estimate_c(self, X, s):
        # fit the model to estimate c

        #choose estimator 1 because it is more stable than 2,3
        #X_train, X_val, y_train, y_val = train_test_split(X, s, test_size=0.25, random_state=1)
        svm1 = svm.SVC(kernel='linear', probability=True, random_state=331, max_iter=1.e6)
        svm1.fit(X,s)
        Totoal = sum(svm1.predict_proba(X)[:,1])
        c = Totoal/ s.size


        return c


class ElkanNotoWeightedClassifier(BaseEstimator):

    def __init__(self,
                 classifier=svm.SVC(kernel='linear', probability=True, random_state=331, max_iter=1.e6),
                 seed=331):

        self.c = None
        self.classifier = classifier

    def set_c(self, c):

        self.c = c

    def fit(self, X, s):
        ## ADD YOUR CODE HERE (see inline tips)

        # create the training data with 1) labeled examples, 2) unlabeled examples treated as positives,
        # 3) unlabeled examples treated as negatives

        X1_train = X
        s1 = s
        svm1 = svm.SVC(kernel='linear', probability=True, random_state=331, max_iter=1.e6)
        svm1.fit(X1_train, s1)

        labeled = np.where(s1 == 1)[0]
        X_labeled = X1_train[labeled]  # labeled data

        unlabel = np.where(s1 == 0)[0]
        X_unlabeled = X1_train[unlabel]


        # create the weights according to eq.3

        lab_prob = svm1.predict_proba(X_unlabeled)[:, 1] #p(s=1|x)
        weights_unlab = (1 - self.c) / (self.c) * (lab_prob) / (1 - lab_prob)  # weight = p(y=1|x,s=0)

        # create the labels for the fitting method
        X_lab_size = labeled.shape[0]
        X_unlab_size = unlabel.shape[0]

        y_lab = np.ones((X_lab_size, 1))
        y_unlab_pos = np.ones((X_unlab_size, 1))  # unlabeled positive data
        y_unlab_neg = np.zeros((X_unlab_size, 1))  # unlabeded negative data
        y = np.vstack([y_lab, y_unlab_pos, y_unlab_neg])

        w_lab = np.ones((X_lab_size, 1))
        w_unlab_pos = weights_unlab.reshape(-1, 1)
        # w_unlab_neg = 1 - weights_unlab
        w_unlab_neg = 1 - weights_unlab.reshape(-1, 1)
        w = np.vstack([w_lab, w_unlab_pos, w_unlab_neg])

        X = np.vstack([X_labeled, X_unlabeled, X_unlabeled])
        combined = np.hstack([X, y, w])

        total_data = combined[:, :-2]
        total_data = np.ascontiguousarray(total_data)

        total_label = combined[:, -2]
        total_label = np.ascontiguousarray(total_label)

        total_weights = combined[:, -1]
        total_weights = np.ascontiguousarray(total_weights)

        self.classifier.fit(total_data, total_label, sample_weight=total_weights)

        return self.classifier

    def predict(self, X):
        prediction = self.classifier.predict(X)
        return prediction

    def predict_proba(self, X):

        probability = self.classifier.predict_proba(X)
        return probability
