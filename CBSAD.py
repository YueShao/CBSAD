from pyod.models.base import BaseDetector
from utils import get_cormat_data,find_groups,sortgroup
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
import numpy as np
from sklearn.utils import check_array
from pyod.utils.utility import check_detector
from pyod.utils.utility import check_parameter
from sklearn.base import clone

MAX_INT = np.iinfo(np.int32).max

def _set_random_states(estimator, random_state=None):
    random_state = check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == 'random_state' or key.endswith('__random_state'):
            to_set[key] = random_state.randint(MAX_INT)

    if to_set:
        estimator.set_params(**to_set)

class CBSAD(BaseDetector):
    def __init__(self, 
                 contamination=0.1,
                 thresh=2.5,
                 base_estimator=None,
                 single_estimator=None, 
                 check_detector=True,
                 n_jobs=1,
                 random_state=None,
                estimator_params=None,
                weighted = True):
        super(CBSAD, self).__init__(contamination=contamination)
        self.thresh=thresh
        self.base_estimator = base_estimator
        self.single_estimator = single_estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.check_detector=check_detector
        self.weighted = weighted
        if estimator_params is not None:
            self.estimator_params = estimator_params
        else:
            self.estimator_params = {}

        
            
    def fit(self, X, y=None):
        

        random_state = check_random_state(self.random_state)

        X = check_array(X)
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        self._set_n_classes(y)

        # expect at least 2 features, does not make sense if only have
        # 1 feature
        check_parameter(self.n_features_, low=2, include_left=True,
                        param_name='n_features')
        self._validate_base_estimator(default=KNN(n_jobs=self.n_jobs))
        self._validate_single_estimator(default=KNN(n_jobs=self.n_jobs))
        
        cormat=get_cormat_data(X)
        connect,groups = find_groups(cormat,self.thresh)
        
        
#         print(groups)
        
        i= X.shape[0]
        datagroup=sortgroup(groups)
        j=len(datagroup)
        self.n_groups_=j
        
        
        self.estimators_=[]
        self.estimators_groups_ = datagroup
        
        
        for m in range(j):
            if len(datagroup[m])==1:
                estimator = self._make_single_estimator(append=False,
                                             random_state=random_state)
            else:
                estimator = self._make_base_estimator(append=False,
                                             random_state=random_state)
            traindata = np.array(X[:,datagroup[m]])
            estimator.fit(traindata)
            self.estimators_.append(estimator)
            
        all_decision_scores = self._get_decision_scores()
        self.decision_scores_ = self._weighted_combination(all_decision_scores)
        
        self._process_decision_scores()
        
        return self
    
    
    def decision_function(self, X):

        check_is_fitted(self, ['estimators_', 'estimators_groups_',
                               'decision_scores_', 'threshold_', 'labels_'])
        X = check_array(X)

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        all_pred_scores = self._predict_decision_scores(X)

        
        return self._weighted_combination(all_pred_scores)
        
    def _predict_decision_scores(self, X):
        all_pred_scores = np.zeros([X.shape[0], self.n_groups_])
        for i in range(self.n_groups_):
            group = self.estimators_groups_[i]
            all_pred_scores[:, i] = self.estimators_[i].decision_function(
                X[:,group])
        return all_pred_scores

    def _get_decision_scores(self):
        all_decision_scores = np.zeros([self.n_samples_, self.n_groups_])
        for i in range(self.n_groups_):
            all_decision_scores[:, i] = self.estimators_[i].decision_scores_
        return all_decision_scores
    
    def _weighted_combination(self,all_decision_scores):
        maxscore=np.max(all_decision_scores)
        minscore=np.min(all_decision_scores)
        if(maxscore-minscore != 0):
            all_decision_scores=(all_decision_scores-minscore)/(maxscore-minscore)
        if self.weighted:
            for i in range(self.n_groups_):
                weight=self._get_xb_score(all_decision_scores[:,i])
                all_decision_scores[:,i]=all_decision_scores[:,i]*weight

        result_final=np.mean(all_decision_scores,axis=1)
        return result_final
    
    def _validate_base_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

        # make sure estimator is consistent with sklearn
        if self.check_detector:
            check_detector(self.base_estimator_)
            
    def _validate_single_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if self.single_estimator is not None:
            self.single_estimator_ = self.single_estimator
        else:
            self.single_estimator_ = default

        if self.single_estimator_ is None:
            raise ValueError("single_estimator cannot be None")

        # make sure estimator is consistent with sklearn
        if self.check_detector:
            check_detector(self.single_estimator_)

            
    def _make_base_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        sklearn/base.py

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """

        # TODO: add a check for estimator_param
        estimator = clone(self.base_estimator_)
        estimator.set_params(**self.estimator_params)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator
    
    def _xb(self,x, c, c_normal, c_anomaly):
        return sum((x-c)**2) / (len(x) * ((c_normal - c_anomaly) ** 2))

    def _get_xb_score(self,x):
        
        x=x.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2).fit(x)
        centera,centerb=kmeans.cluster_centers_
        labels=kmeans.labels_
        if centera==centerb:
            return 0
        if centera<centerb:
            c_normal=centera
            c_anomaly=centerb
            L_anomaly=1
        else:
            c_normal=centerb
            c_anomaly=centera
            L_anomaly=0
        n = x.shape[0]
        c = [c_anomaly if labels[i]== L_anomaly else c_normal for i in range(n)]
        score = 1-self._xb(x, c, c_normal, c_anomaly)
        return score
    
    def _make_single_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `base_estimator_` attribute.

        sklearn/base.py

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """

        # TODO: add a check for estimator_param
        estimator = clone(self.single_estimator_)
        estimator.set_params(**self.estimator_params)

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return iter(self.estimators_)