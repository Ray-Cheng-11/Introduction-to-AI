class WeakClassifier:
    def __init__(self, feature, threshold=0, polarity=1):
        """
        Parameters:
          feature: The HaarFeature class.
          threshold: The threshold for the weak classifier.
          polarity: The polarity of the weak classifier.(1 or -1)
        """
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity
    def train_weak(self, featureVals, labels, features, weights):
        """
        first we use for loop to calculate the total weight of pos and neg
        for each feature in featureVal, we calculate it's error
        and use it's feature as threshold and polarity by how many pos neg it detected
        create and returethe classifiers by the parameter 
        
        """
        total_pos = 0
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, labels):
          if label == 1:
            total_pos+=w
          else:
            total_neg+=w
        classifiers = []
        for index, feature in enumerate(featureVals):
            applied_feature = sorted(zip(weights, feature, labels), key=lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1
                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            clf  = WeakClassifier(best_feature,best_threshold,best_polarity)
            classifiers.append(clf)
        return classifiers

    def __str__(self):
        return "Weak Clf (threshold=%d, polarity=%d, %s" % (
            self.threshold,
            self.polarity,
            str(self.feature),
        )

    def classify(self, x):
        """
        Classifies an integral image based on a feature f
        and the classifiers threshold and polarity.
          Parameters:
            x: A numpy array with shape (m, n) representing the integral image.
          Returns:
            1 if polarity * feature(x) < polarity * threshold
            0 otherwise
        """
        return (
            1
            if self.polarity * self.feature.computeFeature(x)
            < self.polarity * self.threshold
            else 0
        )
