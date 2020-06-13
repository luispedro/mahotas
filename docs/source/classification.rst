======================================
Tutorial: Classification Using Mahotas
======================================

Here is an example of using mahotas and `scikit-learn
<https://scikit-learn.org>`__ for image classification (but most of the code
can easily be adapted to use another machine learning package).  I assume that
there are three important directories: ``positives/`` and ``negatives/``
contain the manually labeled examples, and the rest of the data is in an
``unlabeled/`` directory.

Here is the simple algorithm:

1. Compute features for all of the images in positives and negatives
2. learn a classifier
3. use that classifier on the unlabeled images

In the code below I used `jug <http://luispedro.org/software/jug>`_ to give you
the possibility of running it on multiple processors, but the code also works
if you remove every line which mentions ``TaskGenerator``.

We start with a bunch of imports::

    from glob import glob
    import mahotas
    import mahotas.features
    from jug import TaskGenerator

Now, we define a function which computes features. In general, texture features
are very fast and give very decent results::

    @TaskGenerator
    def features_for(imname):
        img = mahotas.imread(imname)
        return mahotas.features.haralick(img).mean(0)

``mahotas.features.haralick`` returns features in 4 directions. We just take
the mean (sometimes you use the spread ``ptp()`` too).

Now a pair of functions to learn a classifier and apply it. These are just
``scikit-learn`` functions::

    @TaskGenerator
    def learn_model(features, labels):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
        clf.fit(features, labels)
        return clf

    @TaskGenerator
    def classify(model, features):
         return model.predict(features)

We assume we have three pre-prepared directories with the images in jpeg
format. This bit you will have to adapt for your own settings::

    positives = glob('positives/*.jpg')
    negatives = glob('negatives/*.jpg')
    unlabeled = glob('unlabeled/*.jpg')


Finally, the actual computation. Get features for all training data and learn a
model::

    features = map(features_for, negatives + positives)
    labels = [0] * len(negatives) + [1] * len(positives)

    model = learn_model(features, labels)

    labeled = [classify(model, features_for(u)) for u in unlabeled]

This uses texture features, which is probably good enough, but you can play
with other features in ``mahotas.features`` if you'd like (or try
``mahotas.surf``, but that gets more complicated).

(This was motivated by `a question on Stackoverflow
<http://stackoverflow.com/questions/5426482/using-pil-to-detect-a-scan-of-a-blank-page/5505754>`__).


