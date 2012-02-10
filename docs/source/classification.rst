============================
Classification Using Mahotas
============================

Here is an example of using mahotas and `milk <http://luispedro.org/software/milk>`_
for image classification.  I assume that there are three important directories:
``positives/`` and ``negatives/`` contain the manually labeled examples, and
the rest of the data is in an ``unlabeled/`` directory.

Here is the simple algorithm:

1. Compute features for all of the images in positives and negatives
2. learn a classifier
3. use that classifier on the unlabeled images

In the code below I used `jug <http://luispedro.org/software/jug>`_ to give you
the possibility of running it on multiple processors, but the code also works
if you remove every line which mentions ``TaskGenerator``::

    from glob import glob
    import mahotas
    import mahotas.features
    import milk
    from jug import TaskGenerator


    @TaskGenerator
    def features_for(imname):
        img = mahotas.imread(imname)
        return mahotas.features.haralick(img).mean(0)

    @TaskGenerator
    def learn_model(features, labels):
        learner = milk.defaultclassifier()
        return learner.train(features, labels)

    @TaskGenerator
    def classify(model, features):
         return model.apply(features)

    positives = glob('positives/*.jpg')
    negatives = glob('negatives/*.jpg')
    unlabeled = glob('unlabeled/*.jpg')


    features = map(features_for, negatives + positives)
    labels = [0] * len(negatives) + [1] * len(positives)

    model = learn_model(features, labels)

    labeled = [classify(model, features_for(u)) for u in unlabeled]

This uses texture features, which is probably good enough, but you can play
with other features in ``mahotas.features`` if you'd like (or try
``mahotas.surf``, but that gets more complicated). In general, I have found it
hard to do classification with the sort of hard thresholds you are looking for
unless the scanning is very controlled.

(This was motivated by `a question on Stackoverflow <http://stackoverflow.com/questions/5426482/using-pil-to-detect-a-scan-of-a-blank-page/5505754>`__).


