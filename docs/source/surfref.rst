==================================
Implementing SURF-ref With Mahotas
==================================

This is a companion to the paper `Determining the subcellular location of new
proteins from microscope images using local features
<http://dx.doi.org/10.1093/bioinformatics/btt392>`__
by Coelho et al. (2013).

::

    def surf_ref(f, ref):
        '''
        features = surf_ref(f, ref)

        Computer SURF-ref features
        
        Parameters
        ----------
        f : ndarray
            input image
        ref : ndarray
            Corresponding reference image

        Returns
        -------
        features : ndarray
            descriptors
        '''
        fi = surf.integral(f.copy())
        points = surf.interest_points(fi, 6, 24, 1, max_points=1024, is_integral=True)
        descs = surf.descriptors(fi, points, is_integral=True, descriptor_only=True)
        if ref is None:
            return descs
        descsref = surf.descriptors(ref, points, descriptor_only=True)
        return np.hstack( (descs, descsref) )


This function can take any number of reference images.

We now compute all features for all images in widefield dataset::

    from glob import glob
    import re

    basedir = 'rt-widefield' # Edit as needed

    features = []
    labels = []

    # We need the following to keep track of the proteins:
    origins = []
    prev_origin = ''
    origin_counter = -1 # set to -1 so it will be correctly initialized on the first image

    for dir in glob(basedir):
        if dir == 'README': continue
        for f in glob('{}/{}/*-protein.tiff'.format(basedir, dir)):
            origin = f[:5]
            if origin != prev_origin:
                origin_counter += 1
                prev_origin = origin

            f = '{}/{}/{}'.format(basedir, dir, f)
            f = mh.imread(f)
            ref = mh.imread(f.replace('protein','dna'))
            features.append(surf_ref(f, ref))
            labels.append(dir)
            origins.append(origin_counter)

Classification
--------------

With all the precomputed features, we can now run 10~fold cross-validation on
these features.

We will using milk for machine learning::

    import milk

Milk's interface is around learner objects. We are going to define a function::

    def train_model(features, labels):

The first step is to find centroids::

    # concatenate all the features:
    concatenated = np.concatenate(features)

We could use the whole array concatenated for kmeans. However, that would take
a long time, so we will use just 1/16th of it::

    concatenated = concatenated[::16]
    _,centroids = milk.kmeans(concatenated, k=len(labels)//4, R=123)

The R argument is the random seed. We set it to a constant to get reproducible
results, but feel free to vary it.

Based on these centroids, we project the features to histograms. Now, we are
using all of the features::

    features = np.array([
        project_centroids(centroids, fs, histogram=True)
            for fs in features])

Finally, we can use a traditional milk learner (which will perform feature
selection, normalization, and SVM training)::

    learner = milk.defaultlearner()
    model = learner.train(features, labels)

We must return both the centroids that were used and the classification model::

    return centroids, model

To classify an instance, we define another function, which uses the centroids
and the model::

    def apply_many(centroids, model, features):
        features = np.array([
                project_centroids(centroids, fs, histogram=True)
                    for fs in features])
        return model.apply_many(features)

In fact, while the above will work well, milk already provides a learner object
which will perform all of those tasks!

::

    import milk
    from milk.supervised.precluster import frac_precluster_learner

    learner = frac_precluster_learner(kfrac=4, sample=16)
    cmatrix,names = milk.nfoldcrossvalidation(features, labels, origins=origins, learner=learner)
    acc = cmatrix.astype(float).trace()/cmatrix.sum()
    print('Accuracy: {.1}%'.format(100.*acc))
