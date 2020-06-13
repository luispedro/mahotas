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
