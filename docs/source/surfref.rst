==================================
Implementing SURF-ref With Mahotas
==================================

This is a companion to the paper `Determining the subcellular location of new
proteins from microscope images using local features`__ by Coelho et al. (2013).

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
