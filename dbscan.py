# -*- coding: utf-8 -*-

import numpy
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DBSCAN(object):

    def __init__(self, eps, MinPts, verbose = True, precomputed = True, metric = 'cosine'):
        """
        arguments:
            eps
            MinPts
            verbose
            precomputed - use precomputed distances matrxi D
            metric - should be 'cosine' or 'euclidian'
        """
        assert eps > 0
        assert metric in ('euclidian','cosine')
        self.eps = eps
        self.MinPts = MinPts
        self.verbose = verbose
        self.precomputed = precomputed
        self.metric = metric

    def fit(self, D):

        def MyDBSCAN(D, eps, MinPts):
            """
            Cluster the dataset `D` using the DBSCAN algorithm.
            MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
            `eps`, and a required number of points `MinPts`.
            It will return a list of cluster labels. The label -1 means noise, and then
            the clusters are numbered starting from 1.
            """
            # This list will hold the final cluster assignment for each point in D.
            # There are two reserved values:
            #    -1 - Indicates a noise point
            #     0 - Means the point hasn't been considered yet.
            # Initially all labels are 0.
            #import pudb; pudb.set_trace()
            labels = numpy.zeros(len(D))

            # C is the ID of the current cluster.
            C = 0
            # This outer loop is just responsible for picking new seed points--a point
            # from which to grow a new cluster.
            # Once a valid seed point is found, a new cluster is created, and the
            # cluster growth is all handled by the 'expandCluster' routine.
            # For each point P in the Dataset D...
            # ('P' is the index of the datapoint, rather than the datapoint itself.)
            for P in tqdm(range(0, len(D)), disable=(not self.verbose)):
                # Only points that have not already been claimed can be picked as new
                # seed points.
                if not (labels[P] == 0):
                    continue
                # Find all of P's neighboring points.
                NumNeighborPts, NeighborPts = regionQuery(D, P, eps)
                # If the number is below MinPts, this point is noise.
                # This is the only condition under which a point is labeled
                # NOISE--when it's not a valid seed point. A NOISE point may later
                # be picked up by another cluster as a boundary point (this is the only
                # condition under which a cluster label can change--from NOISE to
                # something else).
                if NumNeighborPts < MinPts:
                    labels[P] = -1
                # Otherwise, if there are at least MinPts nearby, use this point as the
                # seed for a new cluster.
                else:
                    C += 1
                    growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
            # All data has been clustered!
            return labels
        def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
            """
            Grow a new cluster with label `C` from the seed point `P`.
            This function searches through the dataset to find all points that belong
            to this new cluster. When this function returns, cluster `C` is complete.
            Parameters:
            `D`      - The dataset (a list of vectors)
            `labels` - List storing the cluster labels for all dataset points
            `P`      - Index of the seed point for this new cluster
            `NeighborPts` - All of the neighbors of `P`
            `C`      - The label for this new cluster.
            `eps`    - Threshold distance
            `MinPts` - Minimum required number of neighbors
            """

            # Assign the cluster label to the seed point.
            labels[P] = C
            # Look at each neighbor of P (neighbors are referred to as Pn).
            # NeighborPts will be used as a FIFO queue of points to search--that is, it
            # will grow as we discover new branch points for the cluster. The FIFO
            # behavior is accomplished by using a while-loop rather than a for-loop.
            # In NeighborPts, the points are represented by their index in the original
            # dataset.
            i = 0
            while i < len(NeighborPts):
                # Get the next point from the queue.
                Pn = NeighborPts[i]
                # If Pn was labelled NOISE during the seed search, then we
                # know it's not a branch point (it doesn't have enough neighbors), so
                # make it a leaf point of cluster C and move on.
                if labels[Pn] == -1:
                    labels[Pn] = C
                # Otherwise, if Pn isn't already claimed, claim it as part of C.
                elif labels[Pn] == 0:
                    # Add Pn to cluster C (Assign cluster label C).
                    labels[Pn] = C
                    # Find all the neighbors of Pn
                    PnNeighborPts_Num, PnNeighborPts = regionQuery(D, Pn, eps, NeighborPts)
                    # If Pn has at least MinPts neighbors, it's a branch point!
                    # Add all of its neighbors to the FIFO queue to be searched.
                    if PnNeighborPts_Num >= MinPts:
                        NeighborPts = numpy.concatenate((NeighborPts, PnNeighborPts))
                    # If Pn *doesn't* have enough neighbors, then it's a leaf point.
                    # Don't queue up it's neighbors as expansion points.
                    #else:
                        # Do nothing
                        #NeighborPts = NeighborPts
                # Advance to the next point in the FIFO queue.
                i += 1
            # We've finished growing cluster C!

        if self.precomputed == False:
            def regionQuery(D, P, eps, lables = None):
                """
                Find all points in dataset `D` within distance `eps` of point `P`.
                This function calculates the distance between a point P and every other
                point in the dataset, and then returns only those points which are within a
                threshold distance `eps`.
                """
                neighbors = []
                # For each point in the dataset...
                for Pn in range(0, len(D)):
                    # If the distance is below the threshold, add it to the neighbors list.
                    if self.metric == 'euclidian':
                        distance = numpy.linalg.norm(D[P] - D[Pn])
                        if  distance < eps:
                            neighbors.append(Pn)
                    elif self.metric == 'cosine': #data should be shifted to zero (zeromean)
                        D_p_norm = numpy.linalg.norm(D[P])
                        D_pn_norm = numpy.linalg.norm(D[Pn])
                        distance = (D_p_norm**2+D_pn_norm**2 - numpy.linalg.norm(D[P] - D[Pn])**2)/(2*D_p_norm*D_pn_norm)
                        if  distance > eps: # cosine is a decreasing function
                            neighbors.append(Pn)
                return len(neighbors), neighbors
            self.labels = MyDBSCAN(D, self.eps, self.MinPts)
        else:
            logger.info('Using precomputed distances')
            def regionQuery(D, P, eps, NeighborPts = None):
                """
                Usin precomputed distances from P to all other points.
                Return indeces of neighboors within eps
                """
                if NeighborPts is None: #initial version
                    if self.metric == 'euclidian':
                        neighbors = numpy.nonzero(D[P]<eps)[0]
                    elif self.metric == 'cosine':
                        neighbors = numpy.nonzero(D[P]>eps)[0]
                    return len(neighbors), neighbors
                else:
                    # due to economy of memmory return only unlabeled neighbors
                    if self.metric == 'euclidian':
                        neighbors = numpy.nonzero(D[P]<eps)[0]
                    elif self.metric == 'cosine':
                        neighbors = numpy.nonzero(D[P]>eps)[0]
                    labeled_NeighborPts_bool = numpy.zeros(len(D), dtype = numpy.bool)
                    labeled_NeighborPts_bool[NeighborPts] = True
                    query_NeighborPts_bool = numpy.zeros(len(D), dtype = numpy.bool)
                    query_NeighborPts_bool[neighbors] = True
                    not_labeled = numpy.nonzero((labeled_NeighborPts_bool + query_NeighborPts_bool)!=labeled_NeighborPts_bool)[0] #get only not labeles points
                    return len(neighbors), not_labeled
            self.labels = MyDBSCAN(D, self.eps, self.MinPts)

