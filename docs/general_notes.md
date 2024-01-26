## Structure of cloud points

When we have a cloud of points on the object that we want to classify, we concatenate all point's coordinates to get a point in high-dimensional space. And this point will represent the whole object.

Notice, that if just permute the points in the cloud we will get a totally different point in a high-dimensional space. This should break our algorithm, because it assumes that similar object are represented by relatively close points in the space, but if we permute a coordinates of some point we are not getting close point even though the cloud points lay on the same object.

That means, that we should have some correspondence between cloud points on the objects.
