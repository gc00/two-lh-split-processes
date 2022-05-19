# two-lh-split-processes

# Rough Notes
Kernel Loader should be able to load two or more lower halves. Once it's done loading all lower halve's, it should then load upper-half and pass the control to the upper half.
These lower halves should be dynamically linked and their linked libraries should have constructors.

We would also want to control where to put each lower half. So, none of the half's collide with each other.

We'll need to build to stub libraries for each lower half.
