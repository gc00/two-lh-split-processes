# two-lh-split-processes

# Rough Notes
Kernel Loader should be able to load two or more lower halves. Once it's done loading all lower halve's, it should then load upper-half and pass the control to the upper half.
These lower halves should be dynamically linked and their linked libraries should have constructors.
