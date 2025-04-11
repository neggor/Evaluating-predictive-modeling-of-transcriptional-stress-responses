## Here the idea is to make 
# a argparse for running each type of model with quite a lot of options
# and then the experiment is just with one particular set of parameters.
# For instance, for the CNN, n models for removing exons, n models for longer sequences and so on...
# Essentially the functionality here should be gooe enough to actually use in a hyperparameter tuning setting.
# Because the idea would be to use the hyperparameters from a json in the case of a CNN.
# But also make a json for agroNT.

# the the idea would be to make another script to do the hyperpamrameter tuning by generating the json files and obtaining 
# the performance.