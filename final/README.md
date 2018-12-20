# ML_Project_2 README

############### File Structure ###############
1. featureVec, objectiveVec : 
- Contain 500 rows of features (6) and objectives (7)

2. build_model.py : 
- Creates the ANN model using the samples and saves the model

3. optimizer.py : 
- Runs an SQP optimization algorithm to find the optimum features that minimize the norm of the objective vector.

4. model.h5 model_1.h5 , model_2.h5 : 
- Three different models corresponding to three different runs (with the exact same parameters) of build_model.py.
- You can use them in optimizer.py to check the results of the optimization using these three runs.		
