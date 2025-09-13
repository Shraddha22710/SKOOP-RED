# utils/config.py


# Design choices 
# ----------------------------------------------------------------
USE_DMD = True            # if True, use SVD-based DMD estimator (experimental)
FEATURE_ON_RESIDUAL = True  # if True, features computed from residual x - D(x); else from x
APPLY_FEATURE_SCALING = True # scale features across the window before regression 
USE_PREDICTIVE_CHECK = False # use short-horizon predictive check 
PREDICT_HORIZON = 1
PREDICT_THRESHOLD = 1.05
BETA_PAPER = 2.0          
RIDGE_ALPHA = 1e-6        
COND_GUARD = 1e12        
# ----------------------------------------------------------------
