import numpy as np 


def center(points):
    
    # return center of ROI
    x = np.float32((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0)
    y = np.float32((points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)

def ignore(x):
    
    # Do Nothing
    pass

def particleEvaluator(back_proj, particle):
    
    # Function to Evaluate Particles in Each Frame
    return back_proj[particle[1],particle[0]]

def resample(weights):
    
    # Function to Resample Particles according to Weights
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices
