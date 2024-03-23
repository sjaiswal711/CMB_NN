import time
import numpy as np 
import healpy as hp
from tqdm import tqdm
import multiprocessing

def RVector(t):
    theta1 = 7.5*np.pi / 180
    theta2 = 85*np.pi / 180
    w1 = 2*np.pi  #rad/min
    w2 = 2*w1 #rad/min
    w3 = 0.000011954  #rad/min

    A=[[np.cos(w1*t),np.sin(w1*t),0],
       [-np.sin(w1*t),np.cos(w1*t),0],
       [0,0,1]]

    B=[[1,0,0],
       [0,np.cos(w2*t),np.sin(w2*t)],
       [0,-np.sin(w2*t),np.cos(w2*t)]]

    C=[[np.cos(theta1),0,np.sin(theta1)],
       [0,1,0],
       [-np.sin(theta1),0,np.cos(theta1)]]

    D=[[np.cos(theta2)],
       [np.sin(theta2)*np.cos(w3*t)],
       [np.sin(theta2)*np.sin(w3*t)]]

    result1 = np.matmul(A,B)
    result2 = np.matmul(result1,C)
    result = np.matmul(result2,D)
    result = np.matrix.transpose(result)
    return result.flatten()

def SVector(t):
    theta1 = 7.5*np.pi / 180
    theta2 = 0
    w1 = 2*np.pi  #rad/min
    w2 = 2*w1 #rad/min
    w3 = 0.000011954  #rad/min

    A=[[np.cos(w1*t),np.sin(w1*t),0],
       [-np.sin(w1*t),np.cos(w1*t),0],
       [0,0,1]]

    B=[[1,0,0],
       [0,np.cos(w2*t),np.sin(w2*t)],
       [0,-np.sin(w2*t),np.cos(w2*t)]]

    C=[[np.cos(theta1),0,np.sin(theta1)],
       [0,1,0],
       [-np.sin(theta1),0,np.cos(theta1)]]

    D=[[np.cos(theta2)],
       [-np.sin(theta2)*np.cos(w3*t)],
       [-np.sin(theta2)*np.sin(w3*t)]]

    result1 = np.dot(A,B)
    result2 = np.dot(C,D)
    result = np.dot(result1,result2)
    result = np.matrix.transpose(result)
    return result.flatten()

#  Angle between two vector

def angle_vec(A, B):
    dot_product = np.dot(A, B) 
    mag_A = np.linalg.norm(A)
    mag_B = np.linalg.norm(B)
    if (mag_A * mag_B) == 0:
        return 0 # To handle the case where one the vector becomes Zeros(R_i == Rc)
    cos_theta = dot_product / (mag_A * mag_B)
    angle = np.arccos(cos_theta)
    return angle

def angle(vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  clipped_dp = np.clip(dot_product, -1.0, 1.0) # Clip dot_product to the valid range for arccos to avoid NaNs
  angle = np.arccos(clipped_dp)
  return angle


theta1 = 7.5*np.pi / 180
theta2 = 85*np.pi / 180
w1 = 2*np.pi  #rad/min
w2 = 2*w1 #rad/min
w3 = 0.000011954  #rad/min

nside=1024
npix = 12*nside**2

# time_step=scan_time
scan_time = np.sqrt(4*np.pi/npix)/w1
fwhm_x = np.radians(10) 
fwhm_y = np.radians(15)
sigma_x = fwhm_x / np.sqrt(8 * np.log(2)) 
sigma_y = fwhm_y / np.sqrt(8 * np.log(2))

temperature_map = hp.read_map("input_map.fits")

def process_time_step(time_step):
    
    t = time_step  

    # 1. Calculate R(t) and S(t) vectors
    R = RVector(t)
    S = SVector(t)

    # 2. Calculate pixel number along R(t) vector (ring format)
    pix_ring = hp.vec2pix(nside, R[0], R[1], R[2], nest=False)


    #3. Calculate Z, I and N (N = I for phi = 0)
    Z_t = np.cross(R,S)
    I_t = np.cross(R, Z_t)
    N_t = I_t
    
    # 4. Find neighboring pixels in RING format
    Rc = hp.pix2vec(nside,pix_ring,nest=False)
    neighbours = hp.query_disc(nside, Rc , radius=(3*sigma_y))

    # 5. angular separation between central pixel and neighbouring pixels
    x = np.zeros_like(neighbours, dtype=float)
    y = np.zeros(len(neighbours))

    for i, neighbour_pix in enumerate(neighbours):
        
        R_i = hp.pix2vec(nside,neighbour_pix,nest=False)
        theta_i = angle(Rc, R_i)

        # 6. A_i = line joining central pixel and neighbour pixel
        R_i = hp.pix2vec(nside,neighbour_pix,nest=False)
        A_i = np.array(Rc)-np.array(R_i)
        # print("A_i = ",A_i,"\nN_t = ",N_t)
        
        # 7. angle between N & A_i
        alpha_i = angle_vec(A_i, N_t,theta_i,R_i) 
        # print("alpha_i=",(alpha_i))
        # 8. x_i and y_i
        x[i] = theta_i * np.cos(alpha_i)
        y[i] = theta_i * np.sin(alpha_i)
        # print(x[i],theta_i * np.cos(alpha_i),y[i])

    # 9. Retrieve temperatures of neighboring pixels
    neighbor_temperatures = temperature_map[neighbours]
    # 10. Apply elliptical convolution
    convolved_temperature = np.sum(neighbor_temperatures * np.exp(-x**2 / (2 * sigma_x**2) -y**2 / (2 * sigma_y**2))) / np.sum(np.exp(-x**2 / (2 * sigma_x**2) -y**2 / (2 * sigma_y**2)))

    return int(pix_ring),convolved_temperature


start = time.time()

start_time=0
duration = 6 #in min (one month)
steps = int(duration / scan_time)
# steps = 1

time_periods = np.linspace(start_time, start_time + duration,steps)
time_periods_iterator = tqdm(time_periods, desc="Processing", total=len(time_periods))
def parallel_execution(chunk):
    results = []
    for time_period in tqdm(chunk, desc="Processing"):
        pixel,temperature = process_time_step(time_period)
        results.append((time_period,pixel,temperature))
    return results

start = time.time()

# Split the time_periods array into chunks for parallel processing
chunks = np.array_split(time_periods, 16)

# Using multiprocessing for parallel execution
with multiprocessing.Pool(processes=16) as pool:
    results = pool.map(parallel_execution, chunks)

print("result processing")
# Flatten the results list of lists
results = [item for sublist in results for item in sublist]

# Write results to the file
# file_path = 'check.dat'
file_path = 'month1.dat'
np.savetxt(file_path, results, fmt='%.4f %d %.16f ')
print(f"Results saved to {file_path}")
