#!/usr/bin/env python3

from sympy import *
from code_gen import *

# q: quaternion describing rotation from frame 1 to frame 2
# returns a rotation matrix derived form q which describes the same
# rotation
def quat2Rot(q, use_legacy_method=0):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    if use_legacy_method == 1:
        Rot = Matrix([[q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                    [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
                    [2*(q1*q3-q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
    else:
        # Use the simplified formula for unit quaternion to rotation matrix
        # as it produces a simpler and more stable EKF derivation given
        # the additional constraint: q0^2 + q1^2 + q2^2 + q3^2 = 1
        Rot = Matrix([[1 - 2*q2**2 - 2*q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                    [2*(q1*q2 + q0*q3), 1 - 2*q1**2 - 2*q3**2, 2*(q2*q3 - q0*q1)],
                    [2*(q1*q3-q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*q1**2 - 2*q2**2]])

    return Rot

def create_cov_matrix(i, j):
    if j >= i:
        return Symbol("P(" + str(i) + "," + str(j) + ")", real=True)
        # legacy array format
        # return Symbol("P[" + str(i) + "][" + str(j) + "]", real=True)
    else:
        return 0

def create_yaw_estimator_cov_matrix():
    # define a symbolic covariance matrix
    P = Matrix(3,3,create_cov_matrix)

    for index in range(3):
        for j in range(3):
            if index > j:
                P[index,j] = P[j,index]

    return P

def create_Tbs_matrix(i, j):
    return Symbol("Tbs(" + str(i) + "," + str(j) + ")", real=True)
    # legacy array format
    # return Symbol("Tbs[" + str(i) + "][" + str(j) + "]", real=True)

def quat_mult(p,q):
    r = Matrix([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]])

    return r

def quat_divide(p,q):
    r = Matrix([(q[0]*p[0] + q[1]*p[1] + q[2]*p[2] + q[3]*p[3]),
                (q[0]*p[1] - q[1]*p[0] - q[2]*p[3] + q[3]*p[2]),
                (q[0]*p[2] + q[1]*p[3] - q[2]*p[0] - q[3]*p[1]),
                (q[0]*p[3] - q[1]*p[2] + q[2]*p[1] - q[3]*p[0])])

    return r

def create_symmetric_cov_matrix():
    # define a symbolic covariance matrix
    P = Matrix(24,24,create_cov_matrix)

    for index in range(24):
        for j in range(24):
            if index > j:
                P[index,j] = P[j,index]

    return P

# generate equations for observation vector innovation variances
def generate_observation_vector_innovation_variances(P,state,observation,variance,n_obs):
    H = observation.jacobian(state)
    innovation_variance = zeros(n_obs,1)
    for index in range(n_obs):
        H[index,:] = Matrix([observation[index]]).jacobian(state)
        innovation_variance[index] = H[index,:] * P * H[index,:].T + Matrix([variance])

    IV_simple = cse(innovation_variance, symbols("IV0:1000"), optimizations='basic')

    return IV_simple

# generate equations for observation Jacobian and Kalman gain
def generate_observation_equations(P,state,observation,variance,varname="HK"):
    H = Matrix([observation]).jacobian(state)
    innov_var = H * P * H.T + Matrix([variance])
    assert(innov_var.shape[0] == 1)
    assert(innov_var.shape[1] == 1)
    K = P * H.T / innov_var[0,0]
    extension="0:1000"
    var_string = varname+extension
    HK_simple = cse(Matrix([H.transpose(), K]), symbols(var_string), optimizations='basic')

    return HK_simple

# generate equations for observation vector Jacobian and Kalman gain
# n_obs is the vector dimension and must be >= 2
def generate_observation_vector_equations(P,state,observation,variance,n_obs):
    K = zeros(24,n_obs)
    H = observation.jacobian(state)
    HK = zeros(n_obs*48,1)
    for index in range(n_obs):
        H[index,:] = Matrix([observation[index]]).jacobian(state)
        innov_var = H[index,:] * P * H[index,:].T + Matrix([variance])
        assert(innov_var.shape[0] == 1)
        assert(innov_var.shape[1] == 1)
        K[:,index] = P * H[index,:].T / innov_var[0,0]
        HK[index*48:(index+1)*48,0] = Matrix([H[index,:].transpose(), K[:,index]])

    HK_simple = cse(HK, symbols("HK0:1000"), optimizations='basic')

    return HK_simple

# write single observation equations to file
def write_equations_to_file(equations,code_generator_id,n_obs):
    if (n_obs < 1):
        return

    if (n_obs == 1):
        code_generator_id.print_string("Sub Expressions")
        code_generator_id.write_subexpressions(equations[0])
        code_generator_id.print_string("Observation Jacobians")
        code_generator_id.write_matrix(Matrix(equations[1][0][0:24]), "Hfusion", False, ".at<", ">()")
        code_generator_id.print_string("Kalman gains")
        code_generator_id.write_matrix(Matrix(equations[1][0][24:]), "Kfusion", False, "(", ")")
    else:
        code_generator_id.print_string("Sub Expressions")
        code_generator_id.write_subexpressions(equations[0])
        for axis_index in range(n_obs):
            start_index = axis_index*48
            code_generator_id.print_string("Observation Jacobians - axis %i" % axis_index)
            code_generator_id.write_matrix(Matrix(equations[1][0][start_index:start_index+24]), "Hfusion", False, ".at<", ">()")
            code_generator_id.print_string("Kalman gains - axis %i" % axis_index)
            code_generator_id.write_matrix(Matrix(equations[1][0][start_index+24:start_index+48]), "Kfusion", False, "(", ")")

    return

# derive equations for sequential fusion of optical flow measurements
def hagl_observation(P,state,pd,ptd):
    obs_var = symbols("R_HAGL", real=True) # optical flow line of sight rate measurement noise variance
    observation = ptd - pd
    equations = generate_observation_equations(P,state,observation,obs_var)
    hagl_code_generator = CodeGenerator("./generated/hagl_generated.cpp")
    write_equations_to_file(equations,hagl_code_generator,1)
    hagl_code_generator.close()

    return

# derive equations for sequential fusion of optical flow measurements
def optical_flow_observation(P,state,R_to_body,vx,vy,vz,pd,ptd):
    flow_code_generator = CodeGenerator("./generated/flow_generated.cpp")
    range = symbols("range", real=True) # range from camera focal point to ground along sensor Z axis
    obs_var = symbols("R_LOS", real=True) # optical flow line of sight rate measurement noise variance

    # Define rotation matrix from body to sensor frame
    Tsb = Matrix(3,3,create_Tbs_matrix)

    # define rotation from nav to sensor frame
    Tsn = Tsb * R_to_body

    hagl = ptd - pd
    range = hagl / Tsn[2,2]

    # Calculate earth relative velocity in a non-rotating sensor frame
    relVelSensor = Tsn * Matrix([vx,vy,vz])

    # Divide by range to get predicted angular LOS rates relative to X and Y
    # axes. Note these are rates in a non-rotating sensor frame
    losRateSensorX = +relVelSensor[1]/range
    losRateSensorY = -relVelSensor[0]/range

    # calculate the observation Jacobian and Kalman gains for the X axis
    equations = generate_observation_equations(P,state,losRateSensorX,obs_var)

    flow_code_generator.print_string("X Axis Equations")
    write_equations_to_file(equations,flow_code_generator,1)

    # calculate the observation Jacobian and Kalman gains for the Y axis
    equations = generate_observation_equations(P,state,losRateSensorY,obs_var)

    flow_code_generator.print_string("Y Axis Equations")
    write_equations_to_file(equations,flow_code_generator,1)

    flow_code_generator.close()

    # calculate a combined result for a possible reduction in operations, but will use more stack
    observation = Matrix([relVelSensor[1]/range,-relVelSensor[0]/range])
    equations = generate_observation_vector_equations(P,state,observation,obs_var,2)
    flow_code_generator_alt = CodeGenerator("./generated/flow_generated_alt.cpp")
    write_equations_to_file(equations,flow_code_generator_alt,2)
    flow_code_generator_alt.close()

    return

# Derive equations for sequential fusion of body frame velocity measurements
def body_frame_velocity_observation(P,state,R_to_body,vx,vy,vz):
    obs_var = symbols("R_VEL", real=True) # measurement noise variance

    # Calculate earth relative velocity in a non-rotating sensor frame
    vel_bf = R_to_body * Matrix([vx,vy,vz])

    vel_bf_code_generator = CodeGenerator("./generated/vel_bf_generated.cpp")
    axes = [0,1,2]
    H_obs = vel_bf.jacobian(state) # observation Jacobians
    K_gain = zeros(24,3)
    for index in axes:
        equations = generate_observation_equations(P,state,vel_bf[index],obs_var)

        vel_bf_code_generator.print_string("axis %i" % index)
        vel_bf_code_generator.write_subexpressions(equations[0])
        vel_bf_code_generator.write_matrix(Matrix(equations[1][0][0:24]), "H_VEL", False, "(", ")")
        vel_bf_code_generator.write_matrix(Matrix(equations[1][0][24:]), "Kfusion", False, "(", ")")

    vel_bf_code_generator.close()

    # calculate a combined result for a possible reduction in operations, but will use more stack
    equations = generate_observation_vector_equations(P,state,vel_bf,obs_var,3)

    vel_bf_code_generator_alt = CodeGenerator("./generated/vel_bf_generated_alt.cpp")
    write_equations_to_file(equations,vel_bf_code_generator_alt,3)
    vel_bf_code_generator_alt.close()

# derive equations for fusion of dual antenna yaw measurement
def gps_yaw_observation(P,state,R_to_body):
    obs_var = symbols("R_YAW", real=True) # measurement noise variance
    ant_yaw = symbols("ant_yaw", real=True) # yaw angle of antenna array axis wrt X body axis

    # define antenna vector in body frame
    ant_vec_bf = Matrix([cos(ant_yaw),sin(ant_yaw),0])

    # rotate into earth frame
    ant_vec_ef = R_to_body.T * ant_vec_bf

    # Calculate the yaw angle from the projection
    observation = atan(ant_vec_ef[1]/ant_vec_ef[0])

    equations = generate_observation_equations(P,state,observation,obs_var)

    gps_yaw_code_generator = CodeGenerator("./generated/gps_yaw_generated.cpp")
    write_equations_to_file(equations,gps_yaw_code_generator,1)
    gps_yaw_code_generator.close()

    return

# derive equations for fusion of declination
def declination_observation(P,state,ix,iy):
    obs_var = symbols("R_DECL", real=True) # measurement noise variance

    # the predicted measurement is the angle wrt magnetic north of the horizontal
    # component of the measured field
    observation = atan(iy/ix)

    equations = generate_observation_equations(P,state,observation,obs_var)

    mag_decl_code_generator = CodeGenerator("./generated/mag_decl_generated.cpp")
    write_equations_to_file(equations,mag_decl_code_generator,1)
    mag_decl_code_generator.close()

    return

# derive equations for fusion of lateral body acceleration (multirotors only)
def body_frame_accel_observation(P,state,R_to_body,vx,vy,vz,wx,wy):
    obs_var = symbols("R_ACC", real=True) # measurement noise variance
    Kaccx = symbols("Kaccx", real=True) # measurement noise variance
    Kaccy = symbols("Kaccy", real=True) # measurement noise variance

    # use relationship between airspeed along the X and Y body axis and the
    # drag to predict the lateral acceleration for a multirotor vehicle type
    # where propulsion forces are generated primarily along the Z body axis

    vrel = R_to_body*Matrix([vx-wx,vy-wy,vz]) # predicted wind relative velocity

    # Use this nonlinear model for the prediction in the implementation only
    # It uses a ballistic coefficient for each axis and a propeller momentum drag coefficient
    #
    # accXpred = -sign(vrel[0]) * vrel[0]*(0.5*rho*vrel[0]/BCoefX + MCoef) # predicted acceleration measured along X body axis
    # accYpred = -sign(vrel[1]) * vrel[1]*(0.5*rho*vrel[1]/BCoefY + MCoef) # predicted acceleration measured along Y body axis
    #
    # BcoefX and BcoefY have units of Kg/m^2
    # Mcoef has units of 1/s

    # Use a simple viscous drag model for the linear estimator equations
    # Use the the derivative from speed to acceleration averaged across the
    # speed range. This avoids the generation of a dirac function in the derivation
    # The nonlinear equation will be used to calculate the predicted measurement in implementation
    observation = Matrix([-Kaccx*vrel[0],-Kaccy*vrel[1]])

    acc_bf_code_generator  = CodeGenerator("./generated/acc_bf_generated.cpp")
    H = observation.jacobian(state)
    K = zeros(24,2)
    axes = [0,1]
    for index in axes:
        equations = generate_observation_equations(P,state,observation[index],obs_var)
        acc_bf_code_generator.print_string("Axis %i equations" % index)
        write_equations_to_file(equations,acc_bf_code_generator,1)

    acc_bf_code_generator.close()

    # calculate a combined result for a possible reduction in operations, but will use more stack
    equations = generate_observation_vector_equations(P,state,observation,obs_var,2)

    acc_bf_code_generator_alt  = CodeGenerator("./generated/acc_bf_generated_alt.cpp")
    write_equations_to_file(equations,acc_bf_code_generator_alt,3)
    acc_bf_code_generator_alt.close()

    return

# yaw fusion
def yaw_observation(P,state,R_to_earth):
    yaw_code_generator = CodeGenerator("./generated/yaw_generated.cpp")

    # Derive observation Jacobian for fusion of 321 sequence yaw measurement
    # Calculate the yaw (first rotation) angle from the 321 rotation sequence
    # Provide alternative angle that avoids singularity at +-pi/2 yaw
    angMeasA = atan(R_to_earth[1,0]/R_to_earth[0,0])
    H_YAW321_A = Matrix([angMeasA]).jacobian(state)
    H_YAW321_A_simple = cse(H_YAW321_A, symbols('SA0:200'))

    angMeasB = pi/2 - atan(R_to_earth[0,0]/R_to_earth[1,0])
    H_YAW321_B = Matrix([angMeasB]).jacobian(state)
    H_YAW321_B_simple = cse(H_YAW321_B, symbols('SB0:200'))

    yaw_code_generator.print_string("calculate 321 yaw observation matrix - option A")
    yaw_code_generator.write_subexpressions(H_YAW321_A_simple[0])
    yaw_code_generator.write_matrix(Matrix(H_YAW321_A_simple[1]).T, "H_YAW", False, ".at<", ">()")

    yaw_code_generator.print_string("calculate 321 yaw observation matrix - option B")
    yaw_code_generator.write_subexpressions(H_YAW321_B_simple[0])
    yaw_code_generator.write_matrix(Matrix(H_YAW321_B_simple[1]).T, "H_YAW", False, ".at<", ">()")

    # Derive observation Jacobian for fusion of 312 sequence yaw measurement
    # Calculate the yaw (first rotation) angle from an Euler 312 sequence
    # Provide alternative angle that avoids singularity at +-pi/2 yaw
    angMeasA = atan(-R_to_earth[0,1]/R_to_earth[1,1])
    H_YAW312_A = Matrix([angMeasA]).jacobian(state)
    H_YAW312_A_simple = cse(H_YAW312_A, symbols('SA0:200'))

    angMeasB = pi/2 - atan(-R_to_earth[1,1]/R_to_earth[0,1])
    H_YAW312_B = Matrix([angMeasB]).jacobian(state)
    H_YAW312_B_simple = cse(H_YAW312_B, symbols('SB0:200'))

    yaw_code_generator.print_string("calculate 312 yaw observation matrix - option A")
    yaw_code_generator.write_subexpressions(H_YAW312_A_simple[0])
    yaw_code_generator.write_matrix(Matrix(H_YAW312_A_simple[1]).T, "H_YAW", False, ".at<", ">()")

    yaw_code_generator.print_string("calculate 312 yaw observation matrix - option B")
    yaw_code_generator.write_subexpressions(H_YAW312_B_simple[0])
    yaw_code_generator.write_matrix(Matrix(H_YAW312_B_simple[1]).T, "H_YAW", False, ".at<", ">()")

    yaw_code_generator.close()

    return

# 3D magnetometer fusion
def mag_observation_variance(P,state,R_to_body,i,ib):
    obs_var = symbols("R_MAG", real=True)  # magnetometer measurement noise variance

    m_mag = R_to_body * i + ib

    # separate calculation of innovation variance equations for the y and z axes
    m_mag[0]=0
    innov_var_equations = generate_observation_vector_innovation_variances(P,state,m_mag,obs_var,3)
    mag_innov_var_code_generator = CodeGenerator("./generated/3Dmag_innov_var_generated.cpp")
    write_equations_to_file(innov_var_equations,mag_innov_var_code_generator,3)
    mag_innov_var_code_generator.close()

    return

# 3D magnetometer fusion
def mag_observation(P,state,R_to_body,i,ib):
    obs_var = symbols("R_MAG", real=True)  # magnetometer measurement noise variance

    m_mag = R_to_body * i + ib

    # calculate a separate set of equations for each axis
    mag_code_generator = CodeGenerator("./generated/3Dmag_generated.cpp")

    axes = [0,1,2]
    label="HK"
    for index in axes:
        if (index==0):
            label="HKX"
        elif (index==1):
            label="HKY"
        elif (index==2):
            label="HKZ"
        else:
            return
        equations = generate_observation_equations(P,state,m_mag[index],obs_var,varname=label)
        mag_code_generator.print_string("Axis %i equations" % index)
        write_equations_to_file(equations,mag_code_generator,1)

    mag_code_generator.close()

    # calculate a combined set of equations for a possible reduction in operations, but will use slighlty more stack
    equations = generate_observation_vector_equations(P,state,m_mag,obs_var,3)

    mag_code_generator_alt  = CodeGenerator("./generated/3Dmag_generated_alt.cpp")
    write_equations_to_file(equations,mag_code_generator_alt,3)
    mag_code_generator_alt.close()

    return

# airspeed fusion
def tas_observation(P,state,vx,vy,vz,wx,wy):
    obs_var = symbols("R_TAS", real=True) # true airspeed measurement noise variance

    observation = sqrt((vx-wx)*(vx-wx)+(vy-wy)*(vy-wy)+vz*vz)

    equations = generate_observation_equations(P,state,observation,obs_var)

    tas_code_generator = CodeGenerator("./generated/tas_generated.cpp")
    write_equations_to_file(equations,tas_code_generator,1)
    tas_code_generator.close()

    return

# sideslip fusion
def beta_observation(P,state,R_to_body,vx,vy,vz,wx,wy):
    obs_var = symbols("R_BETA", real=True) # sideslip measurement noise variance

    v_rel_ef = Matrix([vx-wx,vy-wy,vz])
    v_rel_bf = R_to_body * v_rel_ef
    observation = v_rel_bf[1]/v_rel_bf[0]

    equations = generate_observation_equations(P,state,observation,obs_var)

    beta_code_generator = CodeGenerator("./generated/beta_generated.cpp")
    write_equations_to_file(equations,beta_code_generator,1)
    beta_code_generator.close()

    return

# yaw estimator prediction and observation code
def yaw_estimator():
    dt = symbols("dt", real=True)  # dt (sec)
    psi = symbols("psi", real=True)  # yaw angle of body frame wrt earth frame
    vn, ve = symbols("vn ve", real=True)  # velocity in world frame (north/east) - m/sec
    daz = symbols("daz", real=True)  # IMU z axis delta angle measurement in body axes - rad
    dazVar = symbols("dazVar", real=True) # IMU Z axis delta angle measurement variance (rad^2)
    dvx, dvy = symbols("dvx dvy", real=True)  # IMU x and y axis delta velocity measurement in body axes - m/sec
    dvxVar, dvyVar = symbols("dvxVar dvyVar", real=True)   # IMU x and y axis delta velocity measurement variance (m/s)^2

    # derive the body to nav direction transformation matrix
    Tbn = Matrix([[cos(psi) , -sin(psi)],
                [sin(psi) ,  cos(psi)]])

    # attitude update equation
    psiNew = psi + daz

    # velocity update equations
    velNew = Matrix([vn,ve]) + Tbn*Matrix([dvx,dvy])

    # Define the state vectors
    stateVector = Matrix([vn,ve,psi])

    # Define vector of process equations
    newStateVector = Matrix([velNew,psiNew])

    # Calculate state transition matrix
    F = newStateVector.jacobian(stateVector)

    # Derive the covariance prediction equations
    # Error growth in the inertial solution is assumed to be driven by 'noise' in the delta angles and
    # velocities, after bias effects have been removed.

    # derive the control(disturbance) influence matrix from IMU noise to state noise
    G = newStateVector.jacobian(Matrix([dvx,dvy,daz]))

    # derive the state error matrix
    distMatrix = Matrix([[dvxVar , 0 , 0],
                        [0 , dvyVar , 0],
                        [0 , 0 , dazVar]])

    Q = G * distMatrix * G.T

    # propagate covariance matrix
    P = create_yaw_estimator_cov_matrix()

    P_new = F * P * F.T + Q

    P_new_simple = cse(P_new, symbols("S0:1000"), optimizations='basic')

    yaw_estimator_covariance_generator = CodeGenerator("./generated/yaw_estimator_covariance_prediction_generated.cpp")
    yaw_estimator_covariance_generator.print_string("Equations for covariance matrix prediction")
    yaw_estimator_covariance_generator.write_subexpressions(P_new_simple[0])
    yaw_estimator_covariance_generator.write_matrix(Matrix(P_new_simple[1]), "_ekf_gsf[model_index].P", True)
    yaw_estimator_covariance_generator.close()

    # derive the covariance update equation for a NE velocity observation
    velObsVar = symbols("velObsVar", real=True) # velocity observation variance (m/s)^2
    H = Matrix([[1,0,0],
                [0,1,0]])

    R = Matrix([[velObsVar , 0],
                [0 , velObsVar]])

    S = H * P * H.T + R
    S_det_inv = 1 / S.det()
    S_inv = S.inv()
    K = (P * H.T) * S_inv
    P_new = P - K * S * K.T

    # optimize code
    t, [S_det_inv_s, S_inv_s, K_s, P_new_s] = cse([S_det_inv, S_inv, K, P_new], symbols("t0:1000"), optimizations='basic')

    yaw_estimator_observation_generator = CodeGenerator("./generated/yaw_estimator_measurement_update_generated.cpp")
    yaw_estimator_observation_generator.print_string("Intermediate variables")
    yaw_estimator_observation_generator.write_subexpressions(t)
    yaw_estimator_observation_generator.print_string("Equations for NE velocity innovation variance's determinante inverse")
    yaw_estimator_observation_generator.write_matrix(Matrix([[S_det_inv_s]]), "_ekf_gsf[model_index].S_det_inverse", False)
    yaw_estimator_observation_generator.print_string("Equations for NE velocity innovation variance inverse")
    yaw_estimator_observation_generator.write_matrix(Matrix(S_inv_s), "_ekf_gsf[model_index].S_inverse", True)
    yaw_estimator_observation_generator.print_string("Equations for NE velocity Kalman gain")
    yaw_estimator_observation_generator.write_matrix(Matrix(K_s), "K", False)
    yaw_estimator_observation_generator.print_string("Equations for covariance matrix update")
    yaw_estimator_observation_generator.write_matrix(Matrix(P_new_s), "_ekf_gsf[model_index].P", True)
    yaw_estimator_observation_generator.close()

def generate_code():
    print('Starting code generation:')
    print('Creating symbolic variables ...')

    # state prediction equations

    # define the measured Delta angle and delta velocity vectors
    dax, day, daz = symbols("dax, day, daz", real=True)
    dAngMeas = Matrix([dax, day, daz])
    dvx, dvy, dvz = symbols("dvx, dvy, dvz", real=True)
    dVelMeas = Matrix([dvx, dvy, dvz])

    # define the IMU bias errors and scale factor
    dax_b, day_b, daz_b = symbols("dax_b, day_b, daz_b", real=True)
    dAngBias = Matrix([dax_b, day_b, daz_b])
    dvx_b, dvy_b, dvz_b = symbols("dvx_b, dvy_b, dvz_b", real=True)
    dVelBias = Matrix([dvx_b, dvy_b, dvz_b])

    # define the quaternion rotation vector for the state estimate
    q0, q1, q2, q3 = symbols("q0, q1, q2, q3", real=True)
    estQuat = Matrix([q0, q1, q2, q3])

    # define the attitude error rotation vector, where error = truth - estimate
    rotErrX, rotErrY, rotErrZ = symbols("rotErrX, rotErrY, rotErrZ", real=True)
    errRotVec = Matrix([rotErrX, rotErrY, rotErrZ])

    # define the attitude error quaternion using a first order linearisation
    errQuat = Matrix([1,errRotVec*0.5])

    # Define the truth quaternion as the estimate + error
    truthQuat = quat_mult(estQuat, errQuat)

    # derive the truth body to nav direction cosine matrix
    Tbn = quat2Rot(truthQuat)

    # define the truth delta angle
    # ignore coning compensation as these effects are negligible in terms of
    # covariance growth for our application and grade of sensor
    daxNoise, dayNoise, dazNoise = symbols("daxVar, dayVar, dazVar", real=True)
    dAngTruth = dAngMeas - dAngBias - Matrix([daxNoise, dayNoise, dazNoise])

    # define the attitude update equations
    # use a first order expansion of rotation to calculate the quaternion increment
    # acceptable for propagation of covariances
    deltaQuat = Matrix([1,dAngTruth*0.5])

    truthQuatNew = quat_mult(truthQuat,deltaQuat)

    # calculate the updated attitude error quaternion with respect to the previous estimate
    errQuatNew = quat_divide(truthQuatNew,estQuat)

    # change to a rotaton vector - this is the error rotation vector updated state
    errRotVecNew = Matrix([2*errQuatNew[1], 2*errQuatNew[2], 2*errQuatNew[3]])

    # Define the truth delta velocity -ignore sculling and transport rate
    # corrections as these negligible are in terms of covariance growth for our
    # application and grade of sensor
    dvxNoise, dvyNoise, dvzNoise = symbols("dvxVar, dvyVar, dvzVar", real=True)
    dVelTruth = dVelMeas - dVelBias - Matrix([dvxNoise, dvyNoise, dvzNoise])

    # define the velocity update equations
    # ignore coriolis terms for linearisation purposes
    vn,ve,vd = symbols("vn,ve,vd", real=True)
    vel = Matrix([vn,ve,vd])
    dt = symbols("dt", real=True)  # dt
    g = symbols("g", real=True) # gravity constant
    velNew = vel + Matrix([0, 0, g*dt]) + Tbn*dVelTruth

    # define the position update equations
    pn, pe, pd = symbols("pn, pe, pd", real=True)
    pos = Matrix([pn, pe, pd])
    posNew = pos + vel*dt

    # define the IMU bias error states
    dax_b, day_b, daz_b = symbols("dax_b, day_b, daz_b", real=True)
    delAngBias = Matrix([dax_b, day_b, daz_b])
    dvx_b, dvy_b, dvz_b = symbols("dvx_b, dvy_b, dvz_b", real=True)
    delVelBias = Matrix([dvx_b, dvy_b, dvz_b])

    # define the earth magnetic field states
    magN, magE, magD = symbols("magN magE magD", real=True)
    magField = Matrix([magN, magE, magD])

    # define the magnetic sensor bias states
    magX, magY, magZ = symbols("magX, magY, magZ", real=True)
    magBias = Matrix([magX, magY, magZ])

    # define the wind velocity states
    vwn, vwe = symbols("vwn, vwe", real=True)
    velWind = Matrix([vwn, vwe])

    # define the terrain vertical position state
    ptd = symbols("ptd", real=True)

    # Define the state vector & number of states
    state = Matrix([errRotVec,vel,pos,delAngBias,delVelBias,magField,magBias,velWind,ptd])

    # Define vector of process equations
    state_new = Matrix([errRotVecNew,velNew,posNew,delAngBias,delVelBias,magField,magBias,velWind,ptd])

    # IMU input noise influence matrix
    G = state_new.jacobian(Matrix([daxNoise, dayNoise, dazNoise, dvxNoise, dvyNoise, dvzNoise]))

    # set the rotation error states to zero
    G.subs(rotErrX,0)
    G.subs(rotErrY,0)
    G.subs(rotErrZ,0)

    # remove the disturbance noise from the process equations as it is only
    # needed when calculating the disturbance influence matrix
    state_new.subs(daxNoise,0)
    state_new.subs(dayNoise,0)
    state_new.subs(dazNoise,0)
    state_new.subs(dvxNoise,0)
    state_new.subs(dvyNoise,0)
    state_new.subs(dvzNoise,0)

    print('Computing state propagation jacobian ...')

    # state transition matrix
    A = state_new.jacobian(state)

    # set the rotation error states to zero
    A.subs(rotErrX,0)
    A.subs(rotErrY,0)
    A.subs(rotErrZ,0)

    # IMU input noise variance matrix
    imu_noise_variance = Matrix.diag(daxNoise, dayNoise, dazNoise, dvxNoise, dvyNoise, dvzNoise)

    P = create_symmetric_cov_matrix()

    print('Computing covariance propagation ...')
    P_new = A * P * A.T + G * imu_noise_variance * G.T

    for index in range(24):
        for j in range(24):
            if index > j:
                P_new[index,j] = 0

    print('Simplifying covariance propagation ...')
    P_new_simple = cse(P_new, symbols("PS0:400"), optimizations='basic')

    print('Writing covariance propagation to file ...')
    cov_code_generator = CodeGenerator("./generated/covariance_generated.cpp")
    cov_code_generator.print_string("Equations for covariance matrix prediction, without process noise!")
    cov_code_generator.write_subexpressions(P_new_simple[0])
    cov_code_generator.write_matrix(Matrix(P_new_simple[1]), "nextP", True, "(", ")")

    cov_code_generator.close()

    # use legacy quaternion to rotation matrix conversion for observaton equation as it gives
    # simpler equations
    R_to_earth = quat2Rot(truthQuat,1)
    R_to_body = R_to_earth.T

    # derive autocode for observation methods
    print('Generating heading observation code ...')
    yaw_observation(P,state,R_to_earth)
    print('Generating gps heading observation code ...')
    gps_yaw_observation(P,state,R_to_body)
    print('Generating mag observation code ...')
    mag_observation_variance(P,state,R_to_body,magField,magBias)
    mag_observation(P,state,R_to_body,magField,magBias)
    print('Generating declination observation code ...')
    declination_observation(P,state,magN,magE)
    print('Generating airspeed observation code ...')
    tas_observation(P,state,vn,ve,vd,vwn,vwe)
    print('Generating sideslip observation code ...')
    beta_observation(P,state,R_to_body,vn,ve,vd,vwn,vwe)
    print('Generating optical flow observation code ...')
    optical_flow_observation(P,state,R_to_body,vn,ve,vd,pd,ptd)
    print('Generating HAGL observation code ...')
    hagl_observation(P,state,pd,ptd)
    print('Generating body frame velocity observation code ...')
    body_frame_velocity_observation(P,state,R_to_body,vn,ve,vd)
    print('Generating body frame acceleration observation code ...')
    body_frame_accel_observation(P,state,R_to_body,vn,ve,vd,vwn,vwe)
    print('Generating yaw estimator code ...')
    yaw_estimator()
    print('Code generation finished!')


if __name__ == "__main__":
    generate_code()
