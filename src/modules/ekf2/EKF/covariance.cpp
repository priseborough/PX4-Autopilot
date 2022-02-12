/****************************************************************************
 *
 *   Copyright (c) 2015 Estimation and Control Library (ECL). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name ECL nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file covariance.cpp
 * Contains functions for initialising, predicting and updating the state
 * covariance matrix
 * equations generated using EKF/python/ekf_derivation/main.py
 *
 * @author Roman Bast <bastroman@gmail.com>
 *
 */

#include "ekf.h"
#include "utils.hpp"

#include <math.h>
#include <mathlib/mathlib.h>

// Sets initial values for the covariance matrix
// Do not call before quaternion states have been initialised
void Ekf::initialiseCovariance()
{
	P.zero();

	_delta_angle_bias_var_accum.setZero();
	_delta_vel_bias_var_accum.setZero();

	const float dt = _dt_ekf_avg;

	resetQuatCov();

	// velocity
	P(4,4) = sq(fmaxf(_params.gps_vel_noise, 0.01f));
	P(5,5) = P(4,4);
	P(6,6) = sq(1.5f) * P(4,4);

	// position
	P(7,7) = sq(fmaxf(_params.gps_pos_noise, 0.01f));
	P(8,8) = P(7,7);

	if (_control_status.flags.rng_hgt) {
		P(9,9) = sq(fmaxf(_params.range_noise, 0.01f));

	} else if (_control_status.flags.gps_hgt) {
		P(9,9) = getGpsHeightVariance();

	} else {
		P(9,9) = sq(fmaxf(_params.baro_noise, 0.01f));
	}

	// gyro bias
	P(10,10) = sq(_params.switch_on_gyro_bias * dt);
	P(11,11) = P(10,10);
	P(12,12) = P(10,10);

	// accel bias
	_prev_dvel_bias_var(0) = P(13,13) = sq(_params.switch_on_accel_bias * dt);
	_prev_dvel_bias_var(1) = P(14,14) = P(13,13);
	_prev_dvel_bias_var(2) = P(15,15) = P(13,13);

	resetMagCov();

	// wind
	P(22,22) = sq(_params.initial_wind_uncertainty);
	P(23,23) = P(22,22);

}

void Ekf::predictCovariance()
{
	// assign intermediate state variables

	const float rotErrX = 0.0f;
	const float rotErrY = 0.0f;
	const float rotErrZ = 0.0f;

	const float &q0 = _state.quat_nominal(0);
	const float &q1 = _state.quat_nominal(1);
	const float &q2 = _state.quat_nominal(2);
	const float &q3 = _state.quat_nominal(3);

	const float &dax = _imu_sample_delayed.delta_ang(0);
	const float &day = _imu_sample_delayed.delta_ang(1);
	const float &daz = _imu_sample_delayed.delta_ang(2);

	const float &dvx = _imu_sample_delayed.delta_vel(0);
	const float &dvy = _imu_sample_delayed.delta_vel(1);
	const float &dvz = _imu_sample_delayed.delta_vel(2);

	const float &dax_b = _state.delta_ang_bias(0);
	const float &day_b = _state.delta_ang_bias(1);
	const float &daz_b = _state.delta_ang_bias(2);

	const float &dvx_b = _state.delta_vel_bias(0);
	const float &dvy_b = _state.delta_vel_bias(1);
	const float &dvz_b = _state.delta_vel_bias(2);

	// Use average update interval to reduce accumulated covariance prediction errors due to small single frame dt values
	const float dt = _dt_ekf_avg;
	const float dt_inv = 1.f / dt;

	// convert rate of change of rate gyro bias (rad/s**2) as specified by the parameter to an expected change in delta angle (rad) since the last update
	const float d_ang_bias_sig = dt * dt * math::constrain(_params.gyro_bias_p_noise, 0.0f, 1.0f);

	// convert rate of change of accelerometer bias (m/s**3) as specified by the parameter to an expected change in delta velocity (m/s) since the last update
	const float d_vel_bias_sig = dt * dt * math::constrain(_params.accel_bias_p_noise, 0.0f, 1.0f);

	// inhibit learning of imu accel bias if the manoeuvre levels are too high to protect against the effect of sensor nonlinearities or bad accel data is detected
	// xy accel bias learning is also disabled on ground as those states are poorly observable when perpendicular to the gravity vector
	const float alpha = math::constrain((dt / _params.acc_bias_learn_tc), 0.0f, 1.0f);
	const float beta = 1.0f - alpha;
	_ang_rate_magnitude_filt = fmaxf(dt_inv * _imu_sample_delayed.delta_ang.norm(), beta * _ang_rate_magnitude_filt);
	_accel_magnitude_filt = fmaxf(dt_inv * _imu_sample_delayed.delta_vel.norm(), beta * _accel_magnitude_filt);
	_accel_vec_filt = alpha * dt_inv * _imu_sample_delayed.delta_vel + beta * _accel_vec_filt;

	const bool is_manoeuvre_level_high = _ang_rate_magnitude_filt > _params.acc_bias_learn_gyr_lim
					     || _accel_magnitude_filt > _params.acc_bias_learn_acc_lim;

	const bool do_inhibit_all_axes = (_params.fusion_mode & MASK_INHIBIT_ACC_BIAS)
					 || is_manoeuvre_level_high
					 || _fault_status.flags.bad_acc_vertical;

	for (unsigned stateIndex = 13; stateIndex <= 15; stateIndex++) {
		const unsigned index = stateIndex - 13;

		const bool do_inhibit_axis = do_inhibit_all_axes || _imu_sample_delayed.delta_vel_clipping[index];

		if (do_inhibit_axis) {
			// store the bias state variances to be reinstated later
			if (!_accel_bias_inhibit[index]) {
				_prev_dvel_bias_var(index) = P(stateIndex, stateIndex);
				_accel_bias_inhibit[index] = true;
			}

		} else {
			if (_accel_bias_inhibit[index]) {
				// reinstate the bias state variances
				P(stateIndex, stateIndex) = _prev_dvel_bias_var(index);
				_accel_bias_inhibit[index] = false;
			}
		}
	}

	// Don't continue to grow the earth field variances if they are becoming too large or we are not doing 3-axis fusion as this can make the covariance matrix badly conditioned
	float mag_I_sig;

	if (_control_status.flags.mag_3D && (P(16, 16) + P(17, 17) + P(18, 18)) < 0.1f) {
		mag_I_sig = dt * math::constrain(_params.mage_p_noise, 0.0f, 1.0f);

	} else {
		mag_I_sig = 0.0f;
	}

	// Don't continue to grow the body field variances if they is becoming too large or we are not doing 3-axis fusion as this can make the covariance matrix badly conditioned
	float mag_B_sig;

	if (_control_status.flags.mag_3D && (P(19, 19) + P(20, 20) + P(21, 21)) < 0.1f) {
		mag_B_sig = dt * math::constrain(_params.magb_p_noise, 0.0f, 1.0f);

	} else {
		mag_B_sig = 0.0f;
	}

	float wind_vel_sig;

	// Calculate low pass filtered height rate
	float alpha_height_rate_lpf = 0.1f * dt; // 10 seconds time constant
	_height_rate_lpf = _height_rate_lpf * (1.0f - alpha_height_rate_lpf) + _state.vel(2) * alpha_height_rate_lpf;

	// Don't continue to grow wind velocity state variances if they are becoming too large or we are not using wind velocity states as this can make the covariance matrix badly conditioned
	if (_control_status.flags.wind && (P(22,22) + P(23,23)) < sq(_params.initial_wind_uncertainty)) {
		wind_vel_sig = dt * math::constrain(_params.wind_vel_p_noise, 0.0f, 1.0f) * (1.0f + _params.wind_vel_p_noise_scaler * fabsf(_height_rate_lpf));

	} else {
		wind_vel_sig = 0.0f;
	}

	// compute noise variance for stationary processes
	Vector25f process_noise;

	// Construct the process noise variance diagonal for those states with a stationary process model
	// These are kinematic states and their error growth is controlled separately by the IMU noise variances

	// delta angle bias states
	process_noise.slice<3, 1>(10, 0) = sq(d_ang_bias_sig);
	// delta_velocity bias states
	process_noise.slice<3, 1>(13, 0) = sq(d_vel_bias_sig);
	// earth frame magnetic field states
	process_noise.slice<3, 1>(16, 0) = sq(mag_I_sig);
	// body frame magnetic field states
	process_noise.slice<3, 1>(19, 0) = sq(mag_B_sig);
	// wind velocity states
	process_noise.slice<2, 1>(22, 0) = sq(wind_vel_sig);
	// terrain vertical position state
	process_noise.slice<1, 1>(24, 0) = sq(_imu_sample_delayed.delta_vel_dt * _params.terrain_p_noise) + sq(_imu_sample_delayed.delta_vel_dt * _params.terrain_gradient)
			* (sq(_state.vel(0)) + sq(_state.vel(1)));

	// assign IMU noise variances
	// inputs to the system are 3 delta angles and 3 delta velocities
	float gyro_noise = math::constrain(_params.gyro_noise, 0.0f, 1.0f);
	const float daxVar = sq(dt * gyro_noise);
	const float dayVar = daxVar;
	const float dazVar = daxVar;

	float accel_noise = math::constrain(_params.accel_noise, 0.0f, 1.0f);

	if (_fault_status.flags.bad_acc_vertical) {
		// Increase accelerometer process noise if bad accel data is detected. Measurement errors due to
		// vibration induced clipping commonly reach an equivalent 0.5g offset.
		accel_noise = BADACC_BIAS_PNOISE;
	}

	float dvxVar, dvyVar, dvzVar;
	dvxVar = dvyVar = dvzVar = sq(dt * accel_noise);

	// Accelerometer Clipping
	// delta velocity X: increase process noise if sample contained any X axis clipping
	if (_imu_sample_delayed.delta_vel_clipping[0]) {
		dvxVar = sq(dt * BADACC_BIAS_PNOISE);
	}

	// delta velocity Y: increase process noise if sample contained any Y axis clipping
	if (_imu_sample_delayed.delta_vel_clipping[1]) {
		dvyVar = sq(dt * BADACC_BIAS_PNOISE);
	}

	// delta velocity Z: increase process noise if sample contained any Z axis clipping
	if (_imu_sample_delayed.delta_vel_clipping[2]) {
		dvzVar = sq(dt * BADACC_BIAS_PNOISE);
	}

	// predict the covariance
	// equations generated using EKF/python/ekf_derivation/main.py

	// intermediate calculations
	const float PS0 = 0.5F*q1;
	const float PS1 = q0*rotErrX;
	const float PS2 = q2*rotErrZ;
	const float PS3 = 0.25F*rotErrY;
	const float PS4 = PS0 + 0.25F*PS1 + 0.25F*PS2 - PS3*q3;
	const float PS5 = 0.5F*q2;
	const float PS6 = q0*rotErrY;
	const float PS7 = 0.25F*rotErrZ;
	const float PS8 = q3*rotErrX;
	const float PS9 = PS5 + 0.25F*PS6 - PS7*q1 + 0.25F*PS8;
	const float PS10 = 0.5F*q3;
	const float PS11 = 0.25F*q2;
	const float PS12 = PS10 - PS11*rotErrX + PS3*q1 + PS7*q0;
	const float PS13 = 0.5F*q0;
	const float PS14 = q1*rotErrX;
	const float PS15 = q2*rotErrY;
	const float PS16 = q3*rotErrZ;
	const float PS17 = -PS13 + 0.25F*PS14 + 0.25F*PS15 + 0.25F*PS16;
	const float PS18 = PS12*q3 - PS17*q0 + PS4*q1 + PS9*q2;
	const float PS19 = powf(PS18, 2);
	const float PS20 = PS12*q0 + PS17*q3 + PS4*q2 - PS9*q1;
	const float PS21 = powf(PS20, 2);
	const float PS22 = PS12*q1 + PS17*q2 - PS4*q3 + PS9*q0;
	const float PS23 = powf(PS22, 2);
	const float PS24 = -P(9,10)*PS18;
	const float PS25 = -dax + daxVar + dax_b;
	const float PS26 = 0.25F*q1;
	const float PS27 = PS25*PS26;
	const float PS28 = -day + dayVar + day_b;
	const float PS29 = PS11*PS28;
	const float PS30 = -PS29;
	const float PS31 = -daz + dazVar + daz_b;
	const float PS32 = 0.25F*q3;
	const float PS33 = PS31*PS32;
	const float PS34 = PS13 - PS33;
	const float PS35 = PS27 + PS30 + PS34;
	const float PS36 = PS26*PS31;
	const float PS37 = 0.25F*q0;
	const float PS38 = PS28*PS37;
	const float PS39 = PS25*PS32;
	const float PS40 = PS38 - PS39;
	const float PS41 = -PS36 + PS40 + PS5;
	const float PS42 = PS26*PS28;
	const float PS43 = PS11*PS25;
	const float PS44 = PS31*PS37;
	const float PS45 = PS10 + PS44;
	const float PS46 = PS42 + PS43 + PS45;
	const float PS47 = PS11*PS31;
	const float PS48 = -PS47;
	const float PS49 = PS25*PS37;
	const float PS50 = PS28*PS32;
	const float PS51 = PS49 + PS50;
	const float PS52 = -PS0 + PS48 + PS51;
	const float PS53 = PS35*q0 + PS41*q2 + PS46*q3 - PS52*q1;
	const float PS54 = -PS27;
	const float PS55 = PS29 + PS34 + PS54;
	const float PS56 = -PS42;
	const float PS57 = -PS43 + PS45 + PS56;
	const float PS58 = PS36 + PS40 - PS5;
	const float PS59 = PS0 + PS47 + PS51;
	const float PS60 = PS55*q3 - PS57*q0 - PS58*q1 - PS59*q2;
	const float PS61 = PS36 + PS38 + PS39 + PS5;
	const float PS62 = -PS10 + PS43 + PS44 + PS56;
	const float PS63 = PS13 + PS30 + PS33 + PS54;
	const float PS64 = PS0 + PS48 + PS49 - PS50;
	const float PS65 = PS61*q0 - PS62*q1 - PS63*q2 - PS64*q3;
	const float PS66 = P(0,10)*PS53 + P(1,10)*PS60 + P(10,10)*PS20 - P(10,11)*PS22 + P(2,10)*PS65 + PS24;
	const float PS67 = P(9,10)*PS20;
	const float PS68 = P(9,11)*PS22;
	const float PS69 = P(0,9)*PS53 + P(1,9)*PS60 + P(2,9)*PS65 - P(9,9)*PS18 + PS67 - PS68;
	const float PS70 = -P(9,11)*PS18;
	const float PS71 = P(0,11)*PS53 + P(1,11)*PS60 + P(10,11)*PS20 - P(11,11)*PS22 + P(2,11)*PS65 + PS70;
	const float PS72 = P(0,0)*PS53 + P(0,1)*PS60 + P(0,10)*PS20 - P(0,11)*PS22 + P(0,2)*PS65 - P(0,9)*PS18;
	const float PS73 = P(0,2)*PS53 + P(1,2)*PS60 + P(2,10)*PS20 - P(2,11)*PS22 + P(2,2)*PS65 - P(2,9)*PS18;
	const float PS74 = P(0,1)*PS53 + P(1,1)*PS60 + P(1,10)*PS20 - P(1,11)*PS22 + P(1,2)*PS65 - P(1,9)*PS18;
	const float PS75 = PS18*PS20;
	const float PS76 = -PS12*q2 + PS17*q1 + PS4*q0 + PS9*q3;
	const float PS77 = PS22*dazVar;
	const float PS78 = PS55*q0 + PS57*q3 - PS58*q2 + PS59*q1;
	const float PS79 = -PS61*q3 - PS62*q2 + PS63*q1 - PS64*q0;
	const float PS80 = -PS35*q3 - PS41*q1 + PS46*q0 - PS52*q2;
	const float PS81 = PS22*daxVar;
	const float PS82 = PS76*dayVar;
	const float PS83 = PS61*q2 - PS62*q3 + PS63*q0 + PS64*q1;
	const float PS84 = -PS55*q1 - PS57*q2 - PS58*q3 + PS59*q0;
	const float PS85 = PS35*q2 - PS41*q0 - PS46*q1 - PS52*q3;
	const float PS86 = -PS0*rotErrZ + PS10*rotErrX + PS13*rotErrY + q2;
	const float PS87 = 2*powf(PS86, 2);
	const float PS88 = PS0*rotErrY + PS13*rotErrZ - PS5*rotErrX + q3;
	const float PS89 = 2*powf(PS88, 2) - 1;
	const float PS90 = PS87 + PS89;
	const float PS91 = P(0,12)*PS53 + P(1,12)*PS60 + P(10,12)*PS20 - P(11,12)*PS22 + P(2,12)*PS65 - P(9,12)*PS18;
	const float PS92 = PS88*q0;
	const float PS93 = PS86*q1;
	const float PS94 = -dvx + dvxVar + dvx_b;
	const float PS95 = 2.0F*PS94;
	const float PS96 = -dvy + dvyVar + dvy_b;
	const float PS97 = -2*q0;
	const float PS98 = PS14 + PS15 + PS16 + PS97;
	const float PS99 = PS13*PS98;
	const float PS100 = 1.0F*q3;
	const float PS101 = PS100*PS88;
	const float PS102 = PS101 + PS99;
	const float PS103 = 1.0F*PS86*q2;
	const float PS104 = PS1 - PS100*rotErrY + PS2 + 2*q1;
	const float PS105 = PS0*PS104;
	const float PS106 = PS103 - PS105;
	const float PS107 = -dvz + dvzVar + dvz_b;
	const float PS108 = 1.0F*PS14 + 1.0F*PS15 + 1.0F*PS16 + PS97;
	const float PS109 = PS0*PS108;
	const float PS110 = PS88*q2;
	const float PS111 = 1.0F*PS110;
	const float PS112 = PS104*PS13;
	const float PS113 = PS86*q3;
	const float PS114 = 1.0F*PS113;
	const float PS115 = PS112 - PS114;
	const float PS116 = PS111 + PS115;
	const float PS117 = PS107*(PS109 + PS116) - PS95*(PS92 - PS93) + PS96*(PS102 + PS106);
	const float PS118 = PS86*q0;
	const float PS119 = PS88*q1;
	const float PS120 = PS0*PS98;
	const float PS121 = -PS108*PS13;
	const float PS122 = -PS101;
	const float PS123 = PS121 + PS122;
	const float PS124 = PS107*(-PS103 + PS105 + PS123) - PS95*(PS118 + PS119) + PS96*(PS116 + PS120);
	const float PS125 = 1.0F*PS119;
	const float PS126 = PS5*PS98;
	const float PS127 = PS10*PS104 + 1.0F*PS118;
	const float PS128 = PS10*PS108;
	const float PS129 = 1.0F*PS93;
	const float PS130 = 1.0F*PS92;
	const float PS131 = -PS104*PS5 + PS130;
	const float PS132 = PS107*(-PS128 - PS129 + PS131) + PS95*(PS110 - PS113) + PS96*(PS125 - PS126 + PS127);
	const float PS133 = -PS10*rotErrY + PS13*rotErrX + PS5*rotErrZ + q1;
	const float PS134 = PS133*PS86;
	const float PS135 = PS0*rotErrX + PS10*rotErrZ + PS5*rotErrY - q0;
	const float PS136 = PS135*PS88;
	const float PS137 = PS134 + PS136;
	const float PS138 = P(0,13)*PS53 + P(1,13)*PS60 + P(10,13)*PS20 - P(11,13)*PS22 + P(2,13)*PS65 - P(9,13)*PS18;
	const float PS139 = 2*PS138;
	const float PS140 = PS133*PS88;
	const float PS141 = PS135*PS86;
	const float PS142 = PS140 - PS141;
	const float PS143 = P(0,14)*PS53 + P(1,14)*PS60 + P(10,14)*PS20 - P(11,14)*PS22 + P(2,14)*PS65 - P(9,14)*PS18;
	const float PS144 = 2*PS143;
	const float PS145 = P(0,3)*PS53 + P(1,3)*PS60 + P(2,3)*PS65 + P(3,10)*PS20 - P(3,11)*PS22 - P(3,9)*PS18;
	const float PS146 = 2*powf(PS133, 2);
	const float PS147 = PS146 + PS89;
	const float PS148 = PS133*q3;
	const float PS149 = -PS148;
	const float PS150 = 2.0F*PS96;
	const float PS151 = PS133*q2;
	const float PS152 = 1.0F*PS151;
	const float PS153 = PS10*PS98;
	const float PS154 = 1.0F*q1;
	const float PS155 = -PS154*rotErrZ + PS6 + PS8 + 2*q2;
	const float PS156 = PS0*PS155 + PS130;
	const float PS157 = PS107*(PS152 - PS153 + PS156) - PS150*(PS119 + PS149) + PS94*(-PS109 - PS111 + PS115);
	const float PS158 = PS133*q0;
	const float PS159 = PS133*PS154;
	const float PS160 = -PS155*PS5;
	const float PS161 = PS108*PS5;
	const float PS162 = -PS125;
	const float PS163 = PS107*(PS102 + PS159 + PS160) - PS150*(-PS110 + PS158) + PS94*(PS127 + PS161 + PS162);
	const float PS164 = 1.0F*PS148;
	const float PS165 = PS13*PS155 + PS162;
	const float PS166 = PS107*(PS126 + PS164 + PS165) - PS150*(PS151 + PS92) + PS94*(PS106 + PS123);
	const float PS167 = PS134 - PS136;
	const float PS168 = 2*PS91;
	const float PS169 = PS86*PS88;
	const float PS170 = PS133*PS135;
	const float PS171 = PS169 + PS170;
	const float PS172 = P(0,4)*PS53 + P(1,4)*PS60 + P(2,4)*PS65 + P(4,10)*PS20 - P(4,11)*PS22 - P(4,9)*PS18;
	const float PS173 = PS146 + PS87 - 1;
	const float PS174 = 2.0F*PS107;
	const float PS175 = -PS174*(PS118 + PS149) + PS94*(PS103 + PS105 + PS122 + PS99) + PS96*(PS128 - PS152 + PS156);
	const float PS176 = PS174*(-PS151 + PS93) + PS94*(PS111 + PS112 + PS114 - PS120) + PS96*(-PS161 - PS164 + PS165);
	const float PS177 = -PS174*(PS113 + PS158) + PS94*(PS129 + PS131 + PS153) + PS96*(PS101 + PS121 - PS159 + PS160);
	const float PS178 = PS140 + PS141;
	const float PS179 = PS169 - PS170;
	const float PS180 = P(0,5)*PS53 + P(1,5)*PS60 + P(2,5)*PS65 + P(5,10)*PS20 - P(5,11)*PS22 - P(5,9)*PS18;
	const float PS181 = powf(PS76, 2);
	const float PS182 = -P(10,11)*PS18;
	const float PS183 = P(0,11)*PS80 + P(1,11)*PS78 + P(11,11)*PS76 + P(2,11)*PS79 - P(9,11)*PS20 + PS182;
	const float PS184 = P(10,11)*PS76;
	const float PS185 = P(0,10)*PS80 + P(1,10)*PS78 - P(10,10)*PS18 + P(2,10)*PS79 + PS184 - PS67;
	const float PS186 = P(0,9)*PS80 + P(1,9)*PS78 + P(2,9)*PS79 + P(9,11)*PS76 - P(9,9)*PS20 + PS24;
	const float PS187 = P(0,1)*PS80 + P(1,1)*PS78 - P(1,10)*PS18 + P(1,11)*PS76 + P(1,2)*PS79 - P(1,9)*PS20;
	const float PS188 = P(0,2)*PS80 + P(1,2)*PS78 - P(2,10)*PS18 + P(2,11)*PS76 + P(2,2)*PS79 - P(2,9)*PS20;
	const float PS189 = P(0,0)*PS80 + P(0,1)*PS78 - P(0,10)*PS18 + P(0,11)*PS76 + P(0,2)*PS79 - P(0,9)*PS20;
	const float PS190 = P(0,12)*PS80 + P(1,12)*PS78 - P(10,12)*PS18 + P(11,12)*PS76 + P(2,12)*PS79 - P(9,12)*PS20;
	const float PS191 = P(0,13)*PS80 + P(1,13)*PS78 - P(10,13)*PS18 + P(11,13)*PS76 + P(2,13)*PS79 - P(9,13)*PS20;
	const float PS192 = 2*PS191;
	const float PS193 = P(0,14)*PS80 + P(1,14)*PS78 - P(10,14)*PS18 + P(11,14)*PS76 + P(2,14)*PS79 - P(9,14)*PS20;
	const float PS194 = 2*PS193;
	const float PS195 = P(0,3)*PS80 + P(1,3)*PS78 + P(2,3)*PS79 - P(3,10)*PS18 + P(3,11)*PS76 - P(3,9)*PS20;
	const float PS196 = 2*PS190;
	const float PS197 = P(0,4)*PS80 + P(1,4)*PS78 + P(2,4)*PS79 - P(4,10)*PS18 + P(4,11)*PS76 - P(4,9)*PS20;
	const float PS198 = P(0,5)*PS80 + P(1,5)*PS78 + P(2,5)*PS79 - P(5,10)*PS18 + P(5,11)*PS76 - P(5,9)*PS20;
	const float PS199 = P(0,9)*PS85 + P(1,9)*PS84 + P(2,9)*PS83 - P(9,10)*PS76 + P(9,9)*PS22 + PS70;
	const float PS200 = P(0,11)*PS85 + P(1,11)*PS84 - P(11,11)*PS18 + P(2,11)*PS83 - PS184 + PS68;
	const float PS201 = P(0,10)*PS85 + P(1,10)*PS84 - P(10,10)*PS76 + P(2,10)*PS83 + P(9,10)*PS22 + PS182;
	const float PS202 = P(0,2)*PS85 + P(1,2)*PS84 - P(2,10)*PS76 - P(2,11)*PS18 + P(2,2)*PS83 + P(2,9)*PS22;
	const float PS203 = P(0,1)*PS85 + P(1,1)*PS84 - P(1,10)*PS76 - P(1,11)*PS18 + P(1,2)*PS83 + P(1,9)*PS22;
	const float PS204 = P(0,0)*PS85 + P(0,1)*PS84 - P(0,10)*PS76 - P(0,11)*PS18 + P(0,2)*PS83 + P(0,9)*PS22;
	const float PS205 = P(0,12)*PS85 + P(1,12)*PS84 - P(10,12)*PS76 - P(11,12)*PS18 + P(2,12)*PS83 + P(9,12)*PS22;
	const float PS206 = P(0,13)*PS85 + P(1,13)*PS84 - P(10,13)*PS76 - P(11,13)*PS18 + P(2,13)*PS83 + P(9,13)*PS22;
	const float PS207 = 2*PS206;
	const float PS208 = P(0,14)*PS85 + P(1,14)*PS84 - P(10,14)*PS76 - P(11,14)*PS18 + P(2,14)*PS83 + P(9,14)*PS22;
	const float PS209 = 2*PS208;
	const float PS210 = P(0,3)*PS85 + P(1,3)*PS84 + P(2,3)*PS83 - P(3,10)*PS76 - P(3,11)*PS18 + P(3,9)*PS22;
	const float PS211 = 2*PS205;
	const float PS212 = P(0,4)*PS85 + P(1,4)*PS84 + P(2,4)*PS83 - P(4,10)*PS76 - P(4,11)*PS18 + P(4,9)*PS22;
	const float PS213 = P(0,5)*PS85 + P(1,5)*PS84 + P(2,5)*PS83 - P(5,10)*PS76 - P(5,11)*PS18 + P(5,9)*PS22;
	const float PS214 = 2*PS137;
	const float PS215 = 2*PS142;
	const float PS216 = -P(0,12)*PS132 - P(1,12)*PS124 + P(12,12)*PS90 - P(12,13)*PS214 - P(12,14)*PS215 - P(2,12)*PS117 + P(3,12);
	const float PS217 = -P(0,2)*PS132 - P(1,2)*PS124 + P(2,12)*PS90 - P(2,13)*PS214 - P(2,14)*PS215 - P(2,2)*PS117 + P(2,3);
	const float PS218 = -P(0,1)*PS132 - P(1,1)*PS124 + P(1,12)*PS90 - P(1,13)*PS214 - P(1,14)*PS215 - P(1,2)*PS117 + P(1,3);
	const float PS219 = -P(0,0)*PS132 - P(0,1)*PS124 + P(0,12)*PS90 - P(0,13)*PS214 - P(0,14)*PS215 - P(0,2)*PS117 + P(0,3);
	const float PS220 = -P(0,13)*PS132 - P(1,13)*PS124 + P(12,13)*PS90 - P(13,13)*PS214 - P(13,14)*PS215 - P(2,13)*PS117 + P(3,13);
	const float PS221 = -P(0,14)*PS132 - P(1,14)*PS124 + P(12,14)*PS90 - P(13,14)*PS214 - P(14,14)*PS215 - P(2,14)*PS117 + P(3,14);
	const float PS222 = 4*dvyVar;
	const float PS223 = 4*dvzVar;
	const float PS224 = -P(0,3)*PS132 - P(1,3)*PS124 - P(2,3)*PS117 + P(3,12)*PS90 - P(3,13)*PS214 - P(3,14)*PS215 + P(3,3);
	const float PS225 = 2*PS216;
	const float PS226 = 2*PS171;
	const float PS227 = 2*PS167;
	const float PS228 = PS90*dvxVar;
	const float PS229 = PS147*dvyVar;
	const float PS230 = -P(0,4)*PS132 - P(1,4)*PS124 - P(2,4)*PS117 + P(3,4) + P(4,12)*PS90 - P(4,13)*PS214 - P(4,14)*PS215;
	const float PS231 = 2*PS179;
	const float PS232 = 2*PS178;
	const float PS233 = PS173*dvzVar;
	const float PS234 = -P(0,5)*PS132 - P(1,5)*PS124 - P(2,5)*PS117 + P(3,5) + P(5,12)*PS90 - P(5,13)*PS214 - P(5,14)*PS215;
	const float PS235 = -P(0,13)*PS163 - P(1,13)*PS157 - P(12,13)*PS227 + P(13,13)*PS147 - P(13,14)*PS226 - P(2,13)*PS166 + P(4,13);
	const float PS236 = -P(0,1)*PS163 - P(1,1)*PS157 - P(1,12)*PS227 + P(1,13)*PS147 - P(1,14)*PS226 - P(1,2)*PS166 + P(1,4);
	const float PS237 = -P(0,0)*PS163 - P(0,1)*PS157 - P(0,12)*PS227 + P(0,13)*PS147 - P(0,14)*PS226 - P(0,2)*PS166 + P(0,4);
	const float PS238 = -P(0,2)*PS163 - P(1,2)*PS157 - P(2,12)*PS227 + P(2,13)*PS147 - P(2,14)*PS226 - P(2,2)*PS166 + P(2,4);
	const float PS239 = -P(0,12)*PS163 - P(1,12)*PS157 - P(12,12)*PS227 + P(12,13)*PS147 - P(12,14)*PS226 - P(2,12)*PS166 + P(4,12);
	const float PS240 = -P(0,14)*PS163 - P(1,14)*PS157 - P(12,14)*PS227 + P(13,14)*PS147 - P(14,14)*PS226 - P(2,14)*PS166 + P(4,14);
	const float PS241 = 4*dvxVar;
	const float PS242 = -P(0,4)*PS163 - P(1,4)*PS157 - P(2,4)*PS166 - P(4,12)*PS227 + P(4,13)*PS147 - P(4,14)*PS226 + P(4,4);
	const float PS243 = -P(0,5)*PS163 - P(1,5)*PS157 - P(2,5)*PS166 + P(4,5) - P(5,12)*PS227 + P(5,13)*PS147 - P(5,14)*PS226;
	const float PS244 = -P(0,14)*PS177 - P(1,14)*PS175 - P(12,14)*PS232 - P(13,14)*PS231 + P(14,14)*PS173 - P(2,14)*PS176 + P(5,14);
	const float PS245 = -P(0,12)*PS177 - P(1,12)*PS175 - P(12,12)*PS232 - P(12,13)*PS231 + P(12,14)*PS173 - P(2,12)*PS176 + P(5,12);
	const float PS246 = -P(0,13)*PS177 - P(1,13)*PS175 - P(12,13)*PS232 - P(13,13)*PS231 + P(13,14)*PS173 - P(2,13)*PS176 + P(5,13);
	const float PS247 = -P(0,5)*PS177 - P(1,5)*PS175 - P(2,5)*PS176 - P(5,12)*PS232 - P(5,13)*PS231 + P(5,14)*PS173 + P(5,5);

	// covariance update
	SquareMatrix25f nextP;

	// calculate variances and upper diagonal covariances for angle error, velocity, position and gyro bias states

	nextP(0,0) = -4*PS18*PS69 + 4*PS19*daxVar + 4*PS20*PS66 + 4*PS21*dayVar - 4*PS22*PS71 + 4*PS23*dazVar + 4*PS53*PS72 + 4*PS60*PS74 + 4*PS65*PS73;
	nextP(0,1) = -4*PS18*PS66 - 4*PS20*PS69 + 4*PS71*PS76 + 4*PS72*PS80 + 4*PS73*PS79 + 4*PS74*PS78 + 4*PS75*daxVar - 4*PS75*dayVar - 4*PS76*PS77;
	nextP(1,1) = -4*PS18*PS185 + 4*PS181*dazVar + 4*PS183*PS76 - 4*PS186*PS20 + 4*PS187*PS78 + 4*PS188*PS79 + 4*PS189*PS80 + 4*PS19*dayVar + 4*PS21*daxVar;
	nextP(0,2) = -4*PS18*PS71 + 4*PS18*PS77 - 4*PS18*PS81 - 4*PS20*PS82 + 4*PS22*PS69 - 4*PS66*PS76 + 4*PS72*PS85 + 4*PS73*PS83 + 4*PS74*PS84;
	nextP(1,2) = -4*PS18*PS183 - 4*PS18*PS76*dazVar + 4*PS18*PS82 - 4*PS185*PS76 + 4*PS186*PS22 + 4*PS187*PS84 + 4*PS188*PS83 + 4*PS189*PS85 - 4*PS20*PS81;
	nextP(2,2) = -4*PS18*PS200 + 4*PS181*dayVar + 4*PS19*dazVar + 4*PS199*PS22 - 4*PS201*PS76 + 4*PS202*PS83 + 4*PS203*PS84 + 4*PS204*PS85 + 4*PS23*daxVar;
	nextP(0,3) = -2*PS117*PS73 - 2*PS124*PS74 - 2*PS132*PS72 - 2*PS137*PS139 - 2*PS142*PS144 + 2*PS145 + 2*PS90*PS91;
	nextP(1,3) = -2*PS117*PS188 - 2*PS124*PS187 - 2*PS132*PS189 - 2*PS137*PS192 - 2*PS142*PS194 + 2*PS190*PS90 + 2*PS195;
	nextP(2,3) = -2*PS117*PS202 - 2*PS124*PS203 - 2*PS132*PS204 - 2*PS137*PS207 - 2*PS142*PS209 + 2*PS205*PS90 + 2*PS210;
	nextP(3,3) = -PS117*PS217 - PS124*PS218 - PS132*PS219 + powf(PS137, 2)*PS222 + powf(PS142, 2)*PS223 - PS214*PS220 - PS215*PS221 + PS216*PS90 + PS224 + powf(PS90, 2)*dvxVar;
	nextP(0,4) = 2*PS138*PS147 - 2*PS144*PS171 - 2*PS157*PS74 - 2*PS163*PS72 - 2*PS166*PS73 - 2*PS167*PS168 + 2*PS172;
	nextP(1,4) = 2*PS147*PS191 - 2*PS157*PS187 - 2*PS163*PS189 - 2*PS166*PS188 - 2*PS167*PS196 - 2*PS171*PS194 + 2*PS197;
	nextP(2,4) = 2*PS147*PS206 - 2*PS157*PS203 - 2*PS163*PS204 - 2*PS166*PS202 - 2*PS167*PS211 - 2*PS171*PS209 + 2*PS212;
	nextP(3,4) = PS142*PS171*PS223 + PS147*PS220 - PS157*PS218 - PS163*PS219 - PS166*PS217 - PS167*PS225 - PS214*PS229 - PS221*PS226 - PS227*PS228 + PS230;
	nextP(4,4) = powf(PS147, 2)*dvyVar + PS147*PS235 - PS157*PS236 - PS163*PS237 - PS166*PS238 + powf(PS167, 2)*PS241 + powf(PS171, 2)*PS223 - PS226*PS240 - PS227*PS239 + PS242;
	nextP(0,5) = -2*PS139*PS179 + 2*PS143*PS173 - 2*PS168*PS178 - 2*PS175*PS74 - 2*PS176*PS73 - 2*PS177*PS72 + 2*PS180;
	nextP(1,5) = 2*PS173*PS193 - 2*PS175*PS187 - 2*PS176*PS188 - 2*PS177*PS189 - 2*PS178*PS196 - 2*PS179*PS192 + 2*PS198;
	nextP(2,5) = 2*PS173*PS208 - 2*PS175*PS203 - 2*PS176*PS202 - 2*PS177*PS204 - 2*PS178*PS211 - 2*PS179*PS207 + 2*PS213;
	nextP(3,5) = PS137*PS179*PS222 + PS173*PS221 - PS175*PS218 - PS176*PS217 - PS177*PS219 - PS178*PS225 - PS215*PS233 - PS220*PS231 - PS228*PS232 + PS234;
	nextP(4,5) = PS167*PS178*PS241 + PS173*PS240 - PS175*PS236 - PS176*PS238 - PS177*PS237 - PS226*PS233 - PS229*PS231 - PS231*PS235 - PS232*PS239 + PS243;
	nextP(5,5) = powf(PS173, 2)*dvzVar + PS173*PS244 - PS175*(-P(0,1)*PS177 - P(1,1)*PS175 - P(1,12)*PS232 - P(1,13)*PS231 + P(1,14)*PS173 - P(1,2)*PS176 + P(1,5)) - PS176*(-P(0,2)*PS177 - P(1,2)*PS175 - P(2,12)*PS232 - P(2,13)*PS231 + P(2,14)*PS173 - P(2,2)*PS176 + P(2,5)) - PS177*(-P(0,0)*PS177 - P(0,1)*PS175 - P(0,12)*PS232 - P(0,13)*PS231 + P(0,14)*PS173 - P(0,2)*PS176 + P(0,5)) + powf(PS178, 2)*PS241 + powf(PS179, 2)*PS222 - PS231*PS246 - PS232*PS245 + PS247;
	nextP(0,6) = 2*P(0,6)*PS53 + 2*P(1,6)*PS60 + 2*P(2,6)*PS65 + 2*P(6,10)*PS20 - 2*P(6,11)*PS22 - 2*P(6,9)*PS18 + 2*PS145*dt;
	nextP(1,6) = 2*P(0,6)*PS80 + 2*P(1,6)*PS78 + 2*P(2,6)*PS79 - 2*P(6,10)*PS18 + 2*P(6,11)*PS76 - 2*P(6,9)*PS20 + 2*PS195*dt;
	nextP(2,6) = 2*P(0,6)*PS85 + 2*P(1,6)*PS84 + 2*P(2,6)*PS83 - 2*P(6,10)*PS76 - 2*P(6,11)*PS18 + 2*P(6,9)*PS22 + 2*PS210*dt;
	nextP(3,6) = -P(0,6)*PS132 - P(1,6)*PS124 - P(2,6)*PS117 + P(3,6) + P(6,12)*PS90 - P(6,13)*PS214 - P(6,14)*PS215 + PS224*dt;
	nextP(4,6) = -P(0,6)*PS163 - P(1,6)*PS157 - P(2,6)*PS166 + P(4,6) - P(6,12)*PS227 + P(6,13)*PS147 - P(6,14)*PS226 + dt*(-P(0,3)*PS163 - P(1,3)*PS157 - P(2,3)*PS166 - P(3,12)*PS227 + P(3,13)*PS147 - P(3,14)*PS226 + P(3,4));
	nextP(5,6) = -P(0,6)*PS177 - P(1,6)*PS175 - P(2,6)*PS176 + P(5,6) - P(6,12)*PS232 - P(6,13)*PS231 + P(6,14)*PS173 + dt*(-P(0,3)*PS177 - P(1,3)*PS175 - P(2,3)*PS176 - P(3,12)*PS232 - P(3,13)*PS231 + P(3,14)*PS173 + P(3,5));
	nextP(6,6) = P(3,6)*dt + P(6,6) + dt*(P(3,3)*dt + P(3,6));
	nextP(0,7) = 2*P(0,7)*PS53 + 2*P(1,7)*PS60 + 2*P(2,7)*PS65 + 2*P(7,10)*PS20 - 2*P(7,11)*PS22 - 2*P(7,9)*PS18 + 2*PS172*dt;
	nextP(1,7) = 2*P(0,7)*PS80 + 2*P(1,7)*PS78 + 2*P(2,7)*PS79 - 2*P(7,10)*PS18 + 2*P(7,11)*PS76 - 2*P(7,9)*PS20 + 2*PS197*dt;
	nextP(2,7) = 2*P(0,7)*PS85 + 2*P(1,7)*PS84 + 2*P(2,7)*PS83 - 2*P(7,10)*PS76 - 2*P(7,11)*PS18 + 2*P(7,9)*PS22 + 2*PS212*dt;
	nextP(3,7) = -P(0,7)*PS132 - P(1,7)*PS124 - P(2,7)*PS117 + P(3,7) + P(7,12)*PS90 - P(7,13)*PS214 - P(7,14)*PS215 + PS230*dt;
	nextP(4,7) = -P(0,7)*PS163 - P(1,7)*PS157 - P(2,7)*PS166 + P(4,7) - P(7,12)*PS227 + P(7,13)*PS147 - P(7,14)*PS226 + PS242*dt;
	nextP(5,7) = -P(0,7)*PS177 - P(1,7)*PS175 - P(2,7)*PS176 + P(5,7) - P(7,12)*PS232 - P(7,13)*PS231 + P(7,14)*PS173 + dt*(-P(0,4)*PS177 - P(1,4)*PS175 - P(2,4)*PS176 - P(4,12)*PS232 - P(4,13)*PS231 + P(4,14)*PS173 + P(4,5));
	nextP(6,7) = P(3,7)*dt + P(6,7) + dt*(P(3,4)*dt + P(4,6));
	nextP(7,7) = P(4,7)*dt + P(7,7) + dt*(P(4,4)*dt + P(4,7));
	nextP(0,8) = 2*P(0,8)*PS53 + 2*P(1,8)*PS60 + 2*P(2,8)*PS65 + 2*P(8,10)*PS20 - 2*P(8,11)*PS22 - 2*P(8,9)*PS18 + 2*PS180*dt;
	nextP(1,8) = 2*P(0,8)*PS80 + 2*P(1,8)*PS78 + 2*P(2,8)*PS79 - 2*P(8,10)*PS18 + 2*P(8,11)*PS76 - 2*P(8,9)*PS20 + 2*PS198*dt;
	nextP(2,8) = 2*P(0,8)*PS85 + 2*P(1,8)*PS84 + 2*P(2,8)*PS83 - 2*P(8,10)*PS76 - 2*P(8,11)*PS18 + 2*P(8,9)*PS22 + 2*PS213*dt;
	nextP(3,8) = -P(0,8)*PS132 - P(1,8)*PS124 - P(2,8)*PS117 + P(3,8) + P(8,12)*PS90 - P(8,13)*PS214 - P(8,14)*PS215 + PS234*dt;
	nextP(4,8) = -P(0,8)*PS163 - P(1,8)*PS157 - P(2,8)*PS166 + P(4,8) - P(8,12)*PS227 + P(8,13)*PS147 - P(8,14)*PS226 + PS243*dt;
	nextP(5,8) = -P(0,8)*PS177 - P(1,8)*PS175 - P(2,8)*PS176 + P(5,8) - P(8,12)*PS232 - P(8,13)*PS231 + P(8,14)*PS173 + PS247*dt;
	nextP(6,8) = P(3,8)*dt + P(6,8) + dt*(P(3,5)*dt + P(5,6));
	nextP(7,8) = P(4,8)*dt + P(7,8) + dt*(P(4,5)*dt + P(5,7));
	nextP(8,8) = P(5,8)*dt + P(8,8) + dt*(P(5,5)*dt + P(5,8));
	nextP(0,9) = 2*PS69;
	nextP(1,9) = 2*PS186;
	nextP(2,9) = 2*PS199;
	nextP(3,9) = -P(0,9)*PS132 - P(1,9)*PS124 - P(2,9)*PS117 + P(3,9) + P(9,12)*PS90 - P(9,13)*PS214 - P(9,14)*PS215;
	nextP(4,9) = -P(0,9)*PS163 - P(1,9)*PS157 - P(2,9)*PS166 + P(4,9) - P(9,12)*PS227 + P(9,13)*PS147 - P(9,14)*PS226;
	nextP(5,9) = -P(0,9)*PS177 - P(1,9)*PS175 - P(2,9)*PS176 + P(5,9) - P(9,12)*PS232 - P(9,13)*PS231 + P(9,14)*PS173;
	nextP(6,9) = P(3,9)*dt + P(6,9);
	nextP(7,9) = P(4,9)*dt + P(7,9);
	nextP(8,9) = P(5,9)*dt + P(8,9);
	nextP(9,9) = P(9,9);
	nextP(0,10) = 2*PS66;
	nextP(1,10) = 2*PS185;
	nextP(2,10) = 2*PS201;
	nextP(3,10) = -P(0,10)*PS132 - P(1,10)*PS124 + P(10,12)*PS90 - P(10,13)*PS214 - P(10,14)*PS215 - P(2,10)*PS117 + P(3,10);
	nextP(4,10) = -P(0,10)*PS163 - P(1,10)*PS157 - P(10,12)*PS227 + P(10,13)*PS147 - P(10,14)*PS226 - P(2,10)*PS166 + P(4,10);
	nextP(5,10) = -P(0,10)*PS177 - P(1,10)*PS175 - P(10,12)*PS232 - P(10,13)*PS231 + P(10,14)*PS173 - P(2,10)*PS176 + P(5,10);
	nextP(6,10) = P(3,10)*dt + P(6,10);
	nextP(7,10) = P(4,10)*dt + P(7,10);
	nextP(8,10) = P(5,10)*dt + P(8,10);
	nextP(9,10) = P(9,10);
	nextP(10,10) = P(10,10);
	nextP(0,11) = 2*PS71;
	nextP(1,11) = 2*PS183;
	nextP(2,11) = 2*PS200;
	nextP(3,11) = -P(0,11)*PS132 - P(1,11)*PS124 + P(11,12)*PS90 - P(11,13)*PS214 - P(11,14)*PS215 - P(2,11)*PS117 + P(3,11);
	nextP(4,11) = -P(0,11)*PS163 - P(1,11)*PS157 - P(11,12)*PS227 + P(11,13)*PS147 - P(11,14)*PS226 - P(2,11)*PS166 + P(4,11);
	nextP(5,11) = -P(0,11)*PS177 - P(1,11)*PS175 - P(11,12)*PS232 - P(11,13)*PS231 + P(11,14)*PS173 - P(2,11)*PS176 + P(5,11);
	nextP(6,11) = P(3,11)*dt + P(6,11);
	nextP(7,11) = P(4,11)*dt + P(7,11);
	nextP(8,11) = P(5,11)*dt + P(8,11);
	nextP(9,11) = P(9,11);
	nextP(10,11) = P(10,11);
	nextP(11,11) = P(11,11);

	// process noise contribution for delta angle states can be very small compared to
	// the variances, therefore use algorithm to minimise numerical error
	for (unsigned i = 9; i <= 11; i++) {
		const int index = i - 9;
		nextP(i, i) = kahanSummation(nextP(i, i), process_noise(i), _delta_angle_bias_var_accum(index));
	}

	if (!_accel_bias_inhibit[0]) {
		// calculate variances and upper diagonal covariances for IMU X axis delta velocity bias state
		nextP(0,12) = 2*PS91;
		nextP(1,12) = 2*PS190;
		nextP(2,12) = 2*PS205;
		nextP(3,12) = PS216;
		nextP(4,12) = PS239;
		nextP(5,12) = PS245;
		nextP(6,12) = P(3,12)*dt + P(6,12);
		nextP(7,12) = P(4,12)*dt + P(7,12);
		nextP(8,12) = P(5,12)*dt + P(8,12);
		nextP(9,12) = P(9,12);
		nextP(10,12) = P(10,12);
		nextP(11,12) = P(11,12);
		nextP(12,12) = P(12,12);

		// add process noise that is not from the IMU
		// process noise contribution for delta velocity states can be very small compared to
		// the variances, therefore use algorithm to minimise numerical error
		nextP(12, 12) = kahanSummation(nextP(12, 12), process_noise(12), _delta_vel_bias_var_accum(0));

	} else {
		nextP.uncorrelateCovarianceSetVariance<1>(12, _prev_dvel_bias_var(0));
		_delta_vel_bias_var_accum(0) = 0.f;

	}

	if (!_accel_bias_inhibit[1]) {
		// calculate variances and upper diagonal covariances for IMU Y axis delta velocity bias state

		nextP(0,13) = 2*PS138;
		nextP(1,13) = 2*PS191;
		nextP(2,13) = 2*PS206;
		nextP(3,13) = PS220;
		nextP(4,13) = PS235;
		nextP(5,13) = PS246;
		nextP(6,13) = P(3,13)*dt + P(6,13);
		nextP(7,13) = P(4,13)*dt + P(7,13);
		nextP(8,13) = P(5,13)*dt + P(8,13);
		nextP(9,13) = P(9,13);
		nextP(10,13) = P(10,13);
		nextP(11,13) = P(11,13);
		nextP(12,13) = P(12,13);
		nextP(13,13) = P(13,13);

		// add process noise that is not from the IMU
		// process noise contribution for delta velocity states can be very small compared to
		// the variances, therefore use algorithm to minimise numerical error
		nextP(13, 13) = kahanSummation(nextP(13, 13), process_noise(13), _delta_vel_bias_var_accum(1));

	} else {
		nextP.uncorrelateCovarianceSetVariance<1>(13, _prev_dvel_bias_var(1));
		_delta_vel_bias_var_accum(1) = 0.f;

	}

	if (!_accel_bias_inhibit[2]) {
		// calculate variances and upper diagonal covariances for IMU Z axis delta velocity bias state
		nextP(0,14) = 2*PS143;
		nextP(1,14) = 2*PS193;
		nextP(2,14) = 2*PS208;
		nextP(3,14) = PS221;
		nextP(4,14) = PS240;
		nextP(5,14) = PS244;
		nextP(6,14) = P(3,14)*dt + P(6,14);
		nextP(7,14) = P(4,14)*dt + P(7,14);
		nextP(8,14) = P(5,14)*dt + P(8,14);
		nextP(9,14) = P(9,14);
		nextP(10,14) = P(10,14);
		nextP(11,14) = P(11,14);
		nextP(12,14) = P(12,14);
		nextP(13,14) = P(13,14);
		nextP(14,14) = P(14,14);

		// add process noise that is not from the IMU
		// process noise contribution for delta velocity states can be very small compared to
		// the variances, therefore use algorithm to minimise numerical error
		nextP(14, 14) = kahanSummation(nextP(14, 14), process_noise(14), _delta_vel_bias_var_accum(2));

	} else {
		nextP.uncorrelateCovarianceSetVariance<1>(14, _prev_dvel_bias_var(2));
		_delta_vel_bias_var_accum(2) = 0.f;
	}

	// Don't do covariance prediction on magnetic field states unless we are using 3-axis fusion
	if (_control_status.flags.mag_3D) {
		// calculate variances and upper diagonal covariances for earth and body magnetic field states

		nextP(0,15) = 2*P(0,15)*PS53 + 2*P(1,15)*PS60 + 2*P(10,15)*PS20 - 2*P(11,15)*PS22 + 2*P(2,15)*PS65 - 2*P(9,15)*PS18;
		nextP(1,15) = 2*P(0,15)*PS80 + 2*P(1,15)*PS78 - 2*P(10,15)*PS18 + 2*P(11,15)*PS76 + 2*P(2,15)*PS79 - 2*P(9,15)*PS20;
		nextP(2,15) = 2*P(0,15)*PS85 + 2*P(1,15)*PS84 - 2*P(10,15)*PS76 - 2*P(11,15)*PS18 + 2*P(2,15)*PS83 + 2*P(9,15)*PS22;
		nextP(3,15) = -P(0,15)*PS132 - P(1,15)*PS124 + P(12,15)*PS90 - P(13,15)*PS214 - P(14,15)*PS215 - P(2,15)*PS117 + P(3,15);
		nextP(4,15) = -P(0,15)*PS163 - P(1,15)*PS157 - P(12,15)*PS227 + P(13,15)*PS147 - P(14,15)*PS226 - P(2,15)*PS166 + P(4,15);
		nextP(5,15) = -P(0,15)*PS177 - P(1,15)*PS175 - P(12,15)*PS232 - P(13,15)*PS231 + P(14,15)*PS173 - P(2,15)*PS176 + P(5,15);
		nextP(6,15) = P(3,15)*dt + P(6,15);
		nextP(7,15) = P(4,15)*dt + P(7,15);
		nextP(8,15) = P(5,15)*dt + P(8,15);
		nextP(9,15) = P(9,15);
		nextP(10,15) = P(10,15);
		nextP(11,15) = P(11,15);
		nextP(12,15) = P(12,15);
		nextP(13,15) = P(13,15);
		nextP(14,15) = P(14,15);
		nextP(15,15) = P(15,15);
		nextP(0,16) = 2*P(0,16)*PS53 + 2*P(1,16)*PS60 + 2*P(10,16)*PS20 - 2*P(11,16)*PS22 + 2*P(2,16)*PS65 - 2*P(9,16)*PS18;
		nextP(1,16) = 2*P(0,16)*PS80 + 2*P(1,16)*PS78 - 2*P(10,16)*PS18 + 2*P(11,16)*PS76 + 2*P(2,16)*PS79 - 2*P(9,16)*PS20;
		nextP(2,16) = 2*P(0,16)*PS85 + 2*P(1,16)*PS84 - 2*P(10,16)*PS76 - 2*P(11,16)*PS18 + 2*P(2,16)*PS83 + 2*P(9,16)*PS22;
		nextP(3,16) = -P(0,16)*PS132 - P(1,16)*PS124 + P(12,16)*PS90 - P(13,16)*PS214 - P(14,16)*PS215 - P(2,16)*PS117 + P(3,16);
		nextP(4,16) = -P(0,16)*PS163 - P(1,16)*PS157 - P(12,16)*PS227 + P(13,16)*PS147 - P(14,16)*PS226 - P(2,16)*PS166 + P(4,16);
		nextP(5,16) = -P(0,16)*PS177 - P(1,16)*PS175 - P(12,16)*PS232 - P(13,16)*PS231 + P(14,16)*PS173 - P(2,16)*PS176 + P(5,16);
		nextP(6,16) = P(3,16)*dt + P(6,16);
		nextP(7,16) = P(4,16)*dt + P(7,16);
		nextP(8,16) = P(5,16)*dt + P(8,16);
		nextP(9,16) = P(9,16);
		nextP(10,16) = P(10,16);
		nextP(11,16) = P(11,16);
		nextP(12,16) = P(12,16);
		nextP(13,16) = P(13,16);
		nextP(14,16) = P(14,16);
		nextP(15,16) = P(15,16);
		nextP(16,16) = P(16,16);
		nextP(0,17) = 2*P(0,17)*PS53 + 2*P(1,17)*PS60 + 2*P(10,17)*PS20 - 2*P(11,17)*PS22 + 2*P(2,17)*PS65 - 2*P(9,17)*PS18;
		nextP(1,17) = 2*P(0,17)*PS80 + 2*P(1,17)*PS78 - 2*P(10,17)*PS18 + 2*P(11,17)*PS76 + 2*P(2,17)*PS79 - 2*P(9,17)*PS20;
		nextP(2,17) = 2*P(0,17)*PS85 + 2*P(1,17)*PS84 - 2*P(10,17)*PS76 - 2*P(11,17)*PS18 + 2*P(2,17)*PS83 + 2*P(9,17)*PS22;
		nextP(3,17) = -P(0,17)*PS132 - P(1,17)*PS124 + P(12,17)*PS90 - P(13,17)*PS214 - P(14,17)*PS215 - P(2,17)*PS117 + P(3,17);
		nextP(4,17) = -P(0,17)*PS163 - P(1,17)*PS157 - P(12,17)*PS227 + P(13,17)*PS147 - P(14,17)*PS226 - P(2,17)*PS166 + P(4,17);
		nextP(5,17) = -P(0,17)*PS177 - P(1,17)*PS175 - P(12,17)*PS232 - P(13,17)*PS231 + P(14,17)*PS173 - P(2,17)*PS176 + P(5,17);
		nextP(6,17) = P(3,17)*dt + P(6,17);
		nextP(7,17) = P(4,17)*dt + P(7,17);
		nextP(8,17) = P(5,17)*dt + P(8,17);
		nextP(9,17) = P(9,17);
		nextP(10,17) = P(10,17);
		nextP(11,17) = P(11,17);
		nextP(12,17) = P(12,17);
		nextP(13,17) = P(13,17);
		nextP(14,17) = P(14,17);
		nextP(15,17) = P(15,17);
		nextP(16,17) = P(16,17);
		nextP(17,17) = P(17,17);
		nextP(0,18) = 2*P(0,18)*PS53 + 2*P(1,18)*PS60 + 2*P(10,18)*PS20 - 2*P(11,18)*PS22 + 2*P(2,18)*PS65 - 2*P(9,18)*PS18;
		nextP(1,18) = 2*P(0,18)*PS80 + 2*P(1,18)*PS78 - 2*P(10,18)*PS18 + 2*P(11,18)*PS76 + 2*P(2,18)*PS79 - 2*P(9,18)*PS20;
		nextP(2,18) = 2*P(0,18)*PS85 + 2*P(1,18)*PS84 - 2*P(10,18)*PS76 - 2*P(11,18)*PS18 + 2*P(2,18)*PS83 + 2*P(9,18)*PS22;
		nextP(3,18) = -P(0,18)*PS132 - P(1,18)*PS124 + P(12,18)*PS90 - P(13,18)*PS214 - P(14,18)*PS215 - P(2,18)*PS117 + P(3,18);
		nextP(4,18) = -P(0,18)*PS163 - P(1,18)*PS157 - P(12,18)*PS227 + P(13,18)*PS147 - P(14,18)*PS226 - P(2,18)*PS166 + P(4,18);
		nextP(5,18) = -P(0,18)*PS177 - P(1,18)*PS175 - P(12,18)*PS232 - P(13,18)*PS231 + P(14,18)*PS173 - P(2,18)*PS176 + P(5,18);
		nextP(6,18) = P(3,18)*dt + P(6,18);
		nextP(7,18) = P(4,18)*dt + P(7,18);
		nextP(8,18) = P(5,18)*dt + P(8,18);
		nextP(9,18) = P(9,18);
		nextP(10,18) = P(10,18);
		nextP(11,18) = P(11,18);
		nextP(12,18) = P(12,18);
		nextP(13,18) = P(13,18);
		nextP(14,18) = P(14,18);
		nextP(15,18) = P(15,18);
		nextP(16,18) = P(16,18);
		nextP(17,18) = P(17,18);
		nextP(18,18) = P(18,18);
		nextP(0,19) = 2*P(0,19)*PS53 + 2*P(1,19)*PS60 + 2*P(10,19)*PS20 - 2*P(11,19)*PS22 + 2*P(2,19)*PS65 - 2*P(9,19)*PS18;
		nextP(1,19) = 2*P(0,19)*PS80 + 2*P(1,19)*PS78 - 2*P(10,19)*PS18 + 2*P(11,19)*PS76 + 2*P(2,19)*PS79 - 2*P(9,19)*PS20;
		nextP(2,19) = 2*P(0,19)*PS85 + 2*P(1,19)*PS84 - 2*P(10,19)*PS76 - 2*P(11,19)*PS18 + 2*P(2,19)*PS83 + 2*P(9,19)*PS22;
		nextP(3,19) = -P(0,19)*PS132 - P(1,19)*PS124 + P(12,19)*PS90 - P(13,19)*PS214 - P(14,19)*PS215 - P(2,19)*PS117 + P(3,19);
		nextP(4,19) = -P(0,19)*PS163 - P(1,19)*PS157 - P(12,19)*PS227 + P(13,19)*PS147 - P(14,19)*PS226 - P(2,19)*PS166 + P(4,19);
		nextP(5,19) = -P(0,19)*PS177 - P(1,19)*PS175 - P(12,19)*PS232 - P(13,19)*PS231 + P(14,19)*PS173 - P(2,19)*PS176 + P(5,19);
		nextP(6,19) = P(3,19)*dt + P(6,19);
		nextP(7,19) = P(4,19)*dt + P(7,19);
		nextP(8,19) = P(5,19)*dt + P(8,19);
		nextP(9,19) = P(9,19);
		nextP(10,19) = P(10,19);
		nextP(11,19) = P(11,19);
		nextP(12,19) = P(12,19);
		nextP(13,19) = P(13,19);
		nextP(14,19) = P(14,19);
		nextP(15,19) = P(15,19);
		nextP(16,19) = P(16,19);
		nextP(17,19) = P(17,19);
		nextP(18,19) = P(18,19);
		nextP(19,19) = P(19,19);
		nextP(0,20) = 2*P(0,20)*PS53 + 2*P(1,20)*PS60 + 2*P(10,20)*PS20 - 2*P(11,20)*PS22 + 2*P(2,20)*PS65 - 2*P(9,20)*PS18;
		nextP(1,20) = 2*P(0,20)*PS80 + 2*P(1,20)*PS78 - 2*P(10,20)*PS18 + 2*P(11,20)*PS76 + 2*P(2,20)*PS79 - 2*P(9,20)*PS20;
		nextP(2,20) = 2*P(0,20)*PS85 + 2*P(1,20)*PS84 - 2*P(10,20)*PS76 - 2*P(11,20)*PS18 + 2*P(2,20)*PS83 + 2*P(9,20)*PS22;
		nextP(3,20) = -P(0,20)*PS132 - P(1,20)*PS124 + P(12,20)*PS90 - P(13,20)*PS214 - P(14,20)*PS215 - P(2,20)*PS117 + P(3,20);
		nextP(4,20) = -P(0,20)*PS163 - P(1,20)*PS157 - P(12,20)*PS227 + P(13,20)*PS147 - P(14,20)*PS226 - P(2,20)*PS166 + P(4,20);
		nextP(5,20) = -P(0,20)*PS177 - P(1,20)*PS175 - P(12,20)*PS232 - P(13,20)*PS231 + P(14,20)*PS173 - P(2,20)*PS176 + P(5,20);
		nextP(6,20) = P(3,20)*dt + P(6,20);
		nextP(7,20) = P(4,20)*dt + P(7,20);
		nextP(8,20) = P(5,20)*dt + P(8,20);
		nextP(9,20) = P(9,20);
		nextP(10,20) = P(10,20);
		nextP(11,20) = P(11,20);
		nextP(12,20) = P(12,20);
		nextP(13,20) = P(13,20);
		nextP(14,20) = P(14,20);
		nextP(15,20) = P(15,20);
		nextP(16,20) = P(16,20);
		nextP(17,20) = P(17,20);
		nextP(18,20) = P(18,20);
		nextP(19,20) = P(19,20);
		nextP(20,20) = P(20,20);

		// add process noise that is not from the IMU
		for (unsigned i = 15; i <= 20; i++) {
			nextP(i, i) += process_noise(i);
		}

	}

	// Don't do covariance prediction on wind states unless we are using them
	if (_control_status.flags.wind) {

		// calculate variances and upper diagonal covariances for wind states

		nextP(0,21) = 2*P(0,21)*PS53 + 2*P(1,21)*PS60 + 2*P(10,21)*PS20 - 2*P(11,21)*PS22 + 2*P(2,21)*PS65 - 2*P(9,21)*PS18;
		nextP(1,21) = 2*P(0,21)*PS80 + 2*P(1,21)*PS78 - 2*P(10,21)*PS18 + 2*P(11,21)*PS76 + 2*P(2,21)*PS79 - 2*P(9,21)*PS20;
		nextP(2,21) = 2*P(0,21)*PS85 + 2*P(1,21)*PS84 - 2*P(10,21)*PS76 - 2*P(11,21)*PS18 + 2*P(2,21)*PS83 + 2*P(9,21)*PS22;
		nextP(3,21) = -P(0,21)*PS132 - P(1,21)*PS124 + P(12,21)*PS90 - P(13,21)*PS214 - P(14,21)*PS215 - P(2,21)*PS117 + P(3,21);
		nextP(4,21) = -P(0,21)*PS163 - P(1,21)*PS157 - P(12,21)*PS227 + P(13,21)*PS147 - P(14,21)*PS226 - P(2,21)*PS166 + P(4,21);
		nextP(5,21) = -P(0,21)*PS177 - P(1,21)*PS175 - P(12,21)*PS232 - P(13,21)*PS231 + P(14,21)*PS173 - P(2,21)*PS176 + P(5,21);
		nextP(6,21) = P(3,21)*dt + P(6,21);
		nextP(7,21) = P(4,21)*dt + P(7,21);
		nextP(8,21) = P(5,21)*dt + P(8,21);
		nextP(9,21) = P(9,21);
		nextP(10,21) = P(10,21);
		nextP(11,21) = P(11,21);
		nextP(12,21) = P(12,21);
		nextP(13,21) = P(13,21);
		nextP(14,21) = P(14,21);
		nextP(15,21) = P(15,21);
		nextP(16,21) = P(16,21);
		nextP(17,21) = P(17,21);
		nextP(18,21) = P(18,21);
		nextP(19,21) = P(19,21);
		nextP(20,21) = P(20,21);
		nextP(21,21) = P(21,21);
		nextP(0,22) = 2*P(0,22)*PS53 + 2*P(1,22)*PS60 + 2*P(10,22)*PS20 - 2*P(11,22)*PS22 + 2*P(2,22)*PS65 - 2*P(9,22)*PS18;
		nextP(1,22) = 2*P(0,22)*PS80 + 2*P(1,22)*PS78 - 2*P(10,22)*PS18 + 2*P(11,22)*PS76 + 2*P(2,22)*PS79 - 2*P(9,22)*PS20;
		nextP(2,22) = 2*P(0,22)*PS85 + 2*P(1,22)*PS84 - 2*P(10,22)*PS76 - 2*P(11,22)*PS18 + 2*P(2,22)*PS83 + 2*P(9,22)*PS22;
		nextP(3,22) = -P(0,22)*PS132 - P(1,22)*PS124 + P(12,22)*PS90 - P(13,22)*PS214 - P(14,22)*PS215 - P(2,22)*PS117 + P(3,22);
		nextP(4,22) = -P(0,22)*PS163 - P(1,22)*PS157 - P(12,22)*PS227 + P(13,22)*PS147 - P(14,22)*PS226 - P(2,22)*PS166 + P(4,22);
		nextP(5,22) = -P(0,22)*PS177 - P(1,22)*PS175 - P(12,22)*PS232 - P(13,22)*PS231 + P(14,22)*PS173 - P(2,22)*PS176 + P(5,22);
		nextP(6,22) = P(3,22)*dt + P(6,22);
		nextP(7,22) = P(4,22)*dt + P(7,22);
		nextP(8,22) = P(5,22)*dt + P(8,22);
		nextP(9,22) = P(9,22);
		nextP(10,22) = P(10,22);
		nextP(11,22) = P(11,22);
		nextP(12,22) = P(12,22);
		nextP(13,22) = P(13,22);
		nextP(14,22) = P(14,22);
		nextP(15,22) = P(15,22);
		nextP(16,22) = P(16,22);
		nextP(17,22) = P(17,22);
		nextP(18,22) = P(18,22);
		nextP(19,22) = P(19,22);
		nextP(20,22) = P(20,22);
		nextP(21,22) = P(21,22);
		nextP(22,22) = P(22,22);

		// add process noise that is not from the IMU
		for (unsigned i = 21; i <= 22; i++) {
			nextP(i, i) += process_noise(i);
		}

	}

	if (_terrain_initialised) {
		nextP(0,23) = 2*P(0,23)*PS53 + 2*P(1,23)*PS60 + 2*P(10,23)*PS20 - 2*P(11,23)*PS22 + 2*P(2,23)*PS65 - 2*P(9,23)*PS18;
		nextP(1,23) = 2*P(0,23)*PS80 + 2*P(1,23)*PS78 - 2*P(10,23)*PS18 + 2*P(11,23)*PS76 + 2*P(2,23)*PS79 - 2*P(9,23)*PS20;
		nextP(2,23) = 2*P(0,23)*PS85 + 2*P(1,23)*PS84 - 2*P(10,23)*PS76 - 2*P(11,23)*PS18 + 2*P(2,23)*PS83 + 2*P(9,23)*PS22;
		nextP(3,23) = -P(0,23)*PS132 - P(1,23)*PS124 + P(12,23)*PS90 - P(13,23)*PS214 - P(14,23)*PS215 - P(2,23)*PS117 + P(3,23);
		nextP(4,23) = -P(0,23)*PS163 - P(1,23)*PS157 - P(12,23)*PS227 + P(13,23)*PS147 - P(14,23)*PS226 - P(2,23)*PS166 + P(4,23);
		nextP(5,23) = -P(0,23)*PS177 - P(1,23)*PS175 - P(12,23)*PS232 - P(13,23)*PS231 + P(14,23)*PS173 - P(2,23)*PS176 + P(5,23);
		nextP(6,23) = P(3,23)*dt + P(6,23);
		nextP(7,23) = P(4,23)*dt + P(7,23);
		nextP(8,23) = P(5,23)*dt + P(8,23);
		nextP(9,23) = P(9,23);
		nextP(10,23) = P(10,23);
		nextP(11,23) = P(11,23);
		nextP(12,23) = P(12,23);
		nextP(13,23) = P(13,23);
		nextP(14,23) = P(14,23);
		nextP(15,23) = P(15,23);
		nextP(16,23) = P(16,23);
		nextP(17,23) = P(17,23);
		nextP(18,23) = P(18,23);
		nextP(19,23) = P(19,23);
		nextP(20,23) = P(20,23);
		nextP(21,23) = P(21,23);
		nextP(22,23) = P(22,23);
		nextP(23,23) = P(23,23);

		// add process noise that is not from the IMU
		nextP(23, 23) += process_noise(23);
	}

	// stop position covariance growth if our total position variance reaches 100m
	// this can happen if we lose gps for some time
	if ((P(7, 7) + P(8, 8)) > 1e4f) {
		for (uint8_t i = 7; i <= 8; i++) {
			for (uint8_t j = 0; j < _k_num_states; j++) {
				nextP(i, j) = P(i, j);
				nextP(j, i) = P(j, i);
			}
		}
	}

	// covariance matrix is symmetrical, so copy upper half to lower half
	for (unsigned row = 1; row < _k_num_states; row++) {
		for (unsigned column = 0 ; column < row; column++) {
			P(row, column) = P(column, row) = nextP(column, row);
		}
	}

	// copy variances (diagonals)
	for (unsigned i = 0; i < _k_num_states; i++) {
		P(i, i) = nextP(i, i);
	}

	// fix gross errors in the covariance matrix and ensure rows and
	// columns for un-used states are zero
	fixCovarianceErrors(false);

}

void Ekf::fixCovarianceErrors(bool force_symmetry)
{
	// NOTE: This limiting is a last resort and should not be relied on
	// TODO: Split covariance prediction into separate F*P*transpose(F) and Q contributions
	// and set corresponding entries in Q to zero when states exceed 50% of the limit
	// Covariance diagonal limits. Use same values for states which
	// belong to the same group (e.g. vel_x, vel_y, vel_z)
	float P_lim[9] = {};
	P_lim[0] = 1.0f;		// quaternion max var
	P_lim[1] = 1e6f;		// velocity max var
	P_lim[2] = 1e6f;		// positiion max var
	P_lim[3] = 1.0f;		// gyro bias max var
	P_lim[4] = 1.0f;		// delta velocity z bias max var
	P_lim[5] = 1.0f;		// earth mag field max var
	P_lim[6] = 1.0f;		// body mag field max var
	P_lim[7] = 1e6f;		// wind max var
	P_lim[8] = 1E4f;		// terrain vertical position max variance

	for (int i = 0; i <= 3; i++) {
		// quaternion states
		P(i, i) = math::constrain(P(i, i), 0.0f, P_lim[0]);
	}

	for (int i = 4; i <= 6; i++) {
		// NED velocity states
		P(i, i) = math::constrain(P(i, i), 1e-6f, P_lim[1]);
	}

	for (int i = 7; i <= 9; i++) {
		// NED position states
		P(i, i) = math::constrain(P(i, i), 1e-6f, P_lim[2]);
	}

	for (int i = 10; i <= 12; i++) {
		// gyro bias states
		P(i, i) = math::constrain(P(i, i), 0.0f, P_lim[3]);
	}

	// force symmetry on the quaternion, velocity and position state covariances
	if (force_symmetry) {
		P.makeRowColSymmetric<13>(0);
	}

	// the following states are optional and are deactivated when not required
	// by ensuring the corresponding covariance matrix values are kept at zero

	// accelerometer bias states
	if (!_accel_bias_inhibit[0] || !_accel_bias_inhibit[1] || !_accel_bias_inhibit[2]) {
		// Find the maximum delta velocity bias state variance and request a covariance reset if any variance is below the safe minimum
		const float minSafeStateVar = 1e-9f;
		float maxStateVar = minSafeStateVar;
		bool resetRequired = false;

		for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
			if (_accel_bias_inhibit[stateIndex - 13]) {
				// Skip the check for the inhibited axis
				continue;
			}

			if (P(stateIndex, stateIndex) > maxStateVar) {
				maxStateVar = P(stateIndex, stateIndex);

			} else if (P(stateIndex, stateIndex) < minSafeStateVar) {
				resetRequired = true;
			}
		}

		// To ensure stability of the covariance matrix operations, the ratio of a max and min variance must
		// not exceed 100 and the minimum variance must not fall below the target minimum
		// Also limit variance to a maximum equivalent to a 0.1g uncertainty
		const float minStateVarTarget = 5E-8f;
		float minAllowedStateVar = fmaxf(0.01f * maxStateVar, minStateVarTarget);

		for (uint8_t stateIndex = 13; stateIndex <= 15; stateIndex++) {
			if (_accel_bias_inhibit[stateIndex - 13]) {
				// Skip the check for the inhibited axis
				continue;
			}

			P(stateIndex, stateIndex) = math::constrain(P(stateIndex, stateIndex), minAllowedStateVar,
						    sq(0.1f * CONSTANTS_ONE_G * _dt_ekf_avg));
		}

		// If any one axis has fallen below the safe minimum, all delta velocity covariance terms must be reset to zero
		if (resetRequired) {
			P.uncorrelateCovariance<3>(13);
		}

		// Run additional checks to see if the delta velocity bias has hit limits in a direction that is clearly wrong
		// calculate accel bias term aligned with the gravity vector
		const float dVel_bias_lim = 0.9f * _params.acc_bias_lim * _dt_ekf_avg;
		const float down_dvel_bias = _state.delta_vel_bias.dot(Vector3f(_R_to_earth.row(2)));

		// check that the vertical component of accel bias is consistent with both the vertical position and velocity innovation
		bool bad_acc_bias = (fabsf(down_dvel_bias) > dVel_bias_lim
				     && ((down_dvel_bias * _gps_vel_innov(2) < 0.0f && _control_status.flags.gps)
					 || (down_dvel_bias * _ev_vel_innov(2) < 0.0f && _control_status.flags.ev_vel))
				     && ((down_dvel_bias * _gps_pos_innov(2) < 0.0f && _control_status.flags.gps_hgt)
					 || (down_dvel_bias * _baro_hgt_innov < 0.0f && _control_status.flags.baro_hgt)
					 || (down_dvel_bias * _rng_hgt_innov < 0.0f && _control_status.flags.rng_hgt)
					 || (down_dvel_bias * _ev_pos_innov(2) < 0.0f && _control_status.flags.ev_hgt)));

		// record the pass/fail
		if (!bad_acc_bias) {
			_fault_status.flags.bad_acc_bias = false;
			_time_acc_bias_check = _time_last_imu;

		} else {
			_fault_status.flags.bad_acc_bias = true;
		}

		// if we have failed for 7 seconds continuously, reset the accel bias covariances to fix bad conditioning of
		// the covariance matrix but preserve the variances (diagonals) to allow bias learning to continue
		if (isTimedOut(_time_acc_bias_check, (uint64_t)7e6)) {

			P.uncorrelateCovariance<3>(13);

			_time_acc_bias_check = _time_last_imu;
			_fault_status.flags.bad_acc_bias = false;
			_warning_events.flags.invalid_accel_bias_cov_reset = true;
			ECL_WARN("invalid accel bias - covariance reset");

		} else if (force_symmetry) {
			// ensure the covariance values are symmetrical
			P.makeRowColSymmetric<3>(13);
		}

	}

	// magnetic field states
	if (!_control_status.flags.mag_3D) {
		zeroMagCov();

	} else {
		// constrain variances
		for (int i = 16; i <= 18; i++) {
			P(i, i) = math::constrain(P(i, i), 0.0f, P_lim[5]);
		}

		for (int i = 19; i <= 21; i++) {
			P(i, i) = math::constrain(P(i, i), 0.0f, P_lim[6]);
		}

		// force symmetry
		if (force_symmetry) {
			P.makeRowColSymmetric<3>(16);
			P.makeRowColSymmetric<3>(19);
		}

	}

	// wind velocity states
	if (!_control_status.flags.wind) {
		P.uncorrelateCovarianceSetVariance<2>(22, 0.0f);

	} else {
		// constrain variances
		for (int i = 22; i <= 23; i++) {
			P(i, i) = math::constrain(P(i, i), 0.0f, P_lim[7]);
		}

		// force symmetry
		if (force_symmetry) {
			P.makeRowColSymmetric<2>(22);
		}
	}

	// terrain vertical position
	if (!_terrain_initialised) {
		P.uncorrelateCovarianceSetVariance<1>(24, 0.0f);

	} else {
		// constrain variances
		P(24, 24) = math::constrain(P(24, 24), 0.0f, P_lim[8]);

		// force symmetry
		if (force_symmetry) {
			P.makeRowColSymmetric<1>(24);
		}
	}
}

// if the covariance correction will result in a negative variance, then
// the covariance matrix is unhealthy and must be corrected
bool Ekf::checkAndFixCovarianceUpdate(const SquareMatrix25f &KHP)
{
	bool healthy = true;

	for (int i = 0; i < _k_num_states; i++) {
		if (P(i, i) < KHP(i, i)) {
			P.uncorrelateCovarianceSetVariance<1>(i, 0.0f);
			healthy = false;
		}
	}

	return healthy;
}

void Ekf::resetMagRelatedCovariances()
{
	resetQuatCov();
	resetMagCov();
}

void Ekf::resetQuatCov()
{
	zeroQuatCov();

	// define the initial angle uncertainty as variances for a rotation vector
	Vector3f rot_vec_var;
	rot_vec_var.setAll(sq(_params.initial_tilt_err));

	initialiseQuatCovariances(rot_vec_var);
}

void Ekf::zeroQuatCov()
{
	P.uncorrelateCovarianceSetVariance<2>(0, 0.0f);
	P.uncorrelateCovarianceSetVariance<2>(2, 0.0f);
}

void Ekf::resetMagCov()
{
	// reset the corresponding rows and columns in the covariance matrix and
	// set the variances on the magnetic field states to the measurement variance
	clearMagCov();

	P.uncorrelateCovarianceSetVariance<3>(16, sq(_params.mag_noise));
	P.uncorrelateCovarianceSetVariance<3>(19, sq(_params.mag_noise));

	if (!_control_status.flags.mag_3D) {
		// save covariance data for re-use when auto-switching between heading and 3-axis fusion
		// if already in 3-axis fusion mode, the covariances are automatically saved when switching out
		// of this mode
		saveMagCovData();
	}
}

void Ekf::clearMagCov()
{
	zeroMagCov();
	_mag_decl_cov_reset = false;
}

void Ekf::zeroMagCov()
{
	P.uncorrelateCovarianceSetVariance<3>(16, 0.0f);
	P.uncorrelateCovarianceSetVariance<3>(19, 0.0f);
}

void Ekf::resetZDeltaAngBiasCov()
{
	const float init_delta_ang_bias_var = sq(_params.switch_on_gyro_bias * _dt_ekf_avg);

	P.uncorrelateCovarianceSetVariance<1>(12, init_delta_ang_bias_var);
}

void Ekf::resetWindCovarianceUsingAirspeed()
{
	// Derived using EKF/matlab/scripts/Inertial Nav EKF/wind_cov.py
	// TODO: explicitly include the sideslip angle in the derivation
	const float euler_yaw = getEulerYaw(_R_to_earth);
	const float R_TAS = sq(math::constrain(_params.eas_noise, 0.5f, 5.0f) * math::constrain(_airspeed_sample_delayed.eas2tas, 0.9f, 10.0f));
	constexpr float initial_sideslip_uncertainty = math::radians(15.0f);
	const float initial_wind_var_body_y = sq(_airspeed_sample_delayed.true_airspeed * sinf(initial_sideslip_uncertainty));
	constexpr float R_yaw = sq(math::radians(10.0f));

	const float cos_yaw = cosf(euler_yaw);
	const float sin_yaw = sinf(euler_yaw);

	// rotate wind velocity into earth frame aligned with vehicle yaw
	const float Wx = _state.wind_vel(0) * cos_yaw + _state.wind_vel(1) * sin_yaw;
	const float Wy = -_state.wind_vel(0) * sin_yaw + _state.wind_vel(1) * cos_yaw;

	// it is safer to remove all existing correlations to other states at this time
	P.uncorrelateCovarianceSetVariance<2>(22, 0.0f);

	P(22, 22) = R_TAS * sq(cos_yaw) + R_yaw * sq(-Wx * sin_yaw - Wy * cos_yaw) + initial_wind_var_body_y * sq(sin_yaw);
	P(22, 23) = R_TAS * sin_yaw * cos_yaw + R_yaw * (-Wx * sin_yaw - Wy * cos_yaw) * (Wx * cos_yaw - Wy * sin_yaw) -
		    initial_wind_var_body_y * sin_yaw * cos_yaw;
	P(23, 22) = P(22, 23);
	P(23, 23) = R_TAS * sq(sin_yaw) + R_yaw * sq(Wx * cos_yaw - Wy * sin_yaw) + initial_wind_var_body_y * sq(cos_yaw);

	// Now add the variance due to uncertainty in vehicle velocity that was used to calculate the initial wind speed
	P(22, 22) += P(4, 4);
	P(23, 23) += P(5, 5);
}
