/****************************************************************************
 *
 *   Copyright (c) 2019-2020 PX4 Development Team. All rights reserved.
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
 * 3. Neither the name PX4 nor the names of its contributors may be
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
 * @file sih.cpp
 * Simulator in Hardware
 *
 * @author Romain Chiappinelli      <romain.chiap@gmail.com>
 *
 * Coriolis g Corporation - January 2019
 */

#include "sih.hpp"

#include <px4_platform_common/getopt.h>
#include <px4_platform_common/log.h>

#include <drivers/drv_pwm_output.h>         // to get PWM flags

using namespace math;
using namespace matrix;
using namespace time_literals;

Sih::Sih() :
	ModuleParams(nullptr),
	ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::rate_ctrl)
{
	_px4_accel.set_temperature(T1_C);
	_px4_gyro.set_temperature(T1_C);
	_px4_mag.set_temperature(T1_C);

	parameters_updated();
	init_variables();
	gps_no_fix();

	const hrt_abstime task_start = hrt_absolute_time();
	_last_run = task_start;
	_gps_time = task_start;
	_gt_time = task_start;
	_dist_snsr_time = task_start;
}

Sih::~Sih()
{
	perf_free(_loop_perf);
	perf_free(_loop_interval_perf);
}

bool Sih::init()
{
	int rate = _imu_gyro_ratemax.get();

	// default to 250 Hz (4000 us interval)
	if (rate <= 0) {
		rate = 250;
	}

	// 200 - 2000 Hz
	int interval_us = math::constrain(int(roundf(1e6f / rate)), 500, 5000);
	ScheduleOnInterval(interval_us);

	return true;
}

void Sih::Run()
{
	perf_count(_loop_interval_perf);

	// check for parameter updates
	if (_parameter_update_sub.updated()) {
		// clear update
		parameter_update_s pupdate;
		_parameter_update_sub.copy(&pupdate);

		// update parameters from storage
		updateParams();
		parameters_updated();
	}

	perf_begin(_loop_perf);

	_now = hrt_absolute_time();
	_dt = (_now - _last_run) * 1e-6f;
	_last_run = _now;

	read_motors();

	if (_sih_type == 1) {
		generate_fw_force_and_torques();
	} else {
		generate_mc_force_and_torques();
	}

	equations_of_motion();

	reconstruct_sensors_signals();

	// update IMU every iteration
	_px4_accel.update(_now, _acc(0), _acc(1), _acc(2));
	_px4_gyro.update(_now, _gyro(0), _gyro(1), _gyro(2));

	// magnetometer published at 50 Hz
	if (_now - _mag_time >= 20_ms
	    && fabs(_mag_offset_x) < 10000
	    && fabs(_mag_offset_y) < 10000
	    && fabs(_mag_offset_z) < 10000) {
		_mag_time = _now;
		_px4_mag.update(_now, _mag(0), _mag(1), _mag(2));
	}

	// baro published at 20 Hz
	if (_now - _baro_time >= 50_ms
	    && fabs(_baro_offset_m) < 10000) {
		_baro_time = _now;
		_px4_baro.set_temperature(_baro_temp_c);
		_px4_baro.update(_now, _baro_p_mBar);
	}

	// gps published at 20Hz
	if (_now - _gps_time >= 50_ms) {
		_gps_time = _now;
		send_gps();
	}

	// distance sensor published at 50 Hz
	if (_now - _dist_snsr_time >= 20_ms
	    && fabs(_distance_snsr_override) < 10000) {
		_dist_snsr_time = _now;
		send_dist_snsr();
	}

	// send groundtruth message every 40 ms
	if (_now - _gt_time >= 40_ms) {
		_gt_time = _now;

		publish_sih();  // publish _sih message for debug purpose
	}

	perf_end(_loop_perf);
}

// store the parameters in a more convenient form
void Sih::parameters_updated()
{
	if (_sih_type == 1) {
		// 250 gram flying wing with single motor and two elevons
		_MASS = 0.25f;
		_span = 0.58;
		_chord = 0.16f;
		const float Ixx = (1.0f / 12.0f) * _MASS * powf(0.3f, 2);
		const float Iyy = 1.0f * Ixx;
		const float Izz = 1.5f * Ixx;
		_I = diag(Vector3f(Ixx,Iyy,Izz);
		_area = _span * _chord;
		_max_static_thrust = 0.7f * _MASS * CONSTANTS_ONE_G;
		_zero_thrust_speed = 30.0f;

		_cfz_max = 0.7f;
		_cfz_min = -1.1f;
		_cfz_stall_break = 0.3f;
		_cfz_0 = 0.1f;
		_cfz_90 = 1.0f;
		const float AR = _span / _chord;
		_cfz_aoa = - (M_PI * AR) / (1.0f + sqrtf(1.0f + powf((M_PI * AR) / 6.0f , 2)));
		_cmy_aoa = _cfz_aoa * 0.02f; // based on c.g. 3 mm in front of neutral point
		_cfx_0 = -0.035f; // based on best L/D of 8 at 12 m/s and a CL of 0.55
		_cfx_stall = -0.05;
		_cfx_elev = -0.7;
		_cmy_aoa_dot = M_PI_4_F * 0.9f; // based on theoretical result for 2-D airfoil with knockdown for boundary layer effects

		// based on steady state (pitch_rate * chord) / (2 * flight speed) = 0.1 x elevator deflection
		_cmy_elev = (_chord / _cmy_aoa_dot * 0.1f;
		_cmy_ratio = 0.28f; // ratio of y moment to z force increment produced by deflecting the elevator - Roskam Part VI, Fig 8.166
		_cfz_elev = _cmy_elev / 0.28f; // based on 25% chord elevons and c.p. generated by  each elevon being at hinge line and therfore 50% of MAC behind the neutral point

		_cmx_ail = (0.18f/_span) * _cfz_elev; // based on c.p. of elevons  being 0.18m outboard
		_cmx_roll_rate = - _cmx_ail / 0.12f; // based on steady state (roll_rate * span) / (2 * flight speed) = 0.12 x aileron deflection

		// Calculate yaw moment due to sideslip based on a measured yaw oscillations in 4.5 seconds at 15 m/s flight speed at S.L.
		// Calculate yaw damping moment assuming a lightly damped 1 cycle to half amplitude
		const float yaw_freq = M_TWOPI * (10.0f / 4.5f);
		const float Vref = 15.0f;
		const float q_ref = 0.5f * 1.225f * powf(Vref, 2);
		const float yaw_stiffness = powf(yaw_freq,2) * Izz;
		_cmz_aos = yaw_stiffness / (q_ref * _area * _span);
		const float viscous_damping_ratio = 0.2f;
		const float yaw_damping = 2.0f * viscous_damping_ratio * yaw_freq * Izz;
		_cmz_aos_dot = (yaw_damping * Vref) / (q_ref * _area * _span);

		// Calculate sideslip to sideforce derivative using a test data point that showed that at 20 m/s flight speed, a +=
		// 2 deg sideslip oscillation produces a +- 1 m/s/s lateral acceleration
		_cfy_aos = - (1.0f * _MASS) / (q_ref * _area * math::radians(2.0f));
		_cfy_lim = _cfy_aos * radians(-15.0f);

		// full aileron or full elevator produces 10mm of deflection measured at the trailing edge 45mm from the hinge line
		// and  these sum equally in the elevon mixer
		_elevon_max = 2.0f * asinf(10.0f/45.0f);

	} else {
		// MC vehicle type is the default
		_T_MAX = _sih_t_max.get();
		_Q_MAX = _sih_q_max.get();
		_L_ROLL = _sih_l_roll.get();
		_L_PITCH = _sih_l_pitch.get();
		_KDV = _sih_kdv.get();
		_KDW = _sih_kdw.get();
		_H0 = _sih_h0.get();
		_MASS = _sih_mass.get();
		_I = diag(Vector3f(_sih_ixx.get(), _sih_iyy.get(), _sih_izz.get()));
		_I(0, 1) = _I(1, 0) = _sih_ixy.get();
		_I(0, 2) = _I(2, 0) = _sih_ixz.get();
		_I(1, 2) = _I(2, 1) = _sih_iyz.get();

	}

	_LAT0 = (double)_sih_lat0.get() * 1.0e-7;
	_LON0 = (double)_sih_lon0.get() * 1.0e-7;
	_COS_LAT0 = cosl((long double)radians(_LAT0));

	_W_I = Vector3f(0.0f, 0.0f, _MASS * CONSTANTS_ONE_G);


	_Im1 = inv(_I);

	_mu_I = Vector3f(_sih_mu_x.get(), _sih_mu_y.get(), _sih_mu_z.get());

	_gps_used = _sih_gps_used.get();
	_baro_offset_m = _sih_baro_offset.get();
	_mag_offset_x = _sih_mag_offset_x.get();
	_mag_offset_y = _sih_mag_offset_y.get();
	_mag_offset_z = _sih_mag_offset_z.get();

	_distance_snsr_min = _sih_distance_snsr_min.get();
	_distance_snsr_max = _sih_distance_snsr_max.get();
	_distance_snsr_override = _sih_distance_snsr_override.get();

	_T_TAU = _sih_thrust_tau.get();
}

// initialization of the variables for the simulator
void Sih::init_variables()
{
	srand(1234);    // initialize the random seed once before calling generate_wgn()

	_p_I = Vector3f(0.0f, 0.0f, 0.0f);
	_v_I = Vector3f(0.0f, 0.0f, 0.0f);
	_q = Quatf(1.0f, 0.0f, 0.0f, 0.0f);
	_w_B = Vector3f(0.0f, 0.0f, 0.0f);

	_u[0] = _u[1] = _u[2] = _u[3] = 0.0f;
}

void Sih::gps_fix()
{
	_sensor_gps.fix_type = 3;  // 3D fix
	_sensor_gps.satellites_used = _gps_used;
	_sensor_gps.heading = NAN;
	_sensor_gps.heading_offset = NAN;
	_sensor_gps.s_variance_m_s = 0.5f;
	_sensor_gps.c_variance_rad = 0.1f;
	_sensor_gps.eph = 0.9f;
	_sensor_gps.epv = 1.78f;
	_sensor_gps.hdop = 0.7f;
	_sensor_gps.vdop = 1.1f;
}

void Sih::gps_no_fix()
{
	_sensor_gps.fix_type = 0;  // 3D fix
	_sensor_gps.satellites_used = _gps_used;
	_sensor_gps.heading = NAN;
	_sensor_gps.heading_offset = NAN;
	_sensor_gps.s_variance_m_s = 100.f;
	_sensor_gps.c_variance_rad = 100.f;
	_sensor_gps.eph = 100.f;
	_sensor_gps.epv = 100.f;
	_sensor_gps.hdop = 100.f;
	_sensor_gps.vdop = 100.f;
}


// read the motor signals outputted from the mixer
void Sih::read_motors()
{
	actuator_outputs_s actuators_out;

	if (_actuator_out_sub.update(&actuators_out)) {
		for (int i = 0; i < NB_MOTORS; i++) { // saturate the motor signals
			float u_sp = constrain((actuators_out.output[i] - PWM_DEFAULT_MIN) / (PWM_DEFAULT_MAX - PWM_DEFAULT_MIN), 0.0f, 1.0f);
			_u[i] = _u[i] + _dt / _T_TAU * (u_sp - _u[i]); // first order transfer function with time constant tau
		}
	}
}

// generate the multi copter motors thrust and torque in the body frame
void Sih::generate_mc_force_and_torques()
{
	_T_B = Vector3f(0.0f, 0.0f, -_T_MAX * (+_u[0] + _u[1] + _u[2] + _u[3]));
	_Mt_B = Vector3f(_L_ROLL * _T_MAX * (-_u[0] + _u[1] + _u[2] - _u[3]),
			 _L_PITCH * _T_MAX * (+_u[0] - _u[1] + _u[2] - _u[3]),
			 _Q_MAX * (+_u[0] + _u[1] - _u[2] - _u[3]));

	_Fa_I = -_KDV * _v_I;   // first order drag to slow down the aircraft
	_Ma_B = -_KDW * _w_B;   // first order angular damper
	_Fa_B.setZero();
}

// generate the fixed wing vehicles forces and moments in the body frame.
void Sih::generate_fw_force_and_torques()
{
	// read elevon servos and motor throttle
	actuator_outputs_s actuators_out;
	if (_actuator_out_sub.update(&actuators_out)) {
		float u_sp[3];
		static constexpr float pwm_center = (PWM_DEFAULT_MAX + PWM_DEFAULT_MIN) / 2;
		// read LH (index 0) and RH (index 1) elevon servo demands
		for (int i = 0; i < 2; i++) {
			u_sp[i] = math::constrain(2.0f * (actuators_out.output[i] - pwm_center) / (PWM_DEFAULT_MAX - PWM_DEFAULT_MIN), -1.0f, 1.0f);
		}
		// read throttle ESC demand
		u_sp[2] = math::constrain((actuators_out.output[2] - PWM_DEFAULT_MIN) / (PWM_DEFAULT_MAX - PWM_DEFAULT_MIN), 0.0f, 1.0f);
		// first order transfer function with time constant tau used to approximate propeller acceleration curve
		_u[2] = _u[2] + _dt / _T_TAU * (u_sp[2] - _u[2]);
		// TODO second order elevon actuator model
	}

	// convert elevons to equivalent elevator and aileron deflections assuming a positive deflection is TE up
	const float elevator = 0.5f * (_u[0] + _u[1]) * _elevon_max;
	const float aileron = 0.5f * ( - _u[0] + _u[1]) * _elevon_max;

	// calculate thrust assuming it reduces linearly with airspeed
	const float thrust = _u[2] * _max_static_thrust * (1.0f - _v_B(0) / _zero_thrust_speed);
	_T_B = Vector3f(thrust, 0.0f, 0.0f);

	// calculate dynamic pressure, AoA and AoS
	// TODO vary air dnesity with altitude
	const float q_ref = 0.5 * CONSTANTS_AIR_DENSITY_SEA_LEVEL_15C * powf(_v_B.length(),2);
	const float aoa_rad = atan2f(_v_B(2),_v_B(0));
	const float aos_rad = atan2f(_v_B(1),_v_B(0));
	const float aoa_dot = _w_B(1);
	const float tas_xy = sqrtf(_V_b(0) * _V_b(0) + _V_b(1) * _V_b(1));
	const float roll_rate_dimensionless = _w_B(0) * (_span / fmaxf(2.0f * tas_xy, _span));
	const float aos_dot_dimensionless = -_w_B(1) * (_span / fmaxf(2.0f * tas_xy, _span));

	// calculate longitudinal force and moment coefficients
	float cfx,cfz,cmy;
	calc_wing_coefficients(cfx, cfz, cmy, aoa_rad, aoa_dot, elevator);

	// calculate rolling moment coefficient
	const float cmx = q_ref * _area * _span * (_cmx_ail * aileron + _cmx_roll_rate * roll_rate_dimensionless);

	// calculate yaw moment coefficient
	const float cmz = q_ref * _area * _span * (_cmz_aos * aos_rad + _cmz_aos_dot * aos_dot_dimensionless);

	// calculate side force coefficient
	const float cfy = math::constrain(q_ref * _area * _span * _cfy_aos * aos_rad, - _cfy_lim, _cfy_lim);

	// sum aero forces
	_Fa_B = q_ref * _area * Vector3f(cfx, cfy, cfz);

	// sum aero moments
	_Ma_B = q_ref * _area * Vector3f(cmx * _span, cmy * _chord, cmz * _span);

	_Mt_B = Vector3f(_L_ROLL * _T_MAX * (-_u[0] + _u[1] + _u[2] - _u[3]),
			 _L_PITCH * _T_MAX * (+_u[0] - _u[1] + _u[2] - _u[3]),
			 _Q_MAX * (+_u[0] + _u[1] - _u[2] - _u[3]));

	_Fa_I.setZero();
}

void Sih::calc_wing_coefficients(float &cfx, float &cfz, float &cmy, const float aoa, const float aoa_dot, const float elevator)
{
    // hack to allow scaling of lift curve slope
    const float cfz_alpha_scaled =  _cfz_aoa;

    // calculate negative aoa for stall onset
    const float aoa_bp_1 = (_cfz_max - _cfz_stall_break - _cfz_0) / _cfz_aoa;

    // calculate positive aoa for stall onset
    const float aoa_bp_2 = (_cfz_min + _cfz_stall_break - _cfz_0) / _cfz_aoa;

    // calculate parameters controlling wdfth of stall break region
    const float k_aoa = - _cfz_aoa / _cfz_stall_break;
    const float aoa_stall_break = M_PI / k_aoa;

    // calculate negative aoa for fully developed stall
    const float aoa_bp_0 = aoa_bp_1 - aoa_stall_break;

    // calculate positive aoa for fully developed stall
    const float aoa_bp_3 = aoa_bp_2 + aoa_stall_break;

    float cfz_aoa; // z force coeficient due to angle of attack
    float cop_aoa; // centre of pressure as a fraction of chord measured back from l.e. for cfz_aoa force
    float cfz_elev; // z force coeficient due to elevator
    float cmy_aoa_dot; // Y moment coefficient due to angle of attack rate (dimensionless)
    const float sine_aoa = sinf(aoa);
    const float cos_aoa = cosf(aoa);
    if (aoa < -M_PI_2_F) {
        // in reverse flow region assume flat plate characteristics and use cosine law
        cfz_aoa = - _cfz_90 * sine_aoa;
        // c.p. moves aft to 75% m.a.c. for reverse flow at low angles of attack
        cop_aoa = 0.75f + 0.25f * sine_aoa;
        // control surface works in 'reverse'
        cfz_elev = - 0.5f * _cfz_elev * elevator * cos_aoa;
        // pitch rate damping scales the same as control effectiveness
        cmy_aoa_dot = - _cmy_aoa_dot * cos_aoa;
        // significant X axis wake drag when flying in reverse so use post stall value
        cfx = - _cfx_stall;
    } else if (aoa < aoa_bp_0) {
        // negative alpha post stall - linearly interpolate to flat plate drag value at -90 deg aoa
        const float fraction = (aoa_bp_0 - aoa) / (M_PI_2_F + aoa_bp_0);
        cfz_aoa = (1.0f - fraction) * _cfz_max + fraction * _cfz_90;
        // c.p. moves aft to 50% m.a.c. as angle of attack approaches -90 deg
        cop_aoa = 0.25f - 0.25f * sine_aoa;
        // assume control surface deflection doesn't work in the post stall region
        cfz_elev = 0.0f;
        // no pitch rate damping in this region
        cmy_aoa_dot = 0.0f;
        // use post stall value
        cfx = _cfx_stall;
    } else if (aoa < aoa_bp_1) {
        // negative alpha stall region - model cfz vs alpha using a sine curve in this region but without post stall normal force droppoff
        const float angle = fmaxf(k_aoa * (aoa - aoa_bp_0), M_PI_2_F);
        cfz_aoa = _cfz_max - _cfz_stall_break * (1.0f - sinf(angle));
        // c.p moves aft during stall onset
        const float fraction = (aoa - aoa_bp_0) / aoa_stall_break;
        const float cop_stalled = 0.25f - 0.25f * sinf(aoa_bp_0);
        cop_aoa = 0.25f * fraction + cop_stalled * (1.0f - fraction);
        // control effectiveness reduces linearly to zero at aoa_bp_0
        cfz_elev = (elevator * _cfz_elev) * fraction;
        // pitch rate damping scales the same as control effectiveness
        cmy_aoa_dot = _cmy_aoa_dot * fraction;
        // x axis wake drag increases with amount of flow separation and surface deflection
        cfx = _cfx_0 * fraction + _cfx_stall * (1.0f - fraction);
        cfx += fraction * _cfx_elev * fabsf(elevator / radians(30.0f));
    } else if (aoa < aoa_bp_2) {
        // linear region
        cfz_aoa = _cfz_max - _cfz_stall_break + (aoa - aoa_bp_1) * cfz_alpha_scaled;
        cop_aoa = 0.25f;
        cfz_elev = elevator * _cfz_elev;
        cmy_aoa_dot = _cmy_aoa_dot;
        // X axis wake drag increases exponentially with amount of surface deflection
        cfx = _cfx_0 + _cfx_elev * fabsf(elevator / radians(30.0f));
    } else if (aoa < aoa_bp_3) {
        // positive alpha stall region - model cfz vs alpha using a sine curve in this region
        const float angle = k_aoa * (aoa - aoa_bp_2);
        cfz_aoa = _cfz_min + _cfz_stall_break * (1.0f - sinf(angle));
        // c.p moves aft during stall onset
        const float fraction = (aoa_bp_3 - aoa) / aoa_stall_break;
        const float cop_stalled = 0.25f + 0.25f * sinf(aoa_bp_3);
        cop_aoa = 0.25f * fraction + cop_stalled * (1.0f - fraction);
        // control effectiveness reduces linearly to zero at aoa_bp_3
        cfz_elev = (elevator * _cfz_elev) * fraction;
        // pitch rate damping scales the same as control effectiveness
        cmy_aoa_dot = _cmy_aoa_dot * fraction;
        // x axis wake drag increases with amount of flow separation and surface deflection
        cfx = _cfx_0 * fraction + _cfx_stall * (1.0f - fraction);
        cfx += fraction * _cfx_elev * fabsf(elevator / radians(30.0f));
    } else if (aoa < M_PI_2_F) {
        // positive alpha post stall - linearly interpolate to flat plate drag value at 90 deg aoa
        const float fraction = (aoa - aoa_bp_3) / (M_PI_2_F - aoa_bp_3);
        cfz_aoa = (1.0f - fraction) * (_cfz_min + _cfz_stall_break) - fraction * _cfz_90;
        // c.p. moves aft to 50% m.a.c. as angle of attack approaches 90 deg
        cop_aoa = 0.25f + 0.25f * sine_aoa;
        // assume control surface deflection doesn't work in the post stall region
        cfz_elev = 0.0f;
        // no pitch rate damping in this region
        cmy_aoa_dot = 0.0f;
        // use post stall value
        cfx = _cfx_stall;
    } else {
        // in reverse flow region assume flat plate characteristics and use cosine law
        cfz_aoa = - _cfz_90 * sinf(aoa);
        // c.p. moves aft to 75% m.a.c. for reverse flow at low angles of attack
        cop_aoa = 0.75f - 0.25f * sine_aoa;
        // control surface works in 'reverse'
        cfz_elev = - 0.5f * _cfz_elev * elevator * cos_aoa;
        // pitch rate damping scales the same as control effectiveness
        cmy_aoa_dot = - _cmy_aoa_dot * cos_aoa;
        // significant X axis wake drag when flying in reverse so use post stall value
        cfx = - _cfx_stall;
    }

    // calculate pitching moment coefficient due to aoa induced normal force
    const float cmy_aoa = cfz_aoa * (cop_aoa - 0.25f);

    // calculate pitching moment coefficient due to control surface force
    const float cmy_delta = cfz_elev * _cmy_ratio;

    // sum contributions to normal force and pitching moment coefficient
    cfz = cfz_aoa + cfz_elev;
    const float tas_xz = sqrtf(_V_b(0) * _V_b(0) + _V_b(2) * _V_b(2));
    const float aoa_dot_dimensionless = aoa_dot * (_chord / fmaxf(2.0f * tas_xz, _chord));
    cmy = cmy_aoa + cmy_delta + cmy_aoa_dot * aoa_dot_dimensionless;
}

// apply the equations of motion of a rigid body and integrate one step
void Sih::equations_of_motion()
{
	_C_IB = matrix::Dcm<float>(_q); // body to inertial transformation

	// Equations of motion of a rigid body
	_p_I_dot = _v_I;                        // position differential
	_v_I_dot = (_W_I + _Fa_I + _C_IB * _T_B) / _MASS;   // conservation of linear momentum
	_q_dot = _q.derivative1(_w_B);              // attitude differential
	_w_B_dot = _Im1 * (_Mt_B + _Ma_B - _w_B.cross(_I * _w_B)); // conservation of angular momentum

	// fake ground, avoid free fall
	if (_p_I(2) > 0.0f && (_v_I_dot(2) > 0.0f || _v_I(2) > 0.0f)) {
		if (!_grounded) {    // if we just hit the floor
			// for the accelerometer, compute the acceleration that will stop the vehicle in one time step
			_v_I_dot = -_v_I / _dt;

		} else {
			_v_I_dot.setZero();
		}

		_v_I.setZero();
		_v_B.setZero();
		_w_B.setZero();
		_grounded = true;

	} else {
		// integration: Euler forward
		_p_I = _p_I + _p_I_dot * _dt;
		_v_I = _v_I + _v_I_dot * _dt;
		_v_B = _C_IB.transpose() * _v_I;
		_q = _q + _q_dot * _dt; // as given in attitude_estimator_q_main.cpp
		_q.normalize();
		_w_B = _w_B + _w_B_dot * _dt;
		_grounded = false;
	}
}

// reconstruct the noisy sensor signals
void Sih::reconstruct_sensors_signals()
{
	// The sensor signals reconstruction and noise levels are from
	// Bulka, Eitan, and Meyer Nahon. "Autonomous fixed-wing aerobatics: from theory to flight."
	// In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 6573-6580. IEEE, 2018.

	// IMU
	_acc = _C_IB.transpose() * (_v_I_dot - Vector3f(0.0f, 0.0f, CONSTANTS_ONE_G)) + noiseGauss3f(0.5f, 1.7f, 1.4f);
	_gyro = _w_B + noiseGauss3f(0.14f, 0.07f, 0.03f);
	_mag = _C_IB.transpose() * _mu_I + noiseGauss3f(0.02f, 0.02f, 0.03f);
	_mag(0) += _mag_offset_x;
	_mag(1) += _mag_offset_y;
	_mag(2) += _mag_offset_z;

	// barometer
	float altitude = (_H0 - _p_I(2)) + _baro_offset_m + generate_wgn() * 0.14f; // altitude with noise
	_baro_p_mBar = CONSTANTS_STD_PRESSURE_MBAR *        // reconstructed pressure in mBar
		       powf((1.0f + altitude * TEMP_GRADIENT / T1_K), -CONSTANTS_ONE_G / (TEMP_GRADIENT * CONSTANTS_AIR_GAS_CONST));
	_baro_temp_c = T1_K + CONSTANTS_ABSOLUTE_NULL_CELSIUS + TEMP_GRADIENT * altitude; // reconstructed temperture in celcius

	// GPS
	_gps_lat_noiseless = _LAT0 + degrees((double)_p_I(0) / CONSTANTS_RADIUS_OF_EARTH);
	_gps_lon_noiseless = _LON0 + degrees((double)_p_I(1) / CONSTANTS_RADIUS_OF_EARTH) / _COS_LAT0;
	_gps_alt_noiseless = _H0 - _p_I(2);

	_gps_lat = _gps_lat_noiseless + degrees((double)generate_wgn() * 0.2 / CONSTANTS_RADIUS_OF_EARTH);
	_gps_lon = _gps_lon_noiseless + degrees((double)generate_wgn() * 0.2 / CONSTANTS_RADIUS_OF_EARTH) / _COS_LAT0;
	_gps_alt = _gps_alt_noiseless + generate_wgn() * 0.5f;
	_gps_vel = _v_I + noiseGauss3f(0.06f, 0.077f, 0.158f);
}

void Sih::send_gps()
{
	_sensor_gps.timestamp = _now;
	_sensor_gps.lat = (int32_t)(_gps_lat * 1e7);       // Latitude in 1E-7 degrees
	_sensor_gps.lon = (int32_t)(_gps_lon * 1e7); // Longitude in 1E-7 degrees
	_sensor_gps.alt = (int32_t)(_gps_alt * 1000.0f); // Altitude in 1E-3 meters above MSL, (millimetres)
	_sensor_gps.alt_ellipsoid = (int32_t)(_gps_alt * 1000); // Altitude in 1E-3 meters bove Ellipsoid, (millimetres)
	_sensor_gps.vel_ned_valid = true;              // True if NED velocity is valid
	_sensor_gps.vel_m_s = sqrtf(_gps_vel(0) * _gps_vel(0) + _gps_vel(1) * _gps_vel(
					    1)); // GPS ground speed, (metres/sec)
	_sensor_gps.vel_n_m_s = _gps_vel(0);           // GPS North velocity, (metres/sec)
	_sensor_gps.vel_e_m_s = _gps_vel(1);           // GPS East velocity, (metres/sec)
	_sensor_gps.vel_d_m_s = _gps_vel(2);           // GPS Down velocity, (metres/sec)
	_sensor_gps.cog_rad = atan2(_gps_vel(1),
				    _gps_vel(0)); // Course over ground (NOT heading, but direction of movement), -PI..PI, (radians)

	if (_gps_used >= 4) {
		gps_fix();

	} else {
		gps_no_fix();
	}

	_sensor_gps_pub.publish(_sensor_gps);
}

void Sih::send_dist_snsr()
{
	_distance_snsr.timestamp = _now;
	_distance_snsr.type = distance_sensor_s::MAV_DISTANCE_SENSOR_LASER;
	_distance_snsr.orientation = distance_sensor_s::ROTATION_DOWNWARD_FACING;
	_distance_snsr.min_distance = _distance_snsr_min;
	_distance_snsr.max_distance = _distance_snsr_max;
	_distance_snsr.signal_quality = -1;
	_distance_snsr.device_id = 0;

	if (_distance_snsr_override >= 0.f) {
		_distance_snsr.current_distance = _distance_snsr_override;

	} else {
		_distance_snsr.current_distance = -_p_I(2) / _C_IB(2, 2);

		if (_distance_snsr.current_distance > _distance_snsr_max) {
			// this is based on lightware lw20 behaviour
			_distance_snsr.current_distance = UINT16_MAX / 100.f;

		}
	}

	_distance_snsr_pub.publish(_distance_snsr);
}

void Sih::publish_sih()
{
	// publish angular velocity groundtruth
	_vehicle_angular_velocity_gt.timestamp = hrt_absolute_time();
	_vehicle_angular_velocity_gt.xyz[0] = _w_B(0); // rollspeed;
	_vehicle_angular_velocity_gt.xyz[1] = _w_B(1); // pitchspeed;
	_vehicle_angular_velocity_gt.xyz[2] = _w_B(2); // yawspeed;

	_vehicle_angular_velocity_gt_pub.publish(_vehicle_angular_velocity_gt);

	// publish attitude groundtruth
	_att_gt.timestamp = hrt_absolute_time();
	_att_gt.q[0] = _q(0);
	_att_gt.q[1] = _q(1);
	_att_gt.q[2] = _q(2);
	_att_gt.q[3] = _q(3);

	_att_gt_pub.publish(_att_gt);

	// publish position groundtruth
	_gpos_gt.timestamp = hrt_absolute_time();
	_gpos_gt.lat = _gps_lat_noiseless;
	_gpos_gt.lon = _gps_lon_noiseless;
	_gpos_gt.alt = _gps_alt_noiseless;

	_gpos_gt_pub.publish(_gpos_gt);
}

float Sih::generate_wgn()   // generate white Gaussian noise sample with std=1
{
	// algorithm 1:
	// float temp=((float)(rand()+1))/(((float)RAND_MAX+1.0f));
	// return sqrtf(-2.0f*logf(temp))*cosf(2.0f*M_PI_F*rand()/RAND_MAX);
	// algorithm 2: from BlockRandGauss.hpp
	static float V1, V2, S;
	static bool phase = true;
	float X;

	if (phase) {
		do {
			float U1 = (float)rand() / RAND_MAX;
			float U2 = (float)rand() / RAND_MAX;
			V1 = 2.0f * U1 - 1.0f;
			V2 = 2.0f * U2 - 1.0f;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1.0f || fabsf(S) < 1e-8f);

		X = V1 * float(sqrtf(-2.0f * float(logf(S)) / S));

	} else {
		X = V2 * float(sqrtf(-2.0f * float(logf(S)) / S));
	}

	phase = !phase;
	return X;
}

// generate white Gaussian noise sample vector with specified std
Vector3f Sih::noiseGauss3f(float stdx, float stdy, float stdz)
{
	return Vector3f(generate_wgn() * stdx, generate_wgn() * stdy, generate_wgn() * stdz);
}

int Sih::task_spawn(int argc, char *argv[])
{
	Sih *instance = new Sih();

	if (instance) {
		_object.store(instance);
		_task_id = task_id_is_work_queue;

		if (instance->init()) {
			return PX4_OK;
		}

	} else {
		PX4_ERR("alloc failed");
	}

	delete instance;
	_object.store(nullptr);
	_task_id = -1;

	return PX4_ERROR;
}

int Sih::custom_command(int argc, char *argv[])
{
	return print_usage("unknown command");
}

int Sih::print_usage(const char *reason)
{
	if (reason) {
		PX4_WARN("%s\n", reason);
	}

	PRINT_MODULE_DESCRIPTION(
		R"DESCR_STR(
### Description
This module provide a simulator for quadrotors running fully
inside the hardware autopilot.

This simulator subscribes to "actuator_outputs" which are the actuator pwm
signals given by the mixer.

This simulator publishes the sensors signals corrupted with realistic noise
in order to incorporate the state estimator in the loop.

### Implementation
The simulator implements the equations of motion using matrix algebra.
Quaternion representation is used for the attitude.
Forward Euler is used for integration.
Most of the variables are declared global in the .hpp file to avoid stack overflow.


)DESCR_STR");

    PRINT_MODULE_USAGE_NAME("sih", "simulation");
    PRINT_MODULE_USAGE_COMMAND("start");
    PRINT_MODULE_USAGE_DEFAULT_COMMANDS();

    return 0;
}

extern "C" __EXPORT int sih_main(int argc, char *argv[])
{
	return Sih::main(argc, argv);
}
