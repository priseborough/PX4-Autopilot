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
 * @file terrain_estimator.cpp
 * Function for fusing rangefinder measurements to estimate terrain vertical position/
 *
 * @author Paul Riseborough <p_riseborough@live.com.au>
 *
 */

#include "ekf.h"

#include <mathlib/mathlib.h>

bool Ekf::initHagl()
{
	bool initialized = false;

	if (!_control_status.flags.in_air) {
		// if on ground, do not trust the range sensor, but assume a ground clearance
		_state.posd_terrain = _state.pos(2) + _params.rng_gnd_clearance;
		// use the ground clearance value as our uncertainty
		P(24,24) = sq(_params.rng_gnd_clearance);
		_time_last_fake_hagl_fuse = _time_last_imu;
		initialized = true;

	} else if (shouldUseRangeFinderForHagl()
		   && _range_sensor.isDataHealthy()) {
		// if we have a fresh measurement, use it to initialise the terrain estimator
		_state.posd_terrain = _state.pos(2) + _range_sensor.getDistBottom();
		// initialise state variance to variance of measurement
		P(24,24) = sq(_params.range_noise);
		// success
		initialized = true;

	} else if (shouldUseOpticalFlowForHagl()
		   && _flow_for_terrain_data_ready) {
		// initialise terrain vertical position to origin as this is the best guess we have
		_state.posd_terrain = fmaxf(0.0f,  _state.pos(2));
		P(24,24) = 100.0f;
		initialized = true;

	} else {
		// no information - cannot initialise
	}

	if (initialized) {
		// has initialized with valid data
		_time_last_hagl_fuse = _time_last_imu;
	}

	return initialized;
}

void Ekf::updateTerrainValidity()
{
	// we have been fusing range finder measurements in the last 5 seconds
	const bool recent_range_fusion = isRecent(_time_last_hagl_fuse, (uint64_t)5e6);

	// we have been fusing optical flow measurements for terrain estimation within the last 5 seconds
	// this can only be the case if the main filter does not fuse optical flow
	const bool recent_flow_for_terrain_fusion = isRecent(_time_last_flow_terrain_fuse, (uint64_t)5e6);

	_hagl_valid = (_terrain_initialised && (recent_range_fusion || recent_flow_for_terrain_fusion));

	_hagl_sensor_status.flags.range_finder = shouldUseRangeFinderForHagl()
			&& recent_range_fusion && (_time_last_fake_hagl_fuse != _time_last_hagl_fuse);

	_hagl_sensor_status.flags.flow = shouldUseOpticalFlowForHagl() && recent_flow_for_terrain_fusion;
}

void Ekf::fuseHaglAllStates()
{
	// get a height above ground measurement from the range finder assuming a flat earth
	const float meas_hagl = _range_sensor.getDistBottom();

	// predict the hagl from the vehicle position and terrain height
	const float pred_hagl = _state.posd_terrain - _state.pos(2);

	// calculate the innovation
	_hagl_innov = pred_hagl - meas_hagl;

	// calculate the observation variance adding the variance of the vehicles own height uncertainty
	const float obs_variance = fmaxf(P(9, 9) * _params.vehicle_variance_scaler, 0.0f)
				   + sq(_params.range_noise)
				   + sq(_params.range_noise_scaler * _range_sensor.getRange());

	// calculate the innovation variance - limiting it to prevent a badly conditioned fusion
	_hagl_innov_var = fmaxf(P(24,24) - 2*P(9,24) + P(9,9) + obs_variance, obs_variance);

	// perform an innovation consistency check and only fuse data if it passes
	const float gate_size = fmaxf(_params.range_innov_gate, 1.0f);
	_hagl_test_ratio = sq(_hagl_innov) / (sq(gate_size) * _hagl_innov_var);

	bool is_fused = false;
	if (_hagl_test_ratio <= 1.0f) {
		// calculate the Kalman gain
		const float HK0 = 1.0F/_hagl_innov_var;

		// calculate the observation Jacobians and Kalman gains
		SparseVector25f<9,24> Hfusion; // Optical flow observation Jacobians
		Vector25f Kfusion;

		if (_control_status.flags.rng_hgt) {
			Hfusion.at<9>() = -1.0f;
			if (shouldUseRangeFinderForHagl()) {
				for (uint8_t index=0; index<=23; index++) {
					Kfusion(index) = HK0*(P(index,24) - P(index,9));
				}
			} else {
				for (uint8_t index=0; index<=23; index++) {
					Kfusion(index) = - HK0*P(index,9);
				}
			}
		} else {
			for (unsigned row=0; row<=23; row++) {
				// update of all states other than terrain is inhibited
				Kfusion(row) = 0.0f;
			}
		}

		if (shouldUseRangeFinderForHagl()) {
			Hfusion.at<24>() = 1.0f;
			if (_control_status.flags.rng_hgt) {
				Kfusion(24) = HK0*(P(24,24) - P(24,9));
			} else {
				Kfusion(24) = HK0*P(24,24);
			}
		} else {
			Kfusion(24) = 0.0f;
		}


		is_fused = measurementUpdate(Kfusion, Hfusion, _hagl_innov);
	}

	if (is_fused) {
		// record last successful fusion event
		if (shouldUseRangeFinderForHagl()) {
			_time_last_hagl_fuse = _time_last_imu;
			_innov_check_fail_status.flags.reject_hagl = false;
		}
		if (_control_status.flags.rng_hgt) {
			_time_last_hgt_fuse = _time_last_imu;
			_innov_check_fail_status.flags.reject_ver_pos = false;
		}
	} else {
		if (shouldUseRangeFinderForHagl()) {
			// If we have been rejecting range data for too long, reset to measurement
			const uint64_t timeout = static_cast<uint64_t>(_params.terrain_timeout * 1e6f);
			if (isTimedOut(_time_last_hagl_fuse, timeout)) {
				_state.posd_terrain = _state.pos(2) + meas_hagl;
				P.uncorrelateCovarianceSetVariance<1>(24, 0.0f);
				P(24,24) = obs_variance;
				_terrain_vpos_reset_counter++;
			} else {
				_innov_check_fail_status.flags.reject_hagl = true;
			}
		}
		if (_control_status.flags.rng_hgt) {
			_innov_check_fail_status.flags.reject_ver_pos = true;
		}
	}
}
