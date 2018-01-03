// -*- tab-width: 4; Mode: C++; c-basic-offset: 4; indent-tabs-mode: nil -*-

/****************************************************************************
 *
 *   Copyright (C) 2012 PX4 Development Team. All rights reserved.
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

/// @file	LowPassFilter.cpp
/// @brief	A class to implement a second order low pass filter
/// Author: Leonard Hall <LeonardTHall@gmail.com>

#include <px4_defines.h>
#include "LowPassFilter2p.hpp"
#include <cmath>

namespace math
{

void LowPassFilter2p::set_cutoff_frequency(float sample_freq, float cutoff_freq)
{
	_cutoff_freq = cutoff_freq;

	if (_cutoff_freq <= 0.0f) {
		// no filtering
		return;
	}

	/*
	Equivalent Laplace transfer function:

				  1
	G(s) = --------------------------------------
	       (1/omega^2)*s^2 + (2*zeta/omega)*s + 1

	omega = 2 * PI * cutoff_freq  (rad/sec)
	zeta = viscous damping ratio = 1/Q

	A damping ratio of 1/sqrt(2) achieves a flat pass-band without peaking and minimises phase loss

	Coefficients are calculated for an equivalent bi-quad IIR filter using a bilinear transform.
	Cutoff frequency is normalised for a unity time step and prewarped before applying the transformation.

	Z domain transfer function given by:

						     omega_n^2*(z + 1)^2
	G(z) = -------------------------------------------------------------------------------------------------
	       omega_n^2*z^2 + 2*omega_n^2*z + omega_n^2 + 4*zeta*omega_n*z^2 - 4*zeta*omega_n + 4*z^2 - 8*z + 4
	*/

	// normalise frequency for unit time step and apply pre-warping
	const float omega_n = 2.0f * tanf(M_PI_F * _cutoff_freq / sample_freq);

	// set optimum damping ratio
	const float zeta = 0.7071068f;

	// calculate intermediate terms
	const float omega_n_sq = omega_n * omega_n;
	const float temp = 4.0f * zeta * omega_n;
	const float a0_inv = 1.0f / (omega_n_sq + temp + 4.0f);

	// calculate difference equation coefficients assuming a0 = 1
	_a1 = (2.0f * omega_n_sq - 8.0f) * a0_inv;
	_a2 = (omega_n_sq - temp + 4.0f) * a0_inv;
	_b0 = omega_n_sq * a0_inv;
	_b1 = 2.0f * _b0;
	_b2 = _b0;
}

float LowPassFilter2p::apply(float sample)
{
	if (_cutoff_freq <= 0.0f) {
		// no filtering
		return sample;
	}

	// do the filtering
	float delay_element_0 = sample - _delay_element_1 * _a1 - _delay_element_2 * _a2;

	if (!PX4_ISFINITE(delay_element_0)) {
		// don't allow bad values to propagate via the filter
		delay_element_0 = sample;
	}

	float output = delay_element_0 * _b0 + _delay_element_1 * _b1 + _delay_element_2 * _b2;

	_delay_element_2 = _delay_element_1;
	_delay_element_1 = delay_element_0;

	// return the value.  Should be no need to check limits
	return output;
}

float LowPassFilter2p::reset(float sample)
{
	float dval = sample / (_b0 + _b1 + _b2);
	_delay_element_1 = dval;
	_delay_element_2 = dval;
	return apply(sample);
}

} // namespace math

