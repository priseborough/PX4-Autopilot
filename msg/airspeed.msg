# WINGTRA: this msg file has been modified to fit our PP needs, so most things will have comments related to topic metadata

uint64 timestamp			# time since system start (microseconds)

float32 indicated_airspeed_m_s		# NAME: "Indicated airspeed", DESCRIPTION: "meters per second, -1 if unknown"

float32 true_airspeed_m_s		# NAME: "True filtered airspeed", DESCRIPTION: "meters per second, -1 if unknown"

float32 air_temperature_celsius		# NAME: "Air temperature", DESCRIPTION: "degrees celsius, -1000 if unknown"

float32 confidence			# NAME: "Sensor confidence value", DESCRIPTION: "from 0 to 1 for this sensor"

float32 indicated_airspeed_lpf_m_s	# WINGTRA: NAME: "Lowpass filtered airspeed", DESCRIPTION: "meters per second, -1 if unknown"
float32 diff_pressure_filtered_c_pa # WINGTRA: NAME: "Lowpass filtered differential pressure with temp offset corrected", DESCRIPTION: "Pa"
float32 diff_pressure_raw_c_pa # WINGTRA: NAME: "Raw differential pressure with temp offset corrected", DESCRIPTION: "Pa"

# NAMES Airspeed sensor
