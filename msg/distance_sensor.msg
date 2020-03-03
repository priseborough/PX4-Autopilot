# WINGTRA: this msg file has been modified to fit our PP needs, so most things will have comments related to topic metadata

# DISTANCE_SENSOR message data

uint64 timestamp		# time since system start (microseconds)

float32 min_distance		# NAME: "Minimum distance", DESCRIPTION: "Minimum distance the sensor can measure (in m)"
float32 max_distance		# NAME: "Maximum distance", DESCRIPTION: "Maximum distance the sensor can measure (in m)"
float32 current_distance	# NAME: "Current distance reading", DESCRIPTION: "(in m)"
# float32 variance		# WINGTRA: Not used; Measurement variance (in m^2), 0 for unknown / invalid readings
int8 signal_quality		# NAME: "Signal quality", DESCRIPTION: Signal quality in percent (0...100%), where 0 = invalid signal, 100 = perfect signal, and -1 = unknown signal quality.

uint8 type			# NAME: "Type", DESCRIPTION: "from MAV_DISTANCE_SENSOR enum"
uint8 MAV_DISTANCE_SENSOR_LASER = 0
uint8 MAV_DISTANCE_SENSOR_ULTRASOUND = 1
uint8 MAV_DISTANCE_SENSOR_INFRARED = 2
uint8 MAV_DISTANCE_SENSOR_RADAR = 3

uint8 id			# NAME: "ID", DESCRIPTION: "Onboard ID of the sensor"

# float32 h_fov # WINGTRA: Not used; Sensor horizontal field of view (rad)
# float32 v_fov # WINGTRA: Not used; Sensor vertical field of view (rad)
# float32[4] q # WINGTRA: Not used; Quaterion sensor orientation with respect to the vehicle body frame to specify the orientation ROTATION_CUSTOM

uint8 orientation		# NAME: "Orientation", DESCRIPTION: "Direction the sensor faces from MAV_SENSOR_ORIENTATION enum"

uint8 ROTATION_YAW_0		= 0 # MAV_SENSOR_ROTATION_NONE
uint8 ROTATION_YAW_45		= 1 # MAV_SENSOR_ROTATION_YAW_45
uint8 ROTATION_YAW_90		= 2 # MAV_SENSOR_ROTATION_YAW_90
uint8 ROTATION_YAW_135		= 3 # MAV_SENSOR_ROTATION_YAW_135
uint8 ROTATION_YAW_180		= 4 # MAV_SENSOR_ROTATION_YAW_180
uint8 ROTATION_YAW_225		= 5 # MAV_SENSOR_ROTATION_YAW_225
uint8 ROTATION_YAW_270		= 6 # MAV_SENSOR_ROTATION_YAW_270
uint8 ROTATION_YAW_315		= 7 # MAV_SENSOR_ROTATION_YAW_315

uint8 ROTATION_FORWARD_FACING	= 0 # MAV_SENSOR_ROTATION_NONE
uint8 ROTATION_RIGHT_FACING	= 2 # MAV_SENSOR_ROTATION_YAW_90
uint8 ROTATION_BACKWARD_FACING	= 4 # MAV_SENSOR_ROTATION_YAW_180
uint8 ROTATION_LEFT_FACING	= 6 # MAV_SENSOR_ROTATION_YAW_270

uint8 ROTATION_UPWARD_FACING   = 24 # MAV_SENSOR_ROTATION_PITCH_90
uint8 ROTATION_DOWNWARD_FACING = 25 # MAV_SENSOR_ROTATION_PITCH_270

uint8 ROTATION_CUSTOM          = 100 # MAV_SENSOR_ROTATION_CUSTOM

# WINGTRA fields:
float32 signal_quality_estimated		# WINGTRA: NAME: "Estimated signal quality", DESCRIPTION: "Estimated noise level"

bool stuck				# WINGTRA: NAME: "Distance sensor stuck flag"
float32 stuck_distance	# WINGTRA: NAME: "Stuck distance value", DESCRIPTION: "Distance (in m) at which the sensor is stuck (while stuck is true)"

uint16 comms_errors		# WINGTRA: DESCRIPTION: "Log comm error"
uint16 wait_bailout		# WINGTRA: DESCRIPTION: "Log wait bailout"
uint8 status_register		# WINGTRA: NAME: "Status register"


# WINGTRA: Generate topic of the same structure
# TOPICS distance_sensor distance_sensor_raw
# NAMES Distance sensor filtered information | Distance sensor raw information
