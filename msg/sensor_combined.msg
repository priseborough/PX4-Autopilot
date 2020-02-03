# WINGTRA: this msg file has been modified to fit our PP needs, so most things will have comments related to topic metadata

# Sensor readings in SI-unit form.
# These fields are scaled and offset-compensated where possible and do not
# change with board revisions and sensor updates.

uint64 timestamp				# time since system start (microseconds)
uint64 hrt_absolute_time		# WINGTRA

int32 RELATIVE_TIMESTAMP_INVALID = 2147483647	# (0x7fffffff) If one of the relative timestamps is set to this value, it means the associated sensor values are invalid

# gyro timstamp is equal to the timestamp of the message
float32[3] gyro_rad			# DESCRIPTION: "average angular rate measured in the XYZ body frame in rad/s over the last gyro sampling period"
uint32 gyro_integral_dt		# DESCRIPTION: "gyro measurement sampling period in us"

int32 accelerometer_timestamp_relative	# NAME: "Accelerometer timestamp", DESCRIPTION: "timestamp + accelerometer_timestamp_relative = Accelerometer timestamp"
float32[3] accelerometer_m_s2		# DESCRIPTION: "average value acceleration measured in the XYZ body frame in m/s/s over the last accelerometer sampling period"
uint32 accelerometer_integral_dt	# DESCRIPTION: "accelerometer measurement sampling period in us"

uint32 times_published			# WINGTRA: DESCRIPTION: "number of times the topic have been published"
uint64 integrated_time_us		# WINGTRA: DESCRIPTION: "Integrated gyro measurement sampling period in us"

uint64 error_count_accel  # WINGTRA
uint64 error_count_gyro  # WINGTRA

# NAMES Comnined sensor readings
# DESCRIPTIONS Sensor readings in SI-unit form. (Gyro, accelerometer, magnetometer, barometer)
