uint8 fix_type			# 0-1: no fix, 2: 2D fix, 3: 3D fix, 4: RTCM code differential, 5: Real-Time Kinematic, float, 6: Real-Time Kinematic, fixed, 8: Extrapolated. Some applications will not use the value of this field unless it is at least two, so always correctly fill in the fix.
uint64 time_utc_usec		# Timestamp (microseconds, UTC), this is the timestamp which comes from the gps module.
int32 lat			# Latitude in 1E-7 degrees
int32 lon			# Longitude in 1E-7 degrees
int32 alt			# Altitude in 1E-3 meters above MSL, (millimetres)
float32 ev			# GPS speed accuracy estimate, (metres/sec)
float32 eph			# GPS horizontal position accuracy (metres)
float32 epv			# GPS vertical position accuracy (metres)
float32 vx			# North velocity in NED earth-fixed frame, (metres/sec)
float32 vy			# East velocity in NED earth-fixed frame, (metres/sec)
float32 vz			# Down velocity in NED earth-fixed frame, (metres/sec)
bool lpos_valid			# true if local position is valid/origin is set
float32 x			# North position in NED earth-fixed frame, (metres)
float32 y			# East position in NED earth-fixed frame, (metres)
float32 z			# Down position (negative altitude) in NED earth-fixed frame, (metres)
