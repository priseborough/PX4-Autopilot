uint64 timestamp	# time since system start (microseconds)
float32[3] del_ang	# delta angle measured by IMU aligned with body frame(rad)
float32 del_ang_dt	# delta angle integration time (s)
float32[3] del_vel	# delta velocity measured by IMU aligned with  body frame (m/s)
float32 del_vel_dt	# delta velocity integration time (s)
float32[3] vel		# North East Down velocity measurent (m/s)
float32 vel_err		# 1-sigma velocity measurement uncertainty (m/s)
bool fuse_vel		# True when velocity measurement has been updated
float32[4] quat		# quaternion estimate from onboard estimator.
float32 yaw_composite	# composite yaw estimate
float32[8] yaw		# yaw estimate for each model in the filter bank (rad)
float32[8] innov_vn	# yaw estimate for each model in the filter bank (m/s)
float32[8] innov_ve	# yaw estimate for each model in the filter bank (m/s)
float32[8] weight	# yaw estimate for each model in the filter bank
