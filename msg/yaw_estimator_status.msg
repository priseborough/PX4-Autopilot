uint64 timestamp		# time since system start (microseconds)
uint64 timestamp_sample         # the timestamp of the raw data (microseconds)

float32 yaw_composite	# composite yaw from GSF (rad)
float32 yaw_variance	# composite yaw variance from GSF (rad^2)

float32[5] yaw			# yaw estimate for each model in the filter bank (rad)
float32[5] innov_mps		# velocity innovation vector length for each model in the filter bank scaled in m/s
float32[5] innov_std_dev	# velocity innovation vector length for each model in the filter bank scaled in standard deviations
float32[5] weight		# weighting for each model in the filter bank
