# WINGTRA: this msg file has been modified to fit our PP needs, so most things will have comments related to topic metadata

uint64 timestamp	# time since system start (microseconds)
float32 alt_max 	# DESCRIPTION: "maximum altitude in [m] that can be reached"
bool freefall	# DESCRIPTION: "true if vehicle is currently in free-fall"
bool ground_contact # DESCRIPTION: "true if vehicle has ground contact but is not landed"
bool maybe_landed # DESCRIPTION: "true if the vehicle might have landed"
bool landed		# DESCRIPTION: "true if vehicle is currently landed on the ground"
bool in_ground_effect # DESCRIPTION: " indicates if from the perspective of the landing detector the vehicle might be in ground effect (baro). This flag will become true if the vehicle is not moving horizontally and is descending (crude assumption that user is landing)."

bool land_acc_peak	# DESCRIPTION: "True if landing accel peak observed"
bool upstream_landed	# DESCRIPTION: "True if landing observed upstream style"
bool crash 		# DESCRIPTION: "True if crash observed"

# NAMES Land detector
