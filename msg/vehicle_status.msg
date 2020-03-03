# WINGTRA: this msg file has been modified to fit our PP needs, so most things will have comments related to topic metadata

# If you change the order, add or remove arming_state_t states make sure to update the arrays
# in state_machine_helper.cpp as well.
uint64 timestamp				# time since system start (microseconds)

uint8 ARMING_STATE_INIT = 0
uint8 ARMING_STATE_STANDBY = 1
uint8 ARMING_STATE_ARMED = 2
uint8 ARMING_STATE_STANDBY_ERROR = 3
uint8 ARMING_STATE_SHUTDOWN = 4
uint8 ARMING_STATE_IN_AIR_RESTORE = 5
uint8 ARMING_STATE_MAX = 6

# FailureDetector status
uint8 FAILURE_NONE = 0
uint8 FAILURE_ROLL = 1 	# (1 << 0)
uint8 FAILURE_PITCH = 2	# (1 << 1)
uint8 FAILURE_ALT = 4 	# (1 << 2)

# HIL
uint8 HIL_STATE_OFF = 0
uint8 HIL_STATE_ON = 1

# Navigation state, i.e. "what should vehicle do".
uint8 NAVIGATION_STATE_MANUAL = 0		# Manual mode
uint8 NAVIGATION_STATE_ALTCTL = 1		# Altitude control mode
uint8 NAVIGATION_STATE_POSCTL = 2		# Position control mode
uint8 NAVIGATION_STATE_AUTO_MISSION = 3		# Auto mission mode
uint8 NAVIGATION_STATE_AUTO_LOITER = 4		# Auto loiter mode
uint8 NAVIGATION_STATE_AUTO_RTL = 5		# Auto return to launch mode
uint8 NAVIGATION_STATE_AUTO_RCRECOVER = 6	# RC recover mode
uint8 NAVIGATION_STATE_AUTO_RTGS = 7		# Auto return to groundstation on data link loss
uint8 NAVIGATION_STATE_AUTO_LANDENGFAIL = 8 	# Auto land on engine failure
uint8 NAVIGATION_STATE_AUTO_LANDGPSFAIL = 9	# Auto land on gps failure (e.g. open loop loiter down)
uint8 NAVIGATION_STATE_ACRO = 10		# Acro mode
uint8 NAVIGATION_STATE_UNUSED = 11		# Free slot
uint8 NAVIGATION_STATE_DESCEND = 12		# Descend mode (no position control)
uint8 NAVIGATION_STATE_TERMINATION = 13		# Termination mode
uint8 NAVIGATION_STATE_OFFBOARD = 14
uint8 NAVIGATION_STATE_STAB = 15		# Stabilized mode
uint8 NAVIGATION_STATE_RATTITUDE = 16		# Rattitude (aka "flip") mode
uint8 NAVIGATION_STATE_AUTO_TAKEOFF = 17	# Takeoff
uint8 NAVIGATION_STATE_AUTO_LAND = 18		# Land
uint8 NAVIGATION_STATE_AUTO_FOLLOW_TARGET = 19	# Auto Follow
uint8 NAVIGATION_STATE_AUTO_PRECLAND = 20	# Precision land with landing target
uint8 NAVIGATION_STATE_ORBIT = 21       # Orbit in a circle
uint8 NAVIGATION_STATE_MAX = 22

uint8 RC_IN_MODE_DEFAULT = 0
uint8 RC_IN_MODE_OFF = 1
uint8 RC_IN_MODE_GENERATED = 2

uint8 VEHICLE_TYPE_UNKNOWN = 0
uint8 VEHICLE_TYPE_ROTARY_WING = 1
uint8 VEHICLE_TYPE_FIXED_WING = 2
uint8 VEHICLE_TYPE_ROVER = 3

# state machine / state of vehicle.
# Encodes the complete system state and is set by the commander app.

uint8 nav_state				# set navigation state machine to specified value; NAME: "Navigation state", DESCRIPTION: "0: Manual mode<br/>1: Altitude control mode<br/>2: Position control mode (Assissted)<br/>3: Auto mission mode (Mission)<br/>4: Auto loiter mode<br/>5: Auto return to launch mode (RTH)<br/>6: RC recover mode (RTH)<br/>7: Auto return to groundstation on data link loss (RTH)<br/>8: Auto land on engine failure<br/>9: Auto land on gps failure (e.g. open loop loiter down)<br/>10: Acro mode<br/>12: Descend mode (no position control)<br/>13: Termination mode<br/>15: Stabilized mode<br/>16: Rattitude (aka "flip") mode<br/>17:Takeoff<br/>18: Land"
uint8 arming_state			# NAME: "Current arming state", DESCRIPTION: "ARMING_STATE_INIT = 0<br/>ARMING_STATE_STANDBY = 1<br/>ARMING_STATE_ARMED = 2<br/>ARMING_STATE_ARMED_ERROR = 3<br/>ARMING_STATE_STANDBY_ERROR = 4<br/>ARMING_STATE_REBOOT = 5<br/>ARMING_STATE_IN_AIR_RESTORE = 6<br/>ARMING_STATE_MAX = 7"
uint8 hil_state				# NAME: "Current hil state", DESCRIPTION: "HIL_STATE_OFF = 0<br/>HIL_STATE_ON = 1"
bool failsafe				# NAME: "Failsafe", DESCRIPTION: "true if system is in failsafe state"

uint8 system_type			# NAME: "System type", DESCRIPTION: "contains mavlink MAV_TYPE"
uint8 system_id			# NAME: "System ID", DESCRIPTION: "contains MAVLink's system ID field"
uint8 component_id			# NAME: "Component ID", DESCRIPTION: "subsystem, contains MAVLink's component ID field"

uint8 vehicle_type          # DESCRIPTION: "Type of vehicle (fixed-wing, rotary wing, ground)<br/>If the vehicle is a VTOL, then this value will be VEHICLE_TYPE_ROTARY_WING while flying as a multicopter,<br/>and VEHICLE_TYPE_FIXED_WING when flying as a fixed-wing

bool is_vtol				# DESCRIPTION: "True if the system is VTOL capable"
# bool vtol_fw_permanent_stab		# WINGTRA: DESCRIPTION: "True if VTOL should stabilize attitude for fw in manual mode"
bool in_transition_mode			# DESCRIPTION: "True if VTOL is doing a transition"
bool in_transition_to_fw		# DESCRIPTION: "True if VTOL is doing a transition from MC to FW"

bool rc_signal_lost				# DESCRIPTION: "true if RC reception lost"
uint8 rc_input_mode				# DESCRIPTION: "set to 1 to disable the RC input, 2 to enable manual control to RC in mapping."

bool data_link_lost				# DESCRIPTION: "datalink to GCS lost"
uint8 data_link_lost_counter			# DESCRIPTION: "counts unique data link lost events"

# bool high_latency_data_link_lost 			# WINGTRA: Not used; Set to true if the high latency data link (eg. RockBlock Iridium 9603 telemetry module) is lost

# bool engine_failure				# WINGTRA: Not used; Set to true if an engine failure is detected
bool mission_failure				# DESCRIPTION: "Set to true if mission could not continue/finish"

# uint8 failure_detector_status			# WINGTRA: Not used; Bitmask containing FailureDetector status [0, 0, 0, 0, 0, FAILURE_ALT, FAILURE_PITCH, FAILURE_ROLL]

# see SYS_STATUS mavlink message for the following
uint32 onboard_control_sensors_present  # DESCRIPTION: "see SYS_STATUS mavlink message for the following"
uint32 onboard_control_sensors_enabled  # DESCRIPTION: "see SYS_STATUS mavlink message for the following"
uint32 onboard_control_sensors_health  # DESCRIPTION: "see SYS_STATUS mavlink message for the following"

# WINGTRA: Not used
# airspeed fault and airspeed use status
# float32 arspd_check_level	# integrated airspeed inconsistency as checked against the COM_TAS_FS_INTEG parameter
# bool aspd_check_failing		# true when airspeed consistency checks are failing
# bool aspd_fault_declared	# true when an airspeed fault has been declared
# bool aspd_use_inhibit		# true if switching to a non-airspeed control mode has been requested
# bool aspd_fail_rtl		# true if airspeed failure invoked RTL has been requested
# float32 load_factor_ratio	# ratio of measured to aerodynamic load factor limit. Greater than 1 indicates airspeed low error condition.
# WINGTRA: End


# WINGTRA: Emergency switch action modes
uint8 wingtra_sos				# NAME: "Emergency switch action mode flag", DESCRIPTION: "1: No PWM mode failsafe<br/>2: Pure manual mode"
bool no_pwm 					# NAME: "No PWM flag"
bool pure_manual				# NAME: "Pure manual mode flag"
bool custom_mode				# NAME: "Custom mode flag"
uint8 WINGTRA_SOS_STATE_NO_PWM = 1		# No PWM mode failsafe
uint8 WINGTRA_SOS_STATE_MANUAL = 2		# Pure manual mode
uint8 WINGTRA_SOS_STATE_CUSTOM = 3		# Custom mode for dev flights

bool user_mode					# NAME: "User mode", DESCRIPTION: "Set to true if user mode is enabled"

# WINGTRA: User safety
bool arming_attitude			        # DESCRIPTION: "User safety: True if attitude upon arming is within set boundaries (pitch+roll < 10°)"

bool rtl_climb_boost 				# WINGTRA

uint8 rth_type  # Set to value specified by RTH_INFO_TYPE enum
bool assisted_flight  # True if aircraft flown in assisted mode
bool use_virtual_rc  # True if user decides to use virtual RC
bool physical_rc_active  # True if physical RC is turned on
bool rth_forbidden_resume  # True if RTH reason forbids resume

# WINGTRA: Battery RTH
bool low_battery_rth          			 # DESCRIPTION: "Battery: True if battery RTH was triggered"

# NAMES Vehicle status
# DESCRIPTIONS State of vehicle. Encodes the complete system state and is set by the commander app.
