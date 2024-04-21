from constraints import Constraint,Constraints

class my_test_config():
	def __init__(self):
		self.AREA_THRESHOLD = 411
		self.THRESHOLD_RATIO = 2
		AREA =Constraint(name = "AREA", threshold = self.AREA_THRESHOLD, threshold_ratio = self.THRESHOLD_RATIO)
		self.constraints = Constraints()
		self.constraints.append(AREA)


	def config_check(self):
		print(f"###### Config Check ######")
		print(f"configtype:test")
		print(f"AERA:{self.AREA_THRESHOLD}")
