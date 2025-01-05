from config.constraints import Constraint, Constraints


class Config1:
    def __init__(self):
        AREA = Constraint(name="area", threshold=411, threshold_ratio=3)
        POWER = Constraint(name="power", threshold=135, threshold_ratio=3)
        self.constraints = Constraints()
        self.constraints.append(AREA)
        self.constraints.append(POWER)

    def config_check(self):
        print(f"###### Config Check ######")
        self.constraints.print()



class Config2:
    def __init__(self):
        AREA = Constraint(name="area", threshold=165, threshold_ratio=3)
        POWER = Constraint(name="power", threshold=16, threshold_ratio=3)
        self.constraints = Constraints()
        self.constraints.append(AREA)
        self.constraints.append(POWER)

    def config_check(self):
        print(f"###### Config Check ######")
        self.constraints.print()