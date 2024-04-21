from math import prod

class Constraint():
    def __init__(self,name,threshold,threshold_ratio):
        assert threshold > 0
        self.name =name
        self.threshold = threshold
        self.threshold_ratio = threshold_ratio

        self.value = 0
        self.is_meet_flag = False 

    @property
    def margin(self):
        return (self.threshold-self.value)/self.threshold
    
    @property
    def punishment(self):
        return (self.value / self.threshold) ** (self.threshold_ratio if self.is_meet_flag else 0)

    def update(self, value):
        self.value = value
        self.is_meet_flag = self.margin >= 0

    def __str__(self):
        return (f"Constraint(name={self.name}, threshold={self.threshold}, value={self.value}, "
                f"margin={self.margin:.2f}, is_meet={self.is_meet_flag}, punishment={self.punishment:.2f})")
    

class Constraints():
    def __init__(self):
        self.constraints = []

    def append(self, constranit:Constraint):
        self.constraints.append(constranit)

    def update(self,values):
        if len(values) != len(self.constraints):
            raise ValueError("values count must match constraint count.")
        
        for key,value,constraint in zip(values.keys(), values.values(), self.constraints):
            if key!=constraint.name:
                raise ValueError(f"Key '{key}' does not match constraint name '{constraint.name}'.")
            constraint.update(value)

    def get_margin(self, name):
        for constraint in self.constraints:
            if constraint.name == name:
                return constraint.margin
        return 0

    def is_any_meet(self):
        return any(constraint.is_meet_flag for constraint in self.constraints)

    def is_all_meet(self):
        return all(constraint.is_meet_flag for constraint in self.constraints)

    def get_punishment(self):
        return prod(constraint.punishment for constraint in self.constraints)

    def print(self):
        for constraint in self.constraints:
            print(constraint)
        


if __name__=='__main__':
    # 使用示例
    c = Constraint("Speed Limit", 100, 2)
    c.update(80)
    print(c)  # 输出更加友好和清晰



