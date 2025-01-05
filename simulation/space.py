import logging

logger = logging.getLogger(f"logger")


class ActionDimension:
    def __init__(self, name, default_value, step, rrange, frozen=False, model={"name": "normal", "param": 0.5}):
        self.name = name  # "core"
        self.step = step  # 1

        self.frozen = frozen
        self.model = model

        assert rrange[0] < rrange[-1], "range error"
        self.rrange = rrange  # [1,16]
        self.scale = int((rrange[-1] - rrange[0]) / step + 1)  # 16

        self.default_value = default_value
        self.value = default_value

        self.default_index = int((default_value - rrange[0]) / step)
        self.index = self.default_index

        # 保留一位小数
        self.sample_box = [
            float(f"{self.rrange[0] + idx * step:.1f}") for idx in range(int(self.scale))
        ]

        logger.info(
            f"name : {self.name} , default_value : {self.default_value} , step : {self.step} , rrange : {self.rrange}")
        logger.info(f"sample_box : {self.sample_box}")

    def sample(self, sample_index):
        self.index = sample_index % self.scale
        self.value = self.sample_box[self.index]
        return self.value

    def reset(self):
        self.index = self.default_index
        self.value = self.default_value

    def froze(self):
        self.frozen = True

    def release(self):
        self.frozen = False


class DesignSpace:
    def __init__(self):
        self.dimension_box = []
        self.scale = 1
        self.len = 0

    def append(self, action_dimension: ActionDimension):
        self.dimension_box.append(action_dimension)
        self.scale *= action_dimension.scale
        self.len += 1

    @property
    def states(self):
        states = dict()
        for item in self.dimension_box:
            states[item.name] = item.value
        return states

    def states_reset(self):
        for item in self.dimension_box:
            item.reset()

    def sample_one_dimension(self, dimension_index, sample_index):
        assert 0 <= dimension_index <= self.len - 1, "dimension_index error"
        self.dimension_box[dimension_index].sample(sample_index)
        return self.states


def create_space() -> DesignSpace:
    space = DesignSpace()

    core = ActionDimension("core", 1, 1, [1, 16])
    l1i_size = ActionDimension("l1i_size", 2, 1, [1, 12])
    l1d_size = ActionDimension("l1d_size", 2, 1, [1, 12])
    l2_size = ActionDimension("l2_size", 6, 1, [6, 16])
    l1d_assoc = ActionDimension("l1d_assoc", 1, 1, [1, 4])
    l1i_assoc = ActionDimension("l1i_assoc", 1, 1, [1, 4])
    l2_assoc = ActionDimension("l2_assoc", 1, 1, [1, 4])
    sys_clock = ActionDimension("sys_clock", 2, 0.1, [2, 4])

    dimensions = [
        core,
        l1i_size,
        l1d_size,
        l2_size,
        l1d_assoc,
        l1i_assoc,
        l2_assoc,
        sys_clock,
    ]

    for dimension in dimensions:
        space.append(dimension)

    return space
