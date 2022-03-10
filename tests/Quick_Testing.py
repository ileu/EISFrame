class iter_test:
    def __init__(self, **kwargs):
        self.lesen = ["a", "b", "c", "±"]
        self.kwargs = {"current": -1}
        for kwarg in kwargs:
            self.kwargs[kwarg] = kwargs[kwarg]

    def read(self):
        if self.kwargs["current"] < 0:
            return self.lesen
        return self.lesen[self.kwargs["current"]]

    def __iter__(self):
        print("HALLO")
        return self

    def __next__(self):
        self.kwargs["current"] += 1
        if self.kwargs["current"] < len(self.lesen):
            return self
        self.kwargs["current"] = -1
        raise StopIteration

    @classmethod
    def get_cycle(cls, cycle):
        child = cls(current=cycle)
        return child

    def __getitem__(self, args):
        if isinstance(args, int):
            return self.get_cycle(args)
        raise ValueError


def test_circuit_measure():
    stopper = 0
    testing = iter_test()

    assert testing.read() == ["a", "b", "c", "±"]
    for it in testing:
        stopper += 1
        if stopper > 15:
            break
        print(it.read())
    print(testing.kwargs["current"])
    print(testing[2].read() == "c")
    print(testing.kwargs["current"])
    print(testing.read() == ["a", "b", "c", "±"])
    print(testing.kwargs["current"])
    print(testing.get_cycle(1).read() == "b")
    print(testing.kwargs["current"])
