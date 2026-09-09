class SingletonRegistry:
    """
    Register classes at class-definition time,
    and create singletons at element-access time.

    This is helpful because classes should be instantiated only after their
    class definitions are complete.
    """

    def __init__(self):
        self.classes = {}
        self.instances = {}

    def add_class(self, cls):
        assert cls.__name__ not in self.classes
        self.classes[cls.__name__] = cls

    def __getitem__(self, key):
        try:
            return self.instances[key]
        except KeyError:
            self.instances[key] = self.classes[key]()
            return self.instances[key]

    def __setitem__(self, key, val):
        assert key not in self.classes
        if key in self.instances:
            raise NotImplementedError()
        self.instances[key] = val

    def items(self):
        keys = set(self.classes)
        keys.update(self.instances)
        for key in keys:
            yield (key, self[key])
