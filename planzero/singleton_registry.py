class SingletonRegistry(object):
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
        self.classes[cls.__name__] = cls

    def __getitem__(self, key):
        try:
            return self.instances[key]
        except KeyError:
            self.instances[key] = self.classes[key]()
            return self.instances[key]

    def items(self):
        for key in self.classes:
            yield (key, self[key])
