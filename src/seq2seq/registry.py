class Registry:
    def __init__(self):
        self.registry = {}

    def register(self, name):
        def wrapper(cls):
            if name in self.registry:
                raise ValueError(f"Model {name} already registered")
            self.registry[name] = cls
            return cls

        return wrapper

    def build(self, config):
        type = config.type
        if type not in self.registry:
            raise ValueError(f"Model {type} not found in registry")
        return self.registry[type](config)


ENCODERS = Registry()
DECODERS = Registry()
