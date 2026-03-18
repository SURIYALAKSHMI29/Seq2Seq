class Registry:
    def __init__(self):
        self.registry = {}

    def register(self, category, name):
        def wrapper(cls):
            if category not in self.registry:
                self.registry[category] = {}
            if name in self.registry[category]:
                raise ValueError(f"Model {name} already registered")
            self.registry[category][name] = cls
            return cls

        return wrapper

    def build(self, config):
        category = config.category
        model_name = config.model_name

        if category not in self.registry:
            raise ValueError(f"No such module {category} found in registry")
        if model_name not in self.registry[category]:
            raise ValueError(f"Model {model_name} not found in registry")
        return self.registry[category][model_name](config)


REGISTRY = Registry()
