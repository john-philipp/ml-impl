class _IEnum:
    @classmethod
    def choices(cls):
        return [y for x, y in cls.__dict__.items() if not x.startswith("_") and not isinstance(y, classmethod)]


