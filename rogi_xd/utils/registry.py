from typing import Iterable, Iterator, Mapping, Type, Union


class ClassRegistry(Mapping[str, Type]):
    def __init__(self):
        self.__registry = {}

    def register(self, alias: Union[str, Iterable[str], None] = None):
        def decorator(cls):
            if alias is None:
                keys = [cls.__name__.lower()]
            elif isinstance(alias, str):
                keys = [alias]
            else:
                keys = alias

            cls.alias = keys[0]
            for k in keys:
                self.__registry[k] = cls

            return cls

        return decorator

    __call__ = register

    def __getitem__(self, key: str) -> Type:
        return self.__registry[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__registry)

    def __len__(self) -> int:
        return len(self.__registry)
