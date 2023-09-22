import os
import threading
import typing
from abc import ABC
from abc import abstractmethod

from mrc.core import operators as ops
from nemollm.api import NemoLLM

# Another option from: https://stackoverflow.com/a/73495782
# lock = threading.Lock()
# class Singleton(type):
#     _instances = {}
#     _init = {}

#     def __init__(cls, name, bases, dct):
#         cls._init[cls] = dct.get('__init__', None)

#     def __call__(cls, *args, **kwargs):
#         init = cls._init[cls]
#         if init is not None:
#             args_list = list(args)
#             for idx, arg in enumerate(args_list):
#                 args_list[idx] = str(arg)
#             tmp_kwargs = {}
#             for arg_key, arg_value in kwargs.items():
#                 tmp_kwargs[arg_key] = str(arg_value)
#             key = (cls, frozenset(inspect.getcallargs(init, None, *args_list, **tmp_kwargs).items()))
#         else:
#             key = cls
#         if key not in cls._instances:
#             with lock:
#                 cls._instances[key] = super(SingletonArgs, cls).__call__(*args, **kwargs)
#         return cls._instances[key]


class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_locks: dict[typing.Any, threading.Lock] = {}

    def __call__(cls, *args, **kwargs):
        # double-checked locking pattern (https://en.wikipedia.org/wiki/Double-checked_locking)
        if cls not in cls._instances:
            if cls not in cls._singleton_locks:
                cls._singleton_locks[cls] = threading.Lock()
            with cls._singleton_locks[cls]:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LLMClient(ABC):

    @typing.overload
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    @typing.overload
    @abstractmethod
    def generate(self, prompt: list[str]) -> list[str]:
        ...

    @abstractmethod
    def generate(self, prompt: str | list[str]) -> str | list[str]:
        pass


class LLMService(ABC):

    @abstractmethod
    def get_client(self, model_name: str, model_kwargs: dict | None = None) -> LLMClient:
        pass


class NeMoService(metaclass=ThreadSafeSingleton):

    # __instance: "NeMoService" = None

    @staticmethod
    def instance(*, api_key: str | None = None, org_id: str | None = None):

        api_key = api_key if api_key is not None else os.environ.get("NGC_API_KEY", None)
        org_id = org_id if org_id is not None else os.environ.get("NGC_ORG_ID", None)

        return NeMoService(api_key=api_key, org_id=org_id)

    # def __new__(cls, *, api_key: str = None, org_id: str = None, **kwargs):
    #     instance = cls.__dict__.get("__instance", None)

    #     if instance is not None:
    #         # Verify the api key and org id are the same
    #         if (instance._api_key != api_key):
    #             raise RuntimeError("The api key has changed")
    #         if (instance._org_id != org_id):
    #             raise RuntimeError("The org id has changed")

    #         return instance

    #     instance = object.__new__(cls)

    #     setattr(cls, "__instance", instance)

    #     # instance.__init__(api_key=api_key, org_id=org_id, **kwargs)

    #     return instance

    # def __new__(cls, *, api_key: str = None, org_id: str):
    #     if not hasattr(cls, "__instance"):
    #         cls.__instance = super(NeMoService, cls).__new__(cls)

    #         cls.__instance._api_key = api_key if api_key is not None else os.environ.get("NGC_API_KEY", None)
    #         cls.__instance._org_id = org_id
    #     else:
    #         # Verify the api key and org id are the same
    #         if (cls.__instance._api_key != api_key):
    #             raise RuntimeError("The api key has changed")
    #         if (cls.__instance._org_id != org_id):
    #             raise RuntimeError("The org id has changed")

    #     return cls.__instance

    def __init__(self, *, api_key: str, org_id: str) -> None:

        self._api_key = api_key
        self._org_id = org_id

        # Do checking on api key

        # Class variables
        self._conn: NemoLLM = NemoLLM(
            # The client must configure the authentication and authorization parameters
            # in accordance with the API server security policy.
            # Configure Bearer authorization
            api_key=self._api_key,

            # If you are in more than one LLM-enabled organization, you must
            # specify your org ID in the form of a header. This is optional
            # if you are only in one LLM-enabled org.
            org_id=self._org_id,
        )

    def start(self):
        pass

    def stop(self):
        pass

    def get_client(self, model_name: str, customization_id: str | None = None, infer_kwargs: dict | None = None):

        return NeMoClient(self, model_name=model_name, customization_id=customization_id, infer_kwargs=infer_kwargs)


class NeMoClient:

    def __init__(self,
                 parent: NeMoService,
                 *,
                 model_name: str,
                 customization_id: str | None = None,
                 infer_kwargs: dict | None = None) -> None:
        self._parent = parent
        self._model_name = model_name
        self._customization_id = customization_id
        self._infer_kwargs = infer_kwargs if infer_kwargs is not None else {}

    async def generate(self, prompt: str | list[str]):

        convert_to_string = False

        # Make sure everything is a list
        if (isinstance(prompt, str)):
            prompt = [prompt]
            convert_to_string = True

        result: list[str] = self._parent._conn.generate_multiple(
            model=self._model_name,
            prompts=prompt,
            customization_id=self._customization_id,  # type: ignore
            return_type="text",
            **self._infer_kwargs,
        )  # type: ignore

        if (convert_to_string):
            return result[0]

        return result
