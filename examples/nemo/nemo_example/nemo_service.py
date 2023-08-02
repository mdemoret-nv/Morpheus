import os

from mrc.core import operators as ops
from nemollm.api import NemoLLM


class NeMoService:

    __instance: "NeMoService" = None

    @staticmethod
    def instance(*, api_key: str = None, org_id: str):
        return NeMoService(api_key=api_key, org_id=org_id)

    def __new__(cls, *, api_key: str = None, org_id: str, **kwargs):
        instance = cls.__dict__.get("__instance", None)

        if instance is not None:
            # Verify the api key and org id are the same
            if (instance._api_key != api_key):
                raise RuntimeError("The api key has changed")
            if (instance._org_id != org_id):
                raise RuntimeError("The org id has changed")

            return instance

        instance = object.__new__(cls)

        setattr(cls, "__instance", instance)

        instance.__init__(api_key=api_key, org_id=org_id, **kwargs)

        return instance

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

    def __init__(self, *, api_key: str = None, org_id: str) -> None:

        self._api_key = api_key if api_key is not None else os.environ.get("NGC_API_KEY", None)
        self._org_id = org_id

        # Do checking on api key

        # Class variables
        self._conn: NemoLLM = None

    def start(self):
        self._conn = NemoLLM(
            # The client must configure the authentication and authorization parameters
            # in accordance with the API server security policy.
            # Configure Bearer authorization
            api_key=self._api_key,

            # If you are in more than one LLM-enabled organization, you must
            # specify your org ID in the form of a header. This is optional
            # if you are only in one LLM-enabled org.
            org_id=self._org_id,
        )

    def stop(self):
        pass

    def get_client(self, model_name: str, customization_id: str = None, infer_kwargs: dict = None):

        return NeMoClient(self, model_name=model_name, customization_id=customization_id, infer_kwargs=infer_kwargs)


class NeMoClient:

    def __init__(self,
                 parent: NeMoService,
                 *,
                 model_name: str,
                 customization_id: str = None,
                 infer_kwargs: dict = None) -> None:
        self._parent = parent
        self._model_name = model_name
        self._customization_id = customization_id
        self._infer_kwargs = infer_kwargs if infer_kwargs is not None else {}

    def generate(self, prompt: str | list[str]):

        # Make sure everything is a list
        if (isinstance(prompt, str)):
            prompt = [prompt]

        return self._parent._conn.generate_multiple(
            model=self._model_name,
            prompts=prompt,
            customization_id=self._customization_id,
            return_type="text",
            **self._infer_kwargs,
        )
