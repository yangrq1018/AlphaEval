import fire
from my_qlib import auto_init
from my_qlib.contrib.rolling.base import Rolling
from my_qlib.utils.mod import find_all_classes

if __name__ == "__main__":
    sub_commands = {}
    for cls in find_all_classes("my_qlib.contrib.rolling", Rolling):
        sub_commands[cls.__module__.split(".")[-1]] = cls
    # The sub_commands will be like
    # {'base': <class 'my_qlib.contrib.rolling.base.Rolling'>, ...}
    # So the you can run it with commands like command below
    # - `python -m my_qlib.contrib.rolling base --conf_path <path to the yaml> run`
    # - base can be replace with other module names
    auto_init()
    fire.Fire(sub_commands)
