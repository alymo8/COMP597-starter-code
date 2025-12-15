from types import ModuleType
from typing import Dict, List, Optional
import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

def _discover_submodules(path : List[str]) -> List[pkgutil.ModuleInfo]:
    submodules = []
    for submodule in pkgutil.iter_modules(path=path):
        submodules.append(submodule)
        logger.debug(f"Found submodule '{submodule.name}' under '{path}'")
    return submodules

def _import_submodule_if_contains_attr(package : str, submodule : pkgutil.ModuleInfo, module_attr_name : str) -> Optional[ModuleType]:
    if not submodule.ispkg:
        return None
    try:
        module = importlib.import_module(name=f".{submodule.name}", package=package)
        if getattr(module, module_attr_name, None) is None:
            return None
    except Exception:
        logger.exception(f"Failed to import '{submodule.name}'")
        return None
    return module

def _get_registration_name(module : ModuleType, allow_name_override : bool, name_override_attr_name : str) -> str:
    default_name = module.__package__.split(".")[-1]
    if not allow_name_override:
        return default_name
    return getattr(module, name_override_attr_name, default_name)

def _register_module(found : Dict[str, object], module : ModuleType, module_attr_name : str, allow_name_override : bool, name_override_attr_name : str) -> Dict[str, object]:
    name = _get_registration_name(module, allow_name_override, name_override_attr_name)
    found[name] = getattr(module, module_attr_name)
    return found

def register(package : str, path : List[str], module_attr_name : str, allow_name_override : bool = False, name_override_attr_name : str = "") -> Dict[str, object]:
    submodules = _discover_submodules(path)
    found = {}
    for submodule in submodules:
        module = _import_submodule_if_contains_attr(package, submodule, module_attr_name)
        if module is None:
            logger.debug(f"Found submodule '{submodule.name}' but it did not have the '{module_attr_name}' attribute.")
            continue
        found = _register_module(found, module, module_attr_name, allow_name_override, name_override_attr_name)
    return found
