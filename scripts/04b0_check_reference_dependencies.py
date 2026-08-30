import importlib
import platform
import sys

packages = [
    ("lifelines", "CoxPH"),
    ("sksurv", "RSF"),
    ("torchtuples", "Deep CoxPH"),
    ("pycox", "Deep CoxPH"),
    ("auton_survival", "DSM"),
]

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print()

all_ok = True
for package, model in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK]      {package:16s} {version:12s} -> {model}")
    except Exception as ex:
        all_ok = False
        print(f"[MISSING] {package:16s} {'':12s} -> {model}")
        print("          ", repr(ex))

print()
if all_ok:
    print("All reference-baseline dependencies are available.")
else:
    print(
        "At least one dependency is missing. The four baseline runners are "
        "independent, so completed models do not need to be rerun."
    )
