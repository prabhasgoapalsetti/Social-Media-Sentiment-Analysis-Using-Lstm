import importlib, sys

def check(module):
    try:
        m = importlib.import_module(module)
        location = getattr(m, '__file__', 'built-in or namespace package')
        print(f"OK: {module} -> {location}")
    except Exception as e:
        print(f"ERROR importing {module}: {e}")

modules = [
    'tensorflow',
    'tensorflow.keras.models',
    'tensorflow.keras.preprocessing.sequence',
    'keras',
]

for mod in modules:
    check(mod)

sys.exit(0)
