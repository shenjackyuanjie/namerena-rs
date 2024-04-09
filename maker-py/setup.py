from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {
    "packages": [],
    "excludes": [
        "test",
        "asyncio",
        "xml",
        "tcl",
        "tkinker",
        "multiprocessing",
        "http",
        "html",
        "logging",
        "email",
        "distutils",
        "unittest",
        "concurrent",
        "pydoc_data",
        "socket",
        "random",
        "lzma",
        "bz2",
        "decimal",
        "ssl"
    ],
}

base = "console"

executables = [Executable("main.py", base=base)]

setup(
    name="namerena-maker",
    version="1.0.0",
    description="名字竞技场-八围制造器",
    options={"build_exe": build_options},
    executables=executables,
)
