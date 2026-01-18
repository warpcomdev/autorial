Target python version is 3.11
The application is structured as a pipeline of modules in the modules folder.
Each module reads and writes files. All intermediate results are passed via files.
The files either contain media (video, audio) or metadata (e.g., timestamps, configurations).
Metadata files are typically in JSON format. Each module use datatypes to document the structure of these JSON files. A public dataclass is defined for each JSON file type used in the module's public API.
Each module has a `__main__.py` file with a main() function that can be used to run the module from the command line.
The application config uses the ini style config.ini files. The public API of each module exposes a method that receives the config read by configparser and builds a module-specific config object. The input file and output file / folder paths are not part of the config, but separate arguments. The path to the config file is also a command line parameter to the main() function of the module.
Prefer off-the-shelf libraries when implementing functionality.
Use context7 MCP to acquire up to date information about libraries and APIs.
The public API of each module (classes, functions and methods) is documented using pydoc format.
The private implementation details are documented using inline comments. Each private function should have at least a brief docstring describing its purpose and what features depend on it. It should make it easier to identify and eliminate dead code in the future.