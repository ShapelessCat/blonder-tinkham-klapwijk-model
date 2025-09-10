# Blonder-Tinkham-Klapwijk Model

## Setup Python Virtual Environment

1. Make sure you have installed **uv**:
   <https://docs.astral.sh/uv/getting-started/installation/>

2. In the top level of this project, create a Python virtual environment (with a
   default name, the project name *blonder-tinkham-klapwijk-model*):

   ```shell
   uv venv
   ```
   
   The commands used to activate this virtual environment in different systems
   are different:

   - In *nix: `source .venv/bin/activate`
   - In Windows: `./.venv/Scripts/activate`

3. In the top level of this project, install all project dependencies with command

   ```shell
   uv sync --all-extras --all-groups --no-install-project
   ```

## Package this project and run it through command line

**Make sure you have done with the steps in previous section, and the Python
virtual environment has been created and activated.**

1. Package this project to a *.pyz* format file:

   ```shell
   shiv -e blonder-tinkham-klapwijk-model.main:main -o btk.x .
   ```
   
2. Run this package project through command line:

   In *nix systems, you can run btk.x as an executable, and use `./btk.x --help`
   to learn how to use it. In Windows, for the reason I don't know, you need to
   use Python to run it: `python ./btk.x <parameters>`. Make sure you already
   activated the right Python virtual environment.

   - If you are in the top level of this project, you can use this command to
     run an example:

     ```shell
     python .\btk.x -c tests\data\input_parameters.toml -d tests\data\3K_modified.dat 
     ```

     The first run of this package is slow, the following runs can be faster.
   