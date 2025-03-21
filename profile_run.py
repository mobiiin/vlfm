import cProfile
import runpy

if __name__ == "__main__":
    cProfile.runctx("runpy.run_module('vlfm.run', run_name='__main__')", globals(), locals(), filename="output.prof")