import argparse
import logging
import sys
from pathlib import Path
import mypkg
if __name__ == "__main__":
    work_dir = Path(Path(__file__).parent)
    print(work_dir, __file__)
    work_dir.joinpath('tmp').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=work_dir / 'tmp'/ 'app.log', level=logging.INFO, filemode='w')
    logger = logging.getLogger()
    logger.info('pkg version {}'.format(mypkg.__version__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    myargs = parser.parse_args()
    logger.info(myargs)
    if myargs.debug:
        try:
            import IPython
            IPython.embed()
        except ImportError as e:
            print(e)
    pass