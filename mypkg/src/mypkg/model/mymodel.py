from ..pkg import Pkg

class MyModel(object):
    def __init__(self, name=None):
        self.name = name

    def hi(self, word=None):
        print(f'hi {word}')
    
    def __repr__(self) -> str:
        return 'Pkg'
    
    def __call__(self):
        self.hi()

    @property
    def __version__(self):
        return Pkg.__version__
  