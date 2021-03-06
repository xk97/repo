from pkg_resources import resource_string

class Pkg(object):
    def __init__(self, name=None):
        self.name = name

    def hello(self, word=None):
        print(f'hello {word}')
    
    def __repr__(self) -> str:
        return 'Pkg'
    
    def __call__(self):
        self.hello()

    @property
    def __version__(self):
        return '0'
    
if __name__ == "__main__":
    m = Pkg()
    m()
    print(m.__version__)
    print(resource_string(__name__, 'non_py/query.sql'))