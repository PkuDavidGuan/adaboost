from .Yeast import Yeast

__factory = {
  'yeast':Yeast,
}


def create(name, *args, **kwargs):
  if name not in __factory.keys():
    raise NotImplementedError
  return __factory[name](*args, **kwargs)
