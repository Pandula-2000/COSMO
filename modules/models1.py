
from . import gauss
from . import mfn
from . import relu
from . import siren 
from . import wire
from . import wire2d
from . import incode
from . import finer
from . import BandRC
from . import COSMOV2
from . import COSMOV3
from . import COSMOV4



model_dict = {'gauss': gauss,
              'mfn': mfn,
              'relu': relu,
              'siren': siren,
              'wire': wire,
              'wire2d': wire2d,
              'ffn': None,
              'incode': incode,
              'finer': finer,
              'BandRC': BandRC,
              'COSMOV2': COSMOV2,
              "COSMOV3": COSMOV3,
              "COSMOV4": COSMOV4
              }


class INR():
    def __init__(self, nonlin):
        self.nonlin = nonlin
        self.model = model_dict[nonlin]

    def run(self, *args, **kwargs):

        if self.nonlin == 'ffn':
            if kwargs['ffn_type'] in ['relu', 'swish']:
                self.model = model_dict['relu']
            elif kwargs['ffn_type'] in ['siren']: 
                self.model = model_dict['siren']
            else:
                assert "Invalid ffn_type. Choose from: [relu, swish, siren]"
        
        return self.model.INR(*args, **kwargs)