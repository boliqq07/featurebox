from pyxtal_ff import PyXtal_FF

train_data = r"C:\ProgramData\Anaconda3\Lib\site-packages\pyxtal_ffpyxtal_ff\datasets\Si\PyXtal\Si8.json"
descriptors = {'type': 'SOAP',
               'Rc': 5.0,
               'parameters': {'lmax': 4, 'nmax': 3},
               'N_train': 400,
              }
model = {'system': ['Si'],
         'hiddenlayers': [16, 16],
        }
ff = PyXtal_FF(descriptors=descriptors, model=model)
ff.run(mode='train', TrainData=train_data)
import pymatgen