from .blender import BlenderDataset
from .llff import LLFFDataset
from .NHR import NHRDataset
from .NOPC import NOPCDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'nhr':NHRDataset,
                'nopc':NOPCDataset}