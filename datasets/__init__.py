from .blender import BlenderDataset
from .llff import LLFFDataset
from .NHR import NHRDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'nhr':NHRDataset}