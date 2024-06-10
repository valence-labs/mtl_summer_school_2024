from typing import Union 
import os 
from typing import Union

import subprocess
import os
import re
import tempfile
import numpy as np

from loguru import logger

import openmm.unit as unit
from openmm import app

import fsspec
import fsspec.utils
from tqdm import tqdm
from loguru import logger 
import datamol as dm
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import rdMolHash
from rdkit.Geometry import Point3D  # type: ignore
from dataclasses import dataclass
from typing import Union, Optional, List

class Preprocessor:
    def prepare_receptor(
            self,
            receptor_path: Union[str, os.PathLike],
            vina_receptor_path: Union[str, os.PathLike],
        ):
            """Convert a receptor PDB file to PDBQT for vina docking."""

            convert(
                infile=receptor_path,
                in_format="pdb",
                out_format="pdbqt",
                outfile=vina_receptor_path,
                overwrite=True,
                rigid=True,
                struct_num=0,
                calc_charges=True,
                add_h=True,
            )

    def prepare_ligand(
        self,
        ligand_path: Union[str, os.PathLike],
        vina_ligand_path: Union[str, os.PathLike],
        in_format: str = "sdf",
    ):
        convert(
                infile=ligand_path,
                in_format=in_format,
                out_format="pdbqt",
                outfile=vina_ligand_path,
                overwrite=True,
                struct_num=0,
                calc_charges=True,
                charges_model="gasteiger",
                add_h=True,
                remove_h=False,
            )
        
def convert(
    infile: Union[str, os.PathLike],
    out_format: str,
    outfile: Union[str, os.PathLike] = None,
    in_format: str = None,
    make_3d: bool = False,
    overwrite: bool = False,
    rigid: bool = False,
    struct_num: int = None,
    calc_charges: bool = True,
    charges_model: str = "gasteiger",
    add_h: bool = True,
    remove_h: bool = False,
) -> Union[str, os.PathLike]:
    """Convert an input file to the output_format.
    This is really just a wrapper around pybel
    Args:
        infile: Input file
        out_format: output format
        outfile: Output file. If not provided, we will replace the
        in_format: Input format. If not provided, will be guessed from the extension
        make_3d: Whether to call mol.make3D() on the molecule when a 3D structure is missing
        overwrite: Whether to overwrite file if exists already
        rigid: whether to allow torsion angle in pdbqt writing. For receptors, rigid should be true.
        struct_num: structure number to use when the output is supposed to be pdbqt.
    Returns:
        outfile: path to the converted file
    """
    try:
        from openbabel import pybel
    except ImportError:
        raise ImportError("File convertion requires openbabel >= 3.0.0")
    infile = str(infile)
    outfile = str(outfile)
    if in_format is None:
        in_format = os.path.splitext(infile)[-1].lower().strip(".")
    if out_format not in pybel.outformats:
        raise ValueError(f"Output format {out_format} is not recognized by pybel !")
    if outfile is None:
        outfile = infile.replace(in_format, out_format)
    opt = {"align": None}
    if not rigid:
        opt = {"b": None, "p": None, "h": None}
    else:
        opt = {"r": None, "c": None, "h": None}
    if struct_num is None:
        raise ValueError(
                f"For pdbqt, you need to provide the number `struct_num` of the compound to save"
            )
    mols = _read_ob(infile, in_format)
    if struct_num is not None:
        mols = [mols[struct_num]]
    tmp_file = outfile
    out = pybel.Outputfile(format=out_format, filename=tmp_file, overwrite=overwrite, opt=opt)
    for m in mols:
        if add_h:
            m.addh()
        if remove_h:
            m.removeh()
        if not m.OBMol.HasNonZeroCoords() and make_3d is True:
            m.make3D()
        if calc_charges is True:
            m.calccharges(model=charges_model)
        out.write(m)
    out.close()
    if tmp_file != outfile:
        dm.utils.fs.copy_file(tmp_file, outfile)
        os.unlink(tmp_file)
    return outfile

def _read_ob(infile, in_format):
    """Read a molecule file with open babel
    Args:
        infile (Union[str, os.PathLike]): input file
        in_format (str, optional): input format
    Returns:
        mols (list): list of molecules found in the input file
    """
    try:
        from openbabel import pybel
    except ImportError:
        raise ImportError("Pybel is required for reading openbabel molecules")
    mols = [m for m in pybel.readfile(format=in_format, filename=infile)]
    return mols
  
@dataclass
class Point:
    x: float
    y: float
    z: float
    
    def from_array(arr: np.ndarray):
        return Point(arr[0], arr[1], arr[2])
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
      
@dataclass
class Box:
    center: Point 
    size: Point 
    
    def from_array(center, size):
        return Box(Point.from_array(center), Point.from_array(size))
    
class Docking:
    def __init__(self, receptor_path, box : Box, num_poses : int = 5, exhaustiveness : int = 8):
        self._num_poses = num_poses
        self._exhaustiveness = exhaustiveness
        self.receptor_path = receptor_path
        self.pocket_size = box.size.to_array()
        self.pocket_center = box.center.to_array()
        
    @property
    def num_poses(self):
        return self._num_poses
    
    @property
    def exhaustiveness(self):
        return self._exhaustiveness
    
    
    def dock_one(self, ligand_path, out_path):
        assert os.path.splitext(ligand_path)[-1].lower().strip(".") == "pdbqt", "Ligand file must be in pdbqt format"
        output_text = subprocess.check_output(
            [
                "smina",
                "--ligand",
                str(ligand_path),
                "--receptor",
                str(self.receptor_path),
                "--out",
                str(out_path),
                "--center_x",
                str(self.pocket_center[0]),
                "--center_y",
                str(self.pocket_center[1]),
                "--center_z",
                str(self.pocket_center[2]),
                "--size_x",
                str(self.pocket_size[0]),
                "--size_y",
                str(self.pocket_size[1]),
                "--size_z",
                str(self.pocket_size[2]),
                "--num_modes",
                str(self.num_poses),
                "--exhaustiveness",
                str(self.exhaustiveness)
            ],
            universal_newlines=True  
        )
        return output_text
    
    def parse_output(self, output_text):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas not found")
            
        smina_delimiter = "-----+------------+----------+----------"

        # Parse lines
        lines = [line for line in output_text.split("\n")]
        # Get index of the delimiter
        rows_index = lines.index(smina_delimiter)
        # Parse values
        results = []
        for line in lines[rows_index + 1 : -3]:
            values = [float(x) for x in line.split()[1:]]
            results.append(values)
        results = pd.DataFrame(results)
        results.columns = ["affinity", "rmsd_lb", "rmsd_ub"]
        return results
    
    def parse_mol_to_pbdqt(self, mol,  out_dir: str = "sdf_inputs" , idx: int =0):
        os.makedirs(out_dir, exist_ok=True)
        path=os.path.join(out_dir, f"mol_{idx}.sdf")
        dm.to_sdf(mol, path)
        Preprocessor().prepare_ligand(
            path,
            os.path.join("smina_inputs", f"mol_{idx}.pdbqt")
        )
        
    def dock_multiple_mols(
        self,
        mols: List[dm.Mol], 
        idxs : Optional[List[int]]=None,
        output_dir : Optional[str]=None,
    ):
        import pathlib
        if not idxs:
            idxs = list(range(mols))
        assert len(mols) == len(idxs), "Different numbers of molecules and job ids"
        
        if not output_dir:
            output_dir = "smina_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Converting mols to pdbqt in 'smina_inputs' folder")
        for mol,idx in zip(mols, idxs):
            self.parse_mol_to_pbdqt(mol, "smina_inputs", idx)
            
        logger.info(f"Docking")
        for idx in tqdm(idxs):
            out_path=os.path.join(output_dir, f"poses_{idx}.sdf")
            self.dock_one(os.path.join("smina_inputs", f"mol_{idx}.pdbqt"), out_path)
            
        logger.info(f"Merge all the generated poses together to {output_dir}")
        with dm.without_rdkit_log():
            all_poses = []
            all_poses_paths = pathlib.Path(output_dir).rglob("*.sdf")
            for poses_path in all_poses_paths:
                all_poses += dm.io.read_sdf(poses_path, sanitize=False)
            dm.io.to_sdf(all_poses, os.path.join(output_dir, "poses.sdf"))
        
    
# ACTIVE LEARNING 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from typing import List 
import pandas as pd 

def init_df_fields(df):
    from copy import deepcopy
    df_tmp = deepcopy(df)
    df_tmp["idxs"]=list(range(len(df)))
    df_tmp["pred_affinity"] = np.nan 
    df_tmp["uncertainty"] = np.nan 
    df_tmp["true_affinity"] = np.nan 
    df_tmp["sampled"] = 0
    return df_tmp

def format_df(df, affinities, sampled_idxs, iteration):
    df["true_affinity"][sampled_idxs]=affinities
    df["sampled"][sampled_idxs]=iteration
    return df

def get_results(output_dir, idxs):
    values=[]
    key="minimizedAffinity"
    for idx in idxs:
        poses= dm.read_sdf(os.path.join(output_dir, f"poses_{idx}.sdf"), as_df=True, mol_column="mols", n_jobs=-1, sanitize=False)
        poses=poses.sort_values("minimizedAffinity",inplace=False)
        values.append(poses["minimizedAffinity"][0])
    return values

def train_gp(df) -> GaussianProcessRegressor:
    from sklearn.gaussian_process.kernels import RBF
    X = np.vstack(df["fp"][df["sampled"]>=1].tolist())
    Y = np.vstack(df["true_affinity"][df["sampled"]>=1].tolist())
    return GaussianProcessRegressor(kernel=RBF(length_scale=2.0, length_scale_bounds=(1e-1, 20.0)), random_state=0).fit(X,Y)

def predict_with_gp(df, gp):
    X = np.vstack(df["fp"].tolist())
    Y = np.vstack(df["true_affinity"].tolist())
    mean, std = gp.predict(X, return_std=True)
    df["pred_affinity"] = mean
    df["uncertainty"] = std
    return df

def samples_next(df, n: int = 10, sort_by_uncertainty = True) -> List[int]:
    original_df = df
    if sort_by_uncertainty:
        ascending=False
        name="uncertainty"
    else:
        ascending=True
        name="pred_affinity"        
    return df.sort_values(name, ascending=ascending)["idxs"].tolist()[:n]

def plot_AL(df, to : int=3):
    datum={"affinity": [], "smiles" : [], "mols": [], "it": []}
    for idx in range(1,to):
        values=df["true_affinity"][df["sampled"]==idx].tolist()
        smiles=df["smiles"][df["sampled"]==idx].tolist()
        mols=df["mols"][df["sampled"]==idx].tolist()
        iteration = df["sampled"][df["sampled"]==idx].tolist()
        datum["affinity"].append(values)
        datum["smiles"].append(smiles)
        datum["mols"].append(mols)
        datum["it"].append(iteration)
    return pd.DataFrame(datum)