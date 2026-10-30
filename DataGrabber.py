import struct
from olexFunctions import OlexFunctions
OV = OlexFunctions()

import os, glob
from datetime import datetime
from iotbx.cif import reader
import olex, olex_core
from cctbx import adptbx, uctbx, geometry, sgtbx
import olx
import numpy as np

from refinement import FullMatrixRefine
# Even though never used we need this import since it initializes things we need later on
from olexex import OlexRefinementModel
from cctbx.array_family import flex
from scitbx import matrix
from cctbx import covariance, xray
from cctbx.crystal import calculate_distances
from math import sqrt

debug = bool(OV.GetParam("olex2.debug", False))


instance_path = OV.DataDir()

try:
  from_outside = False
  p_path = os.path.dirname(os.path.abspath(__file__))
except:
  from_outside = True
  p_path = os.path.dirname(os.path.abspath("__file__"))

l = open(os.sep.join([p_path, 'def.txt'])).readlines()
d = {}
for line in l:
  line = line.strip()
  if not line or line.startswith("#"):
    continue
  d[line.split("=")[0].strip()] = line.split("=")[1].strip()

p_name = d['p_name']
p_htm = d['p_htm']
p_img = eval(d['p_img'])
p_scope = d['p_scope']


from PluginTools import PluginTools as PT


def sym6_to_mat3(u):
    """
    CCTBX sym_mat3 ordering:
    [11, 22, 33, 12, 13, 23]
    """
    return np.array([
        [u[0], u[3], u[4]],
        [u[3], u[1], u[5]],
        [u[4], u[5], u[2]],
    ], dtype=float)


def mat3_to_sym6(m):
    return np.array([
        m[0, 0],
        m[1, 1],
        m[2, 2],
        m[0, 1],
        m[0, 2],
        m[1, 2],
    ], dtype=float)


def sym20_to_mat6(u):
    return np.array([
        [u[0], u[1],  u[2],  u[3],  u[4],  u[5]],
        [u[1], u[6],  u[7],  u[8],  u[9], u[10]],
        [u[2], u[7], u[11], u[12], u[13], u[14]],
        [u[3], u[8], u[12], u[15], u[16], u[17]],
        [u[4], u[9], u[13], u[16], u[18], u[19]],
        [u[5], u[10],u[14], u[17], u[19], u[20]],
    ], dtype=float)


def ustar_to_cart_jacobian(unit_cell):
    """
    Jacobian for transforming
        U* -> U_cart

    in sym6 ordering [11,22,33,12,13,23].
    """
    O = np.array(
        unit_cell.orthogonalization_matrix(),
        dtype=float
    ).reshape(3, 3)

    J = np.zeros((6, 6))

    for k in range(6):
        basis = np.zeros(6)
        basis[k] = 1.0

        U_star = sym6_to_mat3(basis)
        U_cart = O @ U_star @ O.T

        J[:, k] = mat3_to_sym6(U_cart)

    return J


def ustar_to_ucif_jacobian(unit_cell):
    """
    U* -> U_cif is diagonal in CCTBX sym6 representation.
    """
    reciprocal = unit_cell.reciprocal_parameters()

    a_star = reciprocal[0]
    b_star = reciprocal[1]
    c_star = reciprocal[2]

    return np.diag([
        1.0 / a_star**2,
        1.0 / b_star**2,
        1.0 / c_star**2,
        1.0 / (a_star * b_star),
        1.0 / (a_star * c_star),
        1.0 / (b_star * c_star),
    ])

def grab_refinement_stats():
    """
    Collect refinement/model-quality statistics.

    Returns a flat dictionary suitable for writing as one row to a CSV.
    """

    def as_float(value):
        """
        Convert CIF/Olex values to float where possible.

        Handles values such as:
            '0.0321'
            '0.0321(4)'
            '?'
            '.'
            None
        """
        if value is None:
            return None

        value = str(value).strip()

        if value in ("", "?", "."):
            return None

        # Remove CIF uncertainty:
        # 0.0321(4) -> 0.0321
        if "(" in value:
            value = value.split("(")[0]

        try:
            return float(value)
        except (ValueError, TypeError):
            return None


    # ------------------------------------------------------------------
    # Start with values directly available from Olex2
    # ------------------------------------------------------------------

    stats = {
        "R1_gt": None,
        "R1_all": None,
        "wR2": None,
        "goof": None,

        "R_int": None,
        "R_sigma": None,

        "n_reflections": None,
        "n_parameters": None,
        "n_restraints": None,

        "max_shift_over_esd": None,
        "mean_shift_over_esd": None,

        "diff_peak": None,
        "diff_hole": None,
        "diff_rms": None,

        "flack": None,
        "hooft": None,

        "completeness": None,
    }

    # Olex2 refinement diagnostics
    olex_parameters = {
        "goof": "snum.refinement.goof",
        "max_shift_over_esd": "snum.refinement.max_shift_over_esd",
        "diff_peak": "snum.refinement.max_peak",
        "diff_hole": "snum.refinement.max_hole",
        "diff_rms": "snum.refinement.res_rms",
    }

    for key, param in olex_parameters.items():
        try:
            stats[key] = as_float(OV.GetParam(param))
        except Exception:
            pass

    # Hooft is sometimes returned as a formatted string
    try:
        stats["hooft"] = str(
            OV.GetParam("snum.refinement.hooft_str")
        )
    except Exception:
        pass


    # ------------------------------------------------------------------
    # Read standardized values from the last refinement, else from CIF
    # ------------------------------------------------------------------

    # olex2.refine keeps the summary block of its last successful run on the
    # method object (method_imp/cctbx.py, writeRefinementInfoForGui); the .cif
    # is only written with ACTA, so it is the fallback, e.g. for SHELXL.
    # shortcut: the block is from the last olex2.refine run of this session,
    # so reap + grab_data without a refine reports that older run.
    block = None
    try:
        import ExternalPrgParameters
        _, RPD = ExternalPrgParameters.get_program_dictionaries()
        method = RPD.programs[OV.GetParam("snum.refinement.program")] \
            .methods[OV.GetParam("snum.refinement.method")]
        if not getattr(method, "failure", True):
            block = getattr(method, "cif", None)
    except Exception:
        pass

    cif_path = os.path.join(
        OV.FilePath(),
        OV.FileName() + ".cif"
    )

    if block is not None or os.path.isfile(cif_path):

        try:
            if block is None:
                cif_model = reader(
                    file_path=cif_path
                ).model()

                block_name = list(cif_model.keys())[0]
                block = cif_model[block_name]

            cif_tags = {
                # Agreement factors
                "R1_gt":
                    "_refine_ls_R_factor_gt",

                "R1_all":
                    "_refine_ls_R_factor_all",

                "wR2":
                    "_refine_ls_wR_factor_ref",

                "goof":
                    "_refine_ls_goodness_of_fit_ref",

                # Reflection agreement
                "R_int":
                    "_diffrn_reflns_av_R_equivalents",

                "R_sigma":
                    "_diffrn_reflns_av_unetI/netI",

                # Refinement size
                "n_reflections":
                    "_refine_ls_number_reflns",

                "n_parameters":
                    "_refine_ls_number_parameters",

                "n_restraints":
                    "_refine_ls_number_restraints",

                # Convergence
                "max_shift_over_esd":
                    "_refine_ls_shift/su_max",

                "mean_shift_over_esd":
                    "_refine_ls_shift/su_mean",

                # Difference density
                "diff_peak":
                    "_refine_diff_density_max",

                "diff_hole":
                    "_refine_diff_density_min",

                "diff_rms":
                    "_refine_diff_density_rms",

                # Absolute structure
                "flack":
                    "_refine_ls_abs_structure_Flack",

                # Completeness
                "completeness":
                    "_diffrn_measured_fraction_theta_full",
            }

            for key, tag in cif_tags.items():

                try:
                    value = block.get(tag)
                except Exception:
                    value = None

                value = as_float(value)

                # Prefer CIF value if available
                if value is not None:
                    stats[key] = value

        except Exception as exc:
            print(
                "Warning: could not read refinement "
                f"statistics from CIF: {exc}"
            )

    else:
        print(
            f"Warning: CIF not found: {cif_path}"
        )


    # ------------------------------------------------------------------
    # Useful derived quantity
    # ------------------------------------------------------------------

    if (
        stats["n_reflections"] is not None
        and stats["n_parameters"] is not None
        and stats["n_parameters"] != 0
    ):
        stats["data_parameter_ratio"] = (
            stats["n_reflections"]
            / stats["n_parameters"]
        )
    else:
        stats["data_parameter_ratio"] = None

    return stats

def write_data(pos_data, disp_data, dist_data, refine_data, loc):
    """
    Write grabbed refinement data into CSV files that can be read
    directly with pandas.read_csv().

    Creates:
        atoms.csv
        distances.csv
    """
    import csv
    import os

    # ============================================================
    # Atom data
    # ============================================================

    atom_file = os.path.join(loc, "atoms.csv")

    atom_fields = [
        "label",

        # Cartesian coordinates
        "x", "y", "z",
        "x_err", "y_err", "z_err",

        # ADP type
        "adp_type",

        # Isotropic ADP
        "u_iso",
        "u_iso_err",

        # CIF anisotropic ADPs
        "u11", "u22", "u33",
        "u12", "u13", "u23",

        "u11_err", "u22_err", "u33_err",
        "u12_err", "u13_err", "u23_err",

        # Cartesian anisotropic ADPs
        "u11_cart", "u22_cart", "u33_cart",
        "u12_cart", "u13_cart", "u23_cart",

        "u11_cart_err", "u22_cart_err", "u33_cart_err",
        "u12_cart_err", "u13_cart_err", "u23_cart_err",
    ]

    with open(atom_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=atom_fields
        )

        writer.writeheader()

        for label, pos in pos_data.items():

            disp = disp_data.get(label, {})

            cart = pos["cart"]
            cart_err = pos["cart_err"]

            u_cif = disp.get("u_cif")
            u_cif_err = disp.get("u_cif_err")

            u_cart = disp.get("u_cart")
            u_cart_err = disp.get("u_cart_err")

            row = {
                "label": label,

                "x": cart[0],
                "y": cart[1],
                "z": cart[2],

                "x_err": cart_err[0],
                "y_err": cart_err[1],
                "z_err": cart_err[2],

                "adp_type": disp.get("adp_type"),

                "u_iso": disp.get("u_iso"),
                "u_iso_err": disp.get("u_iso_err"),
            }

            # ----------------------------------------------------
            # CIF Uij
            # ----------------------------------------------------

            if u_cif is not None:
                row.update({
                    "u11": u_cif[0],
                    "u22": u_cif[1],
                    "u33": u_cif[2],
                    "u12": u_cif[3],
                    "u13": u_cif[4],
                    "u23": u_cif[5],
                })

            if u_cif_err is not None:
                row.update({
                    "u11_err": u_cif_err[0],
                    "u22_err": u_cif_err[1],
                    "u33_err": u_cif_err[2],
                    "u12_err": u_cif_err[3],
                    "u13_err": u_cif_err[4],
                    "u23_err": u_cif_err[5],
                })

            # ----------------------------------------------------
            # Cartesian Uij
            # ----------------------------------------------------

            if u_cart is not None:
                row.update({
                    "u11_cart": u_cart[0],
                    "u22_cart": u_cart[1],
                    "u33_cart": u_cart[2],
                    "u12_cart": u_cart[3],
                    "u13_cart": u_cart[4],
                    "u23_cart": u_cart[5],
                })

            if u_cart_err is not None:
                row.update({
                    "u11_cart_err": u_cart_err[0],
                    "u22_cart_err": u_cart_err[1],
                    "u33_cart_err": u_cart_err[2],
                    "u12_cart_err": u_cart_err[3],
                    "u13_cart_err": u_cart_err[4],
                    "u23_cart_err": u_cart_err[5],
                })

            writer.writerow(row)

    # ============================================================
    # Distance data
    # ============================================================

    distance_file = os.path.join(loc, "distances.csv")

    with open(distance_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bond",
                "distance",
                "distance_err"
            ]
        )

        writer.writeheader()

        distances = dist_data["distances"]
        errors = dist_data["distance_errors"]

        for bond, distance in distances.items():

            writer.writerow({
                "bond": bond,
                "distance": distance,
                "distance_err": errors.get(bond),
            })

    # ============================================================
    # Refinement statistics
    # ============================================================

    refinement_file = os.path.join(
        loc,
        "refinement.csv"
    )

    with open(
        refinement_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(refine_data.keys())
        )

        writer.writeheader()
        writer.writerow(refine_data)

    print(f"Wrote refinement data to: {refinement_file}")
    print(f"Wrote atom data to: {atom_file}")
    print(f"Wrote distance data to: {distance_file}")

class DataGrabber(PT):
  def __init__(self):
    super(DataGrabber, self).__init__()
    self.p_name = p_name
    self.p_path = p_path
    self.p_scope = p_scope
    self.p_htm = p_htm
    self.p_img = p_img
    self.deal_with_phil(operation='read')
    self.print_version_date()
    if not from_outside:
      self.setup_gui()
    OV.registerFunction(self.grab_data,True,"DataGrabber")
    OV.registerFunction(self.run_refinement,True,"DataGrabber")

  def read_NoSpherA2(self, loc: str) -> float:
    """Reads timing information from NoSpherA2 output file
    
    Args:
        loc (str): Location of the NoSpherA2 output file
        
    Returns:
        dict: Dictionary of timing information
    """
    NoSpherA2_File = glob.glob(os.path.join(loc,"*.NoSpherA2"))
    if len(NoSpherA2_File) == 0:
      print("No NoSpherA2 file found!")
      return 0.0
    if len(NoSpherA2_File) > 1:
      print("Found more than one NoSpherA2 file, using the first one")
    with open(NoSpherA2_File[0], 'r') as file:
      file_content = file.read()
    start_timestamp_str = file_content.split("Refinement startet at: ")[1].split("\n")[0]
    finish_timestamp_str = file_content.split("Refinement finished at: ")[1].split("\n")[0]
    start_timestamp = datetime.strptime(start_timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    finish_timestamp = datetime.strptime(finish_timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    refinement_time = (finish_timestamp - start_timestamp).total_seconds()
    return refinement_time
    
  def write_data(self, data: dict, loc: str):
    """Writes the data to the location in csv format

    Args:
        data (dict): Data to be written
        loc (str): Location to write the data
    """
    for key in data:
      with open(os.path.join(loc, f"output_{key}.txt"), "w") as out:
        #Firts write heading with the keys:
        out.write(",".join(data[key].keys()) + "\n")
        #Then write the data
        for i in range(len(data[key][list(data[key].keys())[0]])): #I hate this sooo much, gets the length of the first key in the nested dictionary
          ret_data = []
          for k in data[key].keys():
              if data[key][k]:
                  ret_data.append(str(data[key][k][i]))
                  
          out.write(",".join(ret_data) + "\n")
        
  def grabber_setup(self):
    """
    Set up the grabber.

    If reflection data are available, build the full covariance matrix.
    If not, fall back to the structural data stored in Olex2/CIF.
    """

    self.name = olx.FileName()
    self.filePath = OV.FilePath()

    self.no_hkl_fallback = None

    self.fmr = None
    self.cov = None
    self.xray_structure = None
    self.unit_cell = None
    self.parameter_map = None

    # This does not require us to rebuild the refinement
    self.olx_model = OlexRefinementModel(True)

    try:
        table_name = str(
            OV.GetParam("snum.NoSpherA2.file")
        )

        self.fmr = FullMatrixRefine()

        if table_name != "":
            norm_eq = self.fmr.run(
                build_only=True,
                table_file_name=table_name
            )
        else:
            norm_eq = self.fmr.run(
                build_only=True
            )

        norm_eq.build_up(False)

        self.cov = norm_eq.covariance_matrix(
            jacobian_transpose=
                norm_eq.reparametrisation
                       .jacobian_transpose_matching_grad_fc()
        )

        self.xray_structure = norm_eq.xray_structure
        self.unit_cell = self.xray_structure.unit_cell()

        self.parameter_map = xray.parameter_map(
            self.xray_structure.scatterers()
        )

        self.has_covariance = True

    except:
        print(
            "Falling back to uncertainties stored "
            "in the structural model/CIF."
        )
        # Cell is still available from Olex2
        self.unit_cell = uctbx.unit_cell(
            self.olx_model.getCell()
        )
        
        from GRAB_no_hkl_fallback import NoHKLFallback
        self.no_hkl_fallback = NoHKLFallback(
            os.path.join(self.filePath, OV.FileName() + ".cif")
        )
        
  def grab_xyz(self):
    if self.no_hkl_fallback:
      return self.no_hkl_fallback.grab_xyz()
    
    cov_sites_cart = covariance.orthogonalize_covariance_matrix(
      self.cov,
      self.unit_cell,
      self.parameter_map
    )
    data = {}
    sites_cart = self.xray_structure.sites_cart()
    for i, scatterer in enumerate(self.xray_structure.scatterers()):
      xyz_cart = tuple(sites_cart[i])

      site_cov_cart = covariance.extract_covariance_matrix_for_sites(
          flex.size_t([i]),
          cov_sites_cart,
          self.parameter_map
      )

      xyz_cart_err = tuple(
          [sqrt(max(0.0, site_cov_cart[0])),  #Assuming site_cov_cart ist upper triangle vector with diagonal first
          sqrt(max(0.0, site_cov_cart[3])),
          sqrt(max(0.0, site_cov_cart[5]))]
      )
      
      data[scatterer.label] = {
          "cart": xyz_cart,
          "cart_err": xyz_cart_err,
      }
      
    return data

  def grab_displacement_tensors(self):
    if self.no_hkl_fallback:
      return self.no_hkl_fallback.grab_displacement_tensors()
    
    J_ucif = ustar_to_ucif_jacobian(self.unit_cell)
    J_ucart = ustar_to_cart_jacobian(self.unit_cell)

    data = {}

    for i, scatterer in enumerate(self.xray_structure.scatterers()):

      p = self.parameter_map[i]
      
      adp_type = None
      u_iso = None
      u_iso_err = None
      u_cif = None
      u_cif_err = None
      u_cart = None
      u_cart_err = None

      
      # ------------------------------------------------------------
      # Anisotropic ADP
      # ------------------------------------------------------------
      if scatterer.flags.use_u_aniso():

        adp_type = "anisotropic"

        # CCTBX internally stores anisotropic ADPs as U*
        u_star = scatterer.u_star

        # Standard crystallographic Uij values
        u_cif = tuple(
          adptbx.u_star_as_u_cif(
              self.unit_cell,
              u_star
          )
        )

        # Cartesian displacement tensor
        u_cart = tuple(
          adptbx.u_star_as_u_cart(
              self.unit_cell,
              u_star
          )
        )

        # Was the anisotropic ADP actually refined?
        if p.u_aniso >= 0:
          cov_u_star_packed = (
            covariance.extract_covariance_matrix_for_u_aniso(
              i,
              self.cov,
              self.parameter_map
            )
          )

          cov_u_star = sym20_to_mat6(
              cov_u_star_packed
          )

          # Propagate the complete 6x6 covariance matrix
          cov_u_cif = (
              J_ucif
              @ cov_u_star
              @ J_ucif.T
          )

          cov_u_cart = (
              J_ucart
              @ cov_u_star
              @ J_ucart.T
          )

          u_cif_err = tuple(
              sqrt(max(0.0, cov_u_cif[j, j]))
              for j in range(6)
          )

          u_cart_err = tuple(
              sqrt(max(0.0, cov_u_cart[j, j]))
              for j in range(6)
          )

      # ------------------------------------------------------------
      # Isotropic ADP
      # ------------------------------------------------------------
      elif scatterer.flags.use_u_iso():
        adp_type = "isotropic"

        u_iso = scatterer.u_iso

        if p.u_iso >= 0:
          variance = covariance.variance_for_u_iso(
              i,
              self.cov,
              self.parameter_map
          )

          u_iso_err = sqrt(
              max(0.0, variance)
          )
      
      data[scatterer.label] = {
          "adp_type": adp_type,
          "u_iso": u_iso,
          "u_iso_err": u_iso_err,
          "u_cif": u_cif,
          "u_cif_err": u_cif_err,
          "u_cart": u_cart,
          "u_cart_err": u_cart_err,
      }
          
    return data

  def grab_distances(self):
    """Grabs the distances and returns it as a dictionary
    """
    if self.no_hkl_fallback:
      return self.no_hkl_fallback.grab_distances()
    
    import math
    cell_params = self.fmr.olx_atoms.getCell()
    cell_errors = self.fmr.olx_atoms.getCellErrors()
    cell_vcv = flex.pow2(matrix.diag(cell_errors).as_flex_double_matrix())
    for i in range(3):
      for j in range(i+1,3):
        if (cell_params[i] == cell_params[j] and
            cell_errors[i] == cell_errors[j] and
            cell_params[i+3] == 90 and
            cell_errors[i+3] == 0 and
            cell_params[j+3] == 90 and
            cell_errors[j+3] == 0):
          cell_vcv[i,j] = math.pow(cell_errors[i],2)
          cell_vcv[j,i] = math.pow(cell_errors[i],2)
          
    cell_vcv = cell_vcv.matrix_symmetric_as_packed_u()

    sl = self.xray_structure.scatterers().extract_labels()
    sf = self.xray_structure.sites_frac()
    pat = self.fmr.reparametrisation.connectivity_table.pair_asu_table
    
    distances = calculate_distances(
      pat,
      sf,
      covariance_matrix=self.cov,
      cell_covariance_matrix=cell_vcv,
      parameter_map=self.parameter_map)

    dist_stats = {}
    dist_errs = {}
    for i,d in enumerate(distances):
      bond = sl[d.i_seq]+"-"+sl[d.j_seq]
      dist_stats[bond] = distances.distances[i]
      try:
        dist_errs[bond] = math.sqrt(distances.variances[i])
      except:
        dist_var = distances.variances[i]
        if abs(dist_var) < 1e-10:
          dist_errs[bond] = 0.0
        else:
          print(f"Warning: distance variance for {bond} is negative: {dist_var}")
          dist_errs[bond] = None

      
    return {"distances": dist_stats, "distance_errors": dist_errs}

  def grab_data(self):
    """Grabs the data and returns it as a dictionary"""
    self.grabber_setup()

    pos_data = self.grab_xyz()
    
    disp_data = self.grab_displacement_tensors()
    
    dist_data = self.grab_distances()
    
    refine_stats = grab_refinement_stats()
    
    refine_stats["NoSpherA2_time"] = self.read_NoSpherA2(self.filePath)
    
    write_data(
      pos_data,
      disp_data,
      dist_data,
      refine_stats,
      os.path.join(self.filePath)
    )
    
  def run_refinement(self, model_name: str = "Combo_v6"):
    """Runs the refinement with currently selected parameters
    """
    #Doing one regular refinement
    if os.environ.get("MODEL_NAME"):
      model_name = os.environ.get("MODEL_NAME")
    
    OV.SetParam('snum.NoSpherA2.use_aspherical',False)
    olex.m("sel $H")
    olex.m("isot")
    olex.m("refine 5")
    
    
    olex.m("neutronhdist")
    
    olx.xf.EndUpdate()
    if OV.HasGUI():
      olx.Refresh()
    OV.SetParam('snum.NoSpherA2.multiplicity', "1")
    OV.SetParam('snum.NoSpherA2.h_aniso', True)
    OV.SetParam('snum.NoSpherA2.use_aspherical',True)
    OV.SetParam('snum.NoSpherA2.Calculate',True)
    OV.SetParam('snum.NoSpherA2.source','SALTED')
    OV.SetParam('snum.NoSpherA2.selected_salted_model',os.path.join(r"D:\Models", model_name))
    OV.SetParam('snum.NoSpherA2.full_HAR', True)
    OV.SetParam('snum.NoSpherA2.Max_HAR_Cycles',10)
    OV.SetParam('snum.NoSpherA2.ncpus',10)
    OV.SetParam('snum.NoSpherA2.precise_output',True)
    OV.SetParam('snum.refinement.update_weight', True)

    OV.cmd("addins acta")
    olex.m("refine 10")
    
    print(f"Last shift was: {OV.GetParam('snum.refinement.max_shift_over_esd')}")
    counter = 0
    while abs(OV.GetParam('snum.refinement.max_shift_over_esd')) > 0.005:
      olex.m("refine 20")
      counter += 1
      if counter > 15:
        print()
        break
      print(f"Last shift was: {OV.GetParam('snum.refinement.max_shift_over_esd')}")
    
test_instance = DataGrabber()

if os.environ.get("DATAGRABBER") == "TRUE":
  job_path, ins_name = os.environ.get("JOB_DIR"), os.environ.get("INS_NAME")
  print("Running Data Grabber!")
  olex.m(f"reap {os.path.join(job_path, ins_name)}")
  # olex.m("spy.DataGrabber.")
  # olex.m("spy.DataGrabber.grab_data()")
  