"""Fallback data extraction for Olex2 structures without reflection data.

This module deliberately does not use FullMatrixRefine.  It reads values and
published standard uncertainties directly from the CIF and returns the same
basic dictionary shapes used by DataGrabber.grab_xyz(),
DataGrabber.grab_displacement_tensors(), and DataGrabber.grab_distances().

Important limitation:
    Without the original reflection data / least-squares normal matrix, the
    full parameter covariance matrix cannot be reconstructed.  Therefore
    Cartesian coordinate and ADP uncertainties are propagated assuming the
    CIF component uncertainties are uncorrelated. If a CIF geometry bond loop
    is available, its published bond distances/s.u.s are used. Otherwise bonds
    are inferred geometrically and distance s.u.s are approximated from the
    stored coordinate and unit-cell s.u.s, again assuming unavailable
    covariance terms are zero.
"""

import math
import re

import numpy as np
from cctbx import adptbx, uctbx
from cctbx.crystal import calculate_distances
from iotbx.cif import reader


_CIF_NUMBER_RE = re.compile(
    r"^([+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][+-]?\d+)?)"
    r"(?:\((\d+)\))?$"
)


# Single-bond covalent radii in Angstrom. Values are representative small-
# molecule covalent radii and are used only for the last-resort bond inference
# when the CIF contains no _geom_bond_distance loop.
_COVALENT_RADII = {
    "H": 0.31, "D": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53,
    "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19,
    "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64,
    "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39,
    "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03,
    "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90,
    "Yb": 1.87, "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62,
    "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36,
    "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40,
    "At": 1.50, "Rn": 1.50,
    "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06, "Pa": 2.00,
    "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80, "Cm": 1.69,
}


def parse_cif_number(value):
    """Return ``(value, su)`` from a CIF number.

    Examples
    --------
    ``"0.1234(5)"`` -> ``(0.1234, 0.0005)``
    ``"12.3"``      -> ``(12.3, None)``
    ``"?"``         -> ``(None, None)``
    """
    if value is None:
        return None, None

    text = str(value).strip()
    if text in ("", ".", "?"):
        return None, None

    match = _CIF_NUMBER_RE.match(text)
    if match is None:
        try:
            return float(text), None
        except (TypeError, ValueError):
            return None, None

    number_text, su_digits = match.groups()
    number = float(number_text)

    if su_digits is None:
        return number, None

    mantissa = number_text
    exponent = 0
    if "e" in mantissa.lower():
        mantissa, exponent_text = re.split(r"[Ee]", mantissa)
        exponent = int(exponent_text)

    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    su = int(su_digits) * (10.0 ** (exponent - decimals))
    return number, su


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


def _is_nontrivial_symmetry(value):
    if value is None:
        return False
    value = str(value).strip()
    return value not in ("", ".", "?", "1_555")


class NoHKLFallback:
    """Read structural results from a CIF when no HKL data are available."""

    def __init__(self, cif_path):
        self.cif_path = cif_path

        cif_model = reader(file_path=cif_path).model()
        if len(cif_model) == 0:
            raise RuntimeError("CIF contains no data block: %s" % cif_path)

        self.block_name = list(cif_model.keys())[0]
        self.block = cif_model[self.block_name]

        self.unit_cell = uctbx.unit_cell([
            self._required_scalar("_cell_length_a"),
            self._required_scalar("_cell_length_b"),
            self._required_scalar("_cell_length_c"),
            self._required_scalar("_cell_angle_alpha"),
            self._required_scalar("_cell_angle_beta"),
            self._required_scalar("_cell_angle_gamma"),
        ])

        labels = self._column("_atom_site_label")
        if not labels:
            raise RuntimeError("CIF contains no _atom_site_label loop")
        self.labels = [str(label) for label in labels]

        # A CCTBX xray.structure can be built from the CIF without reflection
        # data. It is used only for symmetry-aware neighbour generation.
        try:
            structures = reader(file_path=cif_path).build_crystal_structures()
            self.xray_structure = structures[self.block_name]
        except Exception:
            # Some CIF readers normalise/block-name keys differently.
            structures = reader(file_path=cif_path).build_crystal_structures()
            self.xray_structure = list(structures.values())[0]

        cif_sites, cif_site_errors = self._read_fractional_sites_and_errors()
        cif_index = {label: i for i, label in enumerate(self.labels)}
        xray_labels = list(self.xray_structure.scatterers().extract_labels())
        try:
            self._fractional_sites = [
                cif_sites[cif_index[label]] for label in xray_labels
            ]
            self._fractional_site_errors = [
                cif_site_errors[cif_index[label]] for label in xray_labels
            ]
        except KeyError as exc:
            raise RuntimeError(
                "CIF atom labels and CCTBX structure labels do not match: %s" % exc
            )

        self._cell_parameters, self._cell_errors = self._read_cell_and_errors()

    def _get(self, *tags):
        for tag in tags:
            try:
                value = self.block.get(tag)
            except Exception:
                value = None
            if value is not None:
                return value
        return None

    def _column(self, *tags):
        return _as_list(self._get(*tags))

    def _required_scalar(self, tag):
        value = self._get(tag)
        if value is None:
            raise RuntimeError("Required CIF item is missing: %s" % tag)
        number, _ = parse_cif_number(value)
        if number is None:
            raise RuntimeError("Could not parse CIF item %s = %r" % (tag, value))
        return number

    @staticmethod
    def _propagate_diagonal(J, errors):
        """Propagate independent component s.u.'s through a linear transform."""
        if errors is None or any(err is None for err in errors):
            return None

        cov_in = np.diag(np.asarray(errors, dtype=float) ** 2)
        cov_out = J @ cov_in @ J.T
        return tuple(np.sqrt(np.maximum(np.diag(cov_out), 0.0)))

    def _u_cif_to_cart_jacobian(self):
        """Linear Jacobian for U_cif -> U_cart in CCTBX sym6 ordering."""
        J = np.zeros((6, 6), dtype=float)
        for k in range(6):
            basis = [0.0] * 6
            basis[k] = 1.0
            transformed = adptbx.u_cif_as_u_cart(self.unit_cell, basis)
            J[:, k] = np.asarray(transformed, dtype=float)
        return J

    def _read_fractional_sites_and_errors(self):
        x_col = self._column("_atom_site_fract_x")
        y_col = self._column("_atom_site_fract_y")
        z_col = self._column("_atom_site_fract_z")

        if not x_col or not y_col or not z_col:
            raise RuntimeError("CIF is missing fractional atom coordinates")

        sites = []
        errors = []
        for i, label in enumerate(self.labels):
            x, sx = parse_cif_number(x_col[i])
            y, sy = parse_cif_number(y_col[i])
            z, sz = parse_cif_number(z_col[i])
            if None in (x, y, z):
                raise RuntimeError("Missing coordinates for atom %s" % label)
            sites.append((x, y, z))
            errors.append((sx, sy, sz))
        return sites, errors

    def _read_cell_and_errors(self):
        tags = [
            "_cell_length_a", "_cell_length_b", "_cell_length_c",
            "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
        ]
        values = []
        errors = []
        for tag in tags:
            value, error = parse_cif_number(self._get(tag))
            if value is None:
                raise RuntimeError("Required CIF item is missing: %s" % tag)
            values.append(value)
            errors.append(error)
        return tuple(values), tuple(errors)

    @staticmethod
    def _cell_covariance_from_errors(cell_params, cell_errors):
        """Approximate 6x6 cell covariance from CIF component s.u.s.

        Equal constrained cell lengths are treated as fully correlated, which
        mirrors the usual crystallographic a=b/c constraints. Other unavailable
        cell correlations are taken as zero.
        """
        C = np.zeros((6, 6), dtype=float)
        for i, err in enumerate(cell_errors):
            if err is not None:
                C[i, i] = err * err

        # Common metric constraints such as a=b in tetragonal/hexagonal cells.
        for i in range(3):
            for j in range(i + 1, 3):
                if (
                    cell_errors[i] is not None
                    and cell_errors[j] is not None
                    and cell_params[i] == cell_params[j]
                    and cell_errors[i] == cell_errors[j]
                ):
                    C[i, j] = C[j, i] = cell_errors[i] ** 2
        return C

    @staticmethod
    def _distance_from_parameters(site_i, site_j, cell_params, rt_mx_ji):
        uc = uctbx.unit_cell(cell_params)
        site_j_equiv = rt_mx_ji * tuple(site_j)
        return uc.distance(tuple(site_i), site_j_equiv)

    def _approximate_distance_su(self, i_seq, j_seq, rt_mx_ji):
        """Approximate a bond-distance s.u. without the refinement VCV.

        Coordinate and cell terms are propagated numerically from the s.u.s
        stored in the CIF. Missing covariance terms are assumed to be zero.
        For a symmetry-related self-contact (i_seq == j_seq), the same site
        variables correctly affect both ends of the distance.
        """
        sites = [list(v) for v in self._fractional_sites]
        cell_params = list(self._cell_parameters)

        variance = 0.0
        have_uncertainty = False

        # Coordinate derivatives. Work with unique atom indices so a
        # symmetry-related i==j contact is not double-counted.
        for atom_index in sorted(set((i_seq, j_seq))):
            errs = self._fractional_site_errors[atom_index]
            for component in range(3):
                sigma = errs[component]
                if sigma is None or sigma == 0:
                    continue

                have_uncertainty = True
                h = max(abs(sigma) * 0.05, 1e-7)
                plus_sites = [v[:] for v in sites]
                minus_sites = [v[:] for v in sites]
                plus_sites[atom_index][component] += h
                minus_sites[atom_index][component] -= h

                d_plus = self._distance_from_parameters(
                    plus_sites[i_seq], plus_sites[j_seq], cell_params, rt_mx_ji
                )
                d_minus = self._distance_from_parameters(
                    minus_sites[i_seq], minus_sites[j_seq], cell_params, rt_mx_ji
                )
                deriv = (d_plus - d_minus) / (2.0 * h)
                variance += (deriv * sigma) ** 2

        # Cell derivatives, including the limited known metric correlations.
        cell_grad = np.zeros(6, dtype=float)
        for k, sigma in enumerate(self._cell_errors):
            if sigma is None or sigma == 0:
                continue

            have_uncertainty = True
            if k < 3:
                h = max(abs(sigma) * 0.05, 1e-6)
            else:
                h = max(abs(sigma) * 0.05, 1e-5)

            plus = cell_params[:]
            minus = cell_params[:]
            plus[k] += h
            minus[k] -= h

            d_plus = self._distance_from_parameters(
                sites[i_seq], sites[j_seq], plus, rt_mx_ji
            )
            d_minus = self._distance_from_parameters(
                sites[i_seq], sites[j_seq], minus, rt_mx_ji
            )
            cell_grad[k] = (d_plus - d_minus) / (2.0 * h)

        C_cell = self._cell_covariance_from_errors(
            self._cell_parameters, self._cell_errors
        )
        variance += float(cell_grad @ C_cell @ cell_grad.T)

        if not have_uncertainty:
            return None
        return math.sqrt(max(0.0, variance))

    @staticmethod
    def _element_radius(scatterer):
        try:
            element = scatterer.element_symbol()
        except Exception:
            element = str(scatterer.scattering_type).strip()

        # Normalise isotope-like D to hydrogen radius.
        if element == "D":
            return _COVALENT_RADII["D"]
        return _COVALENT_RADII.get(element)

    def _infer_bond_distances(
        self,
        bond_tolerance=0.45,
        max_search_distance=4.0,
        min_distance=0.4,
    ):
        """Infer bonds when the CIF has no geometry bond loop.

        CCTBX generates symmetry-aware unique neighbour contacts up to
        ``max_search_distance``. Contacts are then retained when their length
        is no larger than r_cov(i) + r_cov(j) + ``bond_tolerance``.

        This is necessarily heuristic; it is only a last-resort replacement
        for an absent _geom_bond_distance loop.
        """
        xs = self.xray_structure
        scatterers = xs.scatterers()
        labels = list(scatterers.extract_labels())

        pair_table = xs.pair_asu_table(distance_cutoff=max_search_distance)
        generated = calculate_distances(
            pair_table,
            xs.sites_frac(),
            skip_j_seq_less_than_i_seq=True,
        )

        distances = {}
        distance_errors = {}
        duplicate_count = {}

        for info in generated:
            distance = float(info.distance)
            if distance < min_distance:
                continue

            r_i = self._element_radius(scatterers[info.i_seq])
            r_j = self._element_radius(scatterers[info.j_seq])

            # Unknown elements: do not guess a bond.
            if r_i is None or r_j is None:
                continue

            if distance > (r_i + r_j + bond_tolerance):
                continue

            label_i = labels[info.i_seq]
            label_j = labels[info.j_seq]
            bond = "%s-%s" % (label_i, label_j)

            if not info.rt_mx_ji.is_unit_mx():
                try:
                    sym = info.rt_mx_ji.as_xyz()
                except Exception:
                    sym = str(info.rt_mx_ji)
                bond += "[sym=%s]" % sym

            if bond in distances:
                duplicate_count[bond] = duplicate_count.get(bond, 1) + 1
                bond = "%s#%d" % (bond, duplicate_count[bond])

            distances[bond] = distance
            distance_errors[bond] = self._approximate_distance_su(
                info.i_seq, info.j_seq, info.rt_mx_ji
            )

        return {
            "distances": distances,
            "distance_errors": distance_errors,
        }

    def grab_xyz(self):
        """Return Cartesian atom coordinates and approximate Cartesian s.u.'s.

        Return shape matches the normal DataGrabber ``grab_xyz()``::

            {
                "C1": {
                    "cart": (x, y, z),
                    "cart_err": (sx, sy, sz),
                },
                ...
            }

        The Cartesian coordinate s.u.'s are propagated from the three CIF
        fractional-coordinate s.u.'s assuming zero covariance between them.
        """
        x_col = self._column("_atom_site_fract_x")
        y_col = self._column("_atom_site_fract_y")
        z_col = self._column("_atom_site_fract_z")

        if not x_col or not y_col or not z_col:
            raise RuntimeError("CIF is missing fractional atom coordinates")

        O = np.asarray(
            self.unit_cell.orthogonalization_matrix(), dtype=float
        ).reshape(3, 3)

        data = {}

        for i, label in enumerate(self.labels):
            x, sx = parse_cif_number(x_col[i])
            y, sy = parse_cif_number(y_col[i])
            z, sz = parse_cif_number(z_col[i])

            if None in (x, y, z):
                raise RuntimeError("Missing coordinates for atom %s" % label)

            frac = np.asarray((x, y, z), dtype=float)
            cart = O @ frac

            cart_err = self._propagate_diagonal(O, (sx, sy, sz))
            if cart_err is None:
                cart_err = (None, None, None)

            data[label] = {
                "cart": tuple(float(v) for v in cart),
                "cart_err": cart_err,
            }

        return data

    def grab_displacement_tensors(self):
        """Return isotropic/anisotropic ADPs and their stored CIF s.u.'s.

        Return shape matches the normal DataGrabber
        ``grab_displacement_tensors()``.

        ``u_cif_err`` comes directly from the parenthesized CIF uncertainties.
        ``u_cart_err`` is propagated from those component uncertainties while
        assuming zero covariance between Uij components, because the original
        full ADP covariance matrix is unavailable without the refinement data.
        """
        data = {
            label: {
                "adp_type": None,
                "u_iso": None,
                "u_iso_err": None,
                "u_cif": None,
                "u_cif_err": None,
                "u_cart": None,
                "u_cart_err": None,
            }
            for label in self.labels
        }

        # --------------------------------------------------------------
        # Anisotropic atoms
        # --------------------------------------------------------------
        aniso_labels = self._column("_atom_site_aniso_label")

        if aniso_labels:
            u_tags = [
                "_atom_site_aniso_U_11",
                "_atom_site_aniso_U_22",
                "_atom_site_aniso_U_33",
                "_atom_site_aniso_U_12",
                "_atom_site_aniso_U_13",
                "_atom_site_aniso_U_23",
            ]
            b_tags = [
                "_atom_site_aniso_B_11",
                "_atom_site_aniso_B_22",
                "_atom_site_aniso_B_33",
                "_atom_site_aniso_B_12",
                "_atom_site_aniso_B_13",
                "_atom_site_aniso_B_23",
            ]

            columns = [self._column(tag) for tag in u_tags]
            scale = 1.0

            if any(col is None for col in columns):
                columns = [self._column(tag) for tag in b_tags]
                scale = 1.0 / (8.0 * math.pi * math.pi)

            if not any(col is None for col in columns):
                J = self._u_cif_to_cart_jacobian()

                for row, raw_label in enumerate(aniso_labels):
                    label = str(raw_label)
                    if label not in data:
                        continue

                    values = []
                    errors = []

                    for col in columns:
                        value, error = parse_cif_number(col[row])
                        values.append(None if value is None else value * scale)
                        errors.append(None if error is None else error * scale)

                    if any(value is None for value in values):
                        continue

                    u_cif = tuple(values)
                    u_cif_err = (
                        tuple(errors)
                        if not any(error is None for error in errors)
                        else None
                    )

                    u_cart = tuple(
                        adptbx.u_cif_as_u_cart(self.unit_cell, u_cif)
                    )
                    u_cart_err = self._propagate_diagonal(J, u_cif_err)

                    data[label].update({
                        "adp_type": "anisotropic",
                        "u_cif": u_cif,
                        "u_cif_err": u_cif_err,
                        "u_cart": u_cart,
                        "u_cart_err": u_cart_err,
                    })

        # --------------------------------------------------------------
        # Isotropic atoms
        # --------------------------------------------------------------
        u_iso_col = self._column(
            "_atom_site_U_iso_or_equiv",
            "_atom_site_u_iso_or_equiv",
        )
        iso_scale = 1.0

        if u_iso_col is None:
            u_iso_col = self._column(
                "_atom_site_B_iso_or_equiv",
                "_atom_site_b_iso_or_equiv",
            )
            iso_scale = 1.0 / (8.0 * math.pi * math.pi)

        if u_iso_col is not None:
            for i, label in enumerate(self.labels):
                # Preserve anisotropic description when present.
                if data[label]["adp_type"] == "anisotropic":
                    continue

                value, error = parse_cif_number(u_iso_col[i])
                if value is None:
                    continue

                data[label].update({
                    "adp_type": "isotropic",
                    "u_iso": value * iso_scale,
                    "u_iso_err": None if error is None else error * iso_scale,
                })

        return data

    def grab_distances(self):
        """Return stored CIF bond distances and their published s.u.'s.

        If a CIF geometry loop is present, its published distance/s.u. is used
        directly and is preferred over any reconstruction. If the loop is
        absent, bonds are inferred from symmetry-aware distances and covalent
        radii, and their s.u.s are approximated from the CIF coordinate/cell
        component s.u.s with unavailable covariances assumed to be zero.

        Return shape matches the normal DataGrabber ``grab_distances()``::

            {
                "distances": {"C1-C2": 1.234, ...},
                "distance_errors": {"C1-C2": 0.004, ...},
            }
        """
        label_1 = self._column("_geom_bond_atom_site_label_1")
        label_2 = self._column("_geom_bond_atom_site_label_2")
        distance_col = self._column("_geom_bond_distance")

        if not label_1 or not label_2 or not distance_col:
            print(
                "No _geom_bond_distance loop found in CIF; "
                "inferring bonds from coordinates and covalent radii."
            )
            return self._infer_bond_distances()

        symmetry_1 = self._column("_geom_bond_site_symmetry_1")
        symmetry_2 = self._column("_geom_bond_site_symmetry_2")

        distances = {}
        distance_errors = {}
        duplicate_count = {}

        for i, raw_distance in enumerate(distance_col):
            atom_1 = str(label_1[i])
            atom_2 = str(label_2[i])
            distance, error = parse_cif_number(raw_distance)

            if distance is None:
                continue

            bond = "%s-%s" % (atom_1, atom_2)

            sym_parts = []
            if symmetry_1 and i < len(symmetry_1) and _is_nontrivial_symmetry(symmetry_1[i]):
                sym_parts.append("%s:%s" % (atom_1, symmetry_1[i]))
            if symmetry_2 and i < len(symmetry_2) and _is_nontrivial_symmetry(symmetry_2[i]):
                sym_parts.append("%s:%s" % (atom_2, symmetry_2[i]))

            if sym_parts:
                bond += "[" + ",".join(sym_parts) + "]"

            # Avoid silently overwriting repeated label pairs.
            if bond in distances:
                duplicate_count[bond] = duplicate_count.get(bond, 1) + 1
                bond = "%s#%d" % (bond, duplicate_count[bond])

            distances[bond] = distance
            distance_errors[bond] = error

        if not distances:
            print(
                "CIF bond loop contained no usable distances; "
                "inferring bonds from coordinates and covalent radii."
            )
            return self._infer_bond_distances()

        return {
            "distances": distances,
            "distance_errors": distance_errors,
        }
