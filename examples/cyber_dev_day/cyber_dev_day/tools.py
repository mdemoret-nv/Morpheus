# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import warnings
from textwrap import dedent

from packaging.version import InvalidVersion
from packaging.version import parse as parse_version

logger = logging.getLogger(__name__)


def range_version_comparator(software_version: str, vulnerability_lower_range: str, vulnerability_upper_range: str):
    """
    Compare a software's version to a range of vulnerable versions to determine vulnerability.

    Parameters
    ----------
    software_version : str
        The version of the software currently in use.
    vulnerability_lower_range : str
        The lower bound of the vulnerable version range.
    vulnerability_upper_range : str
        The upper bound of the vulnerable version range.

    Returns
    -------
    bool
        Returns True if the software version is within the range of vulnerable versions,
        indicating potential vulnerability.

    Raises
    ------
    InvalidVersion
        If the version strings are not in a valid format, a warning is issued and alphabetic
        comparison is used instead.

    Notes
    -----
    This function assumes that the software is vulnerable if its version falls inclusively
    between the lower and upper bounds of the vulnerability range. It uses the `parse_version`
    function to interpret the versions and compares them accordingly. If `parse_version` fails,
    Debian version parsing is attempted. Finally, if both of these fail, it falls
    back to a simple string comparison.
    """
    try:
        sv = parse_version(str(software_version))
        lvv = parse_version(str(vulnerability_lower_range))
        uvv = parse_version(str(vulnerability_upper_range))
        return sv <= uvv and sv >= lvv
    except InvalidVersion:
        #Failed PEP440 versioning; moving on to Debian
        pass

    try:
        return Dpkg.compare_versions(str(software_version),
                                     str(vulnerability_lower_range)) != -1 and Dpkg.compare_versions(
                                         str(software_version), str(vulnerability_upper_range)) != 1
    except DpkgVersionError:
        warnings.warn('Unable to parse provided versions. Using alpha sorting.', stacklevel=2)
    # Fallback to alphabetic comparison if version parsing fails
    return str(software_version) <= str(vulnerability_upper_range) and str(software_version) >= str(
        vulnerability_lower_range)


def single_version_comparator(software_version: str, vulnerability_version: str):
    """
    Compare a software's version to a known vulnerable version.

    Parameters
    ----------
    software_version : str
        The version of the software currently in use.
    vulnerability_version : str
        The version of the software that is known to be vulnerable.

    Returns
    -------
    bool
        Returns True if the software version is less than or equal to the vulnerability version,
        indicating potential vulnerability.

    Raises
    ------
    InvalidVersion
        If the version strings are not in a valid format, a warning is issued and alphabetic
        comparison is used instead.
    """
    try:
        sv = parse_version(str(software_version))
        vv = parse_version(str(vulnerability_version))
        return sv <= vv
    except InvalidVersion:
        #Failed PEP440 versioning; moving on to Debian
        pass
    try:
        return Dpkg.compare_versions(str(software_version), str(vulnerability_version)) != 1
    except DpkgVersionError:
        warnings.warn('Unable to parse provided versions. Using alpha sorting.', stacklevel=2)
    return str(software_version) <= str(vulnerability_version)


def version_comparison(software_version: str):
    """
    Compare a software's version to multiple known vulnerable versions.

    Parameters
    ----------
    software_version : str
        A string containing the software version to compare, and the vulnerable versions,
        separated by commas. A single vulnerable version, a vulnerable range (two versions),
        or multiple specific vulnerable versions can be provided.

    Returns
    -------
    bool or str
        Returns True if the software version matches any of the vulnerable versions,
        or is within the vulnerable range. Returns a string message if the input doesn't
        contain enough information for a comparison.

    Notes
    -----
    This function can compare against a single vulnerable version, a range of versions,
    or a list of specific versions. It uses the `single_version_comparator` for single comparisons,
    and `range_version_comparator` for range comparisons.
    """
    v = software_version.split(',')
    if len(v) == 2:
        return single_version_comparator(v[0], v[1])
    elif len(v) == 3:
        return range_version_comparator(v[0], v[1], v[2])
    elif len(v) > 3:
        return any([v[0] == v_ for v_ in v[1:]])
    else:
        return "Couldn't able compare the software version, not enough input"


class SBOMChecker:

    tool_description = dedent("""
        Useful for when you need to check the Docker container's software bill of
        materials (SBOM) to get whether or not a given library is in the container.
        Input should be the name of the library or software. If the package is
        present a version number is returned, otherwise False is returned if the
        package is not present.
    """).replace("\n", "")

    def __init__(self, sbom_map: dict[str, str]):
        self.sbom_map = sbom_map

    def sbom_checker(self, package_name: str):
        "use this tool to check the version of the software package from the SBOM"
        "returns the software version if the package is present in the SBOM"
        "if the package is not in the SBOM returns False"

        try:
            version = self.sbom_map.get(package_name.lower().strip(), False)
        except Exception as e:
            warnings.warn(str(e), stacklevel=2)
            version = False
        return version

    @staticmethod
    def from_csv(file_path: str) -> "SBOMChecker":
        """
        Use this tool to load the SBOM from a CSV file returns an instance of the SBOMChecker class
        """
        try:
            import pandas as pd
            sbom = pd.read_csv(file_path)
            sbom_map = dict(zip(sbom['package_name'].str.lower(), sbom['version']))
            return SBOMChecker(sbom_map)
        except Exception as e:
            logger.error("Error loading SBOM from CSV file: %s. Error: %s", file_path, str(e), exc_info=True)
            raise e
