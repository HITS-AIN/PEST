import os
import time
from dataclasses import dataclass
from enum import Enum

import requests

from pest.get_illustris_apy_key import get_illustris_api_key


class PropertyType(Enum):
    MASS = "mass"


@dataclass
class Selector:
    property: PropertyType
    min_value: float
    max_value: float


class IllustrisDownloader:
    def __init__(
        self,
        base_url: str = "http://www.tng-project.org/api",
        simulation: str = "TNG100-1",
        snapshot: int = 99,
        timeout: int = 100,
    ):
        """Downloads data from the Illustris API.

        Args:
            base_path (str, optional): Defaults to "http://www.tng-project.org/api/TNG100-1/".
            snap_num (int, optional): Defaults to 99.

        """
        self.headers = {"api-key": get_illustris_api_key()}
        self.url = os.path.join(base_url, simulation, "snapshots", str(snapshot))
        self.params = {
            "stars": "Coordinates,Masses",
            "gas": "Coordinates,Potential",
        }
        self.timeout = timeout

    def get_subhalos(
        self,
        limit: int | None = None,
        selector: Selector | None = None,
    ) -> dict:
        """Get subhalo ids from the Illustris API."""

        params = {}
        if limit is not None:
            params["limit"] = limit

        search_query = ""
        if selector is not None:
            search_query = (
                f"?{selector.property.value}__gt="
                + str(selector.min_value)
                + f"&{selector.property.value}__lt="
                + str(selector.max_value)
            )

        start_time = time.time()
        subhalos = requests.get(
            url=os.path.join(self.url, "subhalos", search_query),
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        print(f"Request time: {time.time() - start_time:.2f} seconds")

        subhalos.raise_for_status()
        if subhalos.headers["content-type"] != "application/json":
            raise RuntimeError("Response content is not JSON")

        return subhalos.json()

    def get(self, subhalo_id: int):
        r = requests.get(
            url=os.path.join(self.url, "subhalos", str(subhalo_id)),
            headers=self.headers,
            timeout=self.timeout,
        )

        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()

        if r.headers["content-type"] != "application/json":
            raise RuntimeError("Response content is not JSON")

        return r.json()

    # def get_hdf5(self, subhalo_id: int):
    #     # make HTTP GET request to path
    #     r = requests.get(
    #         url=os.path.join(self.url, "subhalos", str(subhalo_id), "cutout.hdf5"),
    #         headers=self.headers,
    #         params=self.params,
    #         timeout=self.timeout,
    #     )

    #     # raise exception if response code is not HTTP SUCCESS (200)
    #     r.raise_for_status()

    #     if "content-disposition" not in r.headers:
    #         raise RuntimeError("No content-disposition header found")

    #     filename = r.headers["content-disposition"].split("filename=")[1]
    #     with open(os.path.join(self.download_path, filename), "wb") as f:
    #         f.write(r.content)
    #     return filename
