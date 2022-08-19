import requests
from bs4 import BeautifulSoup
import re
import pdb
import pandas as pd
import os
import pathlib
from tqdm import tqdm

from path_constants import ZEOLITE_PRIOR_FILE
from utilities import save_matrix


def scrape_main_page(url):
    properties = {}
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="containerbody")

    top_table = results.find_all("tr")[0].find_all("tr")
    abc = top_table[1].find_all("td")
    properties["a"] = re.sub("[^0-9|.]", "", abc[2].text)
    properties["b"] = re.sub("[^0-9|.]", "", abc[3].text)
    properties["c"] = re.sub("[^0-9|.]", "", abc[4].text)

    alphabettagamma = top_table[2].find_all("td")
    properties["alpha"] = re.sub("[^0-9|.]", "", alphabettagamma[1].text)
    properties["betta"] = re.sub("[^0-9|.]", "", alphabettagamma[2].text)
    properties["gamma"] = re.sub("[^0-9|.]", "", alphabettagamma[3].text)

    properties["volume"] = re.sub("[^0-9|.]", "", top_table[3].text.replace("Å3", ""))
    # refinement distance least squares
    properties["rdls"] = re.sub("[^0-9|.]", "", top_table[4].text)
    properties["framework_density"] = re.sub(
        "[^0-9|.]", "", top_table[5].text.replace("/1000", "").replace("Å3", "")
    )

    densities = top_table[6].find_all("td")
    properties["td_10"] = re.sub("[^0-9|.]", "", densities[2].text.replace("TD10", ""))
    properties["td"] = re.sub("[^0-9|.]", "", densities[3].text)

    ring_sizes = top_table[7].find_all("td")[2]
    ring_sizes_array = ring_sizes.text.split("\xa0\xa0")
    ring_sizes_array.remove("")
    for index, ring_size in enumerate(ring_sizes_array):
        properties["ring_size_" + str(index)] = ring_size

    properties["included_sphere_diameter"] = re.sub(
        "[^0-9|.]", "", top_table[10].find_all("td")[2].text
    )

    diffused_sphere_diameters = top_table[11].find_all("td")
    properties["diffused_sphere_diameter_a"] = re.sub(
        "[^0-9|.]", "", diffused_sphere_diameters[2].text
    )
    properties["diffused_sphere_diameter_b"] = re.sub(
        "[^0-9|.]", "", diffused_sphere_diameters[3].text
    )
    properties["diffused_sphere_diameter_c"] = re.sub(
        "[^0-9|.]", "", diffused_sphere_diameters[4].text
    )

    properties["accessible_volume"] = re.sub("[^0-9|.]", "", top_table[12].text)
    return properties


def scrape_framework_cs(url):
    # TODO: ask if we should be treating these as ordinal...
    # or if these are rightfully one hot encodings.
    properties = {}
    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="containerbody")
    divs = results.find_all("tr")[2].find_all("div")
    properties["T_atom_name"] = divs[0].text
    for index, div in enumerate(divs[1:13]):
        properties["N_" + str(index + 1)] = div.text
    # FYI: this ignores </sub> tags so '4·6·4·6·6·8<sub>2</sub>' will be saved as '4·6·4·6·6·82'
    properties["vertex_symbol"] = divs[13].text
    return properties

if __name__ == '__main__':
    TABLE = "https://america.iza-structure.org/IZA-SC/ftc_table.php"
    URL_STEM = "https://america.iza-structure.org/IZA-SC/"

    table_page = requests.get(TABLE)
    table_soup = BeautifulSoup(table_page.content, "html.parser")
    all_zeolites = table_soup.find_all("td", attrs={"class": "CodeTable"})
    zeolite_data = pd.DataFrame()


    for zeolite in tqdm(all_zeolites):
        code = zeolite.find("a").text.replace(" ", "")
        url = URL_STEM + zeolite.find("a")["href"]
        properties = scrape_main_page(url)
        # This .replace() call is a bit hacky... but it works so...
        framework_url = URL_STEM + zeolite.find("a")["href"].replace(
            "framework.php", "framework_cs.php"
        )
        properties.update(scrape_framework_cs(framework_url))
        series = pd.Series(properties)
        series.name = code
        zeolite_data = zeolite_data.append(series)

    save_matrix(zeolite_data, ZEOLITE_PRIOR_FILE)
