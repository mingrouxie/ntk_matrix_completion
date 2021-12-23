import requests
from bs4 import BeautifulSoup
import re
import pdb
import pandas as pd
import os
import pathlib
from tqdm import tqdm


def scrape(url):
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
    if len(ring_sizes) >= 3:
        properties["ring_size_1"] = ring_sizes.text.split("\xa0\xa0")[0]
        properties["ring_size_2"] = ring_sizes.text.split("\xa0\xa0")[1]
        properties["ring_size_3"] = ring_sizes.text.split("\xa0\xa0")[2]

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


def save_matrix(matrix, file_name):
    file = os.path.abspath("")
    dir_main = pathlib.Path(file).parent.absolute()
    savepath = os.path.join(dir_main, file_name)
    matrix.to_pickle(savepath)
    matrix.to_csv(savepath.replace(".pkl", ".csv"))


TABLE = "https://america.iza-structure.org/IZA-SC/ftc_table.php"
URL_STEM = "https://america.iza-structure.org/IZA-SC/"
ZEOLITE_DATA = "scraped_zeolite_data.pkl"

table_page = requests.get(TABLE)
table_soup = BeautifulSoup(table_page.content, "html.parser")
all_zeolites = table_soup.find_all("td", attrs={"class": "CodeTable"})
zeolite_data = pd.DataFrame()


for zeolite in tqdm(all_zeolites):
    code = zeolite.find("a").text.replace(" ", "")
    url = URL_STEM + zeolite.find("a")["href"]
    properties = scrape(url)
    series = pd.Series(properties)
    series.name = code
    zeolite_data = zeolite_data.append(series)

save_matrix(zeolite_data, ZEOLITE_DATA)
