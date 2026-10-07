import os, certifi

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import lightkurve as lk

def download_raw_data(target_name: str, quarter: int):
    print(f"Searching for {target_name}(Quarter {quarter})...")

    search_result = lk.search_lightcurve(target=target_name, mission="Kepler", quarter=quarter)
    print(search_result)

    if len(search_result) == 0:
        print("No data found")
        return
    
    print("\nDownloading light curve")
    lc = search_result.download()

    print("\nAll columns in Lightcurve Object:")
    print(lc.columns)

    return lc

if __name__ == "__main__":
    raw_lc = download_raw_data("Kepler-10", quarter=2)