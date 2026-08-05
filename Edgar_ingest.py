"""
Phase 1 — SEC EDGAR ingestion
Pulls real 10-K and 10-Q filing metadata + documents for a list of companies.

NOTE: requires internet access to data.sec.gov and www.sec.gov.
Run this on your own machine (or later, inside the AWS pipeline) —
it will NOT run inside this sandbox, which only has access to package
repositories, not general internet.

Install: pip install requests
"""

import requests
import time
import json
import os

# SEC requires a descriptive User-Agent identifying you. Replace with your info.
HEADERS = {"User-Agent": "Chirag <your_email@example.com>"}

# CIK numbers for a small starter set of companies (10-digit, zero-padded)
COMPANIES = {
    "TSLA": "0001318605",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
}

OUTPUT_DIR = "sec_filings_raw"
FORM_TYPES_WANTED = {"10-K", "10-Q"}
MAX_FILINGS_PER_COMPANY = 4  # keep the initial pull small while testing


def get_submissions(cik: str) -> dict:
    """Step 1: pull the full filing history JSON for one company."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def extract_filing_list(submissions: dict) -> list[dict]:
    """
    Step 2: the JSON stores filings as parallel arrays (all accession numbers,
    all form types, all dates, etc. — same index = same filing). Zip them
    together into a list of clean per-filing dicts and keep only 10-K/10-Q.
    """
    recent = submissions["filings"]["recent"]
    filings = []
    for i in range(len(recent["form"])):
        if recent["form"][i] in FORM_TYPES_WANTED:
            filings.append({
                "form": recent["form"][i],
                "filingDate": recent["filingDate"][i],
                "accessionNumber": recent["accessionNumber"][i],
                "primaryDocument": recent["primaryDocument"][i],
            })
    return filings


def build_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Step 3: assemble the real document URL from CIK + accession number + filename."""
    accession_no_dashes = accession_number.replace("-", "")
    cik_no_zeros = str(int(cik))  # the /Archives/ path uses CIK without leading zeros
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_document}"


def download_filing(url: str, save_path: str):
    """Step 4: download and save the raw filing, respecting rate limits."""
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    time.sleep(0.3)  # be polite — stay well under SEC's rate limit


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []  # keeps track of what we downloaded, for the next pipeline stage

    for ticker, cik in COMPANIES.items():
        print(f"\n--- {ticker} (CIK {cik}) ---")
        submissions = get_submissions(cik)
        time.sleep(0.3)

        filings = extract_filing_list(submissions)
        print(f"Found {len(filings)} 10-K/10-Q filings total, downloading first {MAX_FILINGS_PER_COMPANY}")

        for filing in filings[:MAX_FILINGS_PER_COMPANY]:
            doc_url = build_document_url(cik, filing["accessionNumber"], filing["primaryDocument"])
            filename = f"{ticker}_{filing['form']}_{filing['filingDate']}.htm"
            save_path = os.path.join(OUTPUT_DIR, filename)

            print(f"  Downloading {filing['form']} filed {filing['filingDate']} -> {filename}")
            download_filing(doc_url, save_path)

            manifest.append({
                "ticker": ticker,
                "cik": cik,
                "form": filing["form"],
                "filingDate": filing["filingDate"],
                "sourceUrl": doc_url,
                "localPath": save_path,
            })

    # save the manifest — this is what Phase 2 (chunking + embedding) will read
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(manifest)} filings downloaded. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()