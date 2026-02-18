"""
EuroMLions — Fetch Latest EuroMillions Results

Scrapes the National Lottery site for the latest draw results.
Updates the historical CSV and fills in actual results in
history-predictions.json.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta
from io import StringIO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'historical-kaggle-clean.csv')
HISTORY_PATH = os.path.join(DATA_DIR, 'history-predictions.json')

NL_RESULTS_URL = 'https://www.national-lottery.co.uk/results/euromillions/draw-history/csv'
NL_RESULTS_PAGE = 'https://www.national-lottery.co.uk/results/euromillions'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def fetch_latest_from_csv_endpoint():
    """
    Try the NL CSV download endpoint for the last 180 days of results.
    Returns a list of draw dicts, newest first.
    """
    print("Fetching from NL CSV endpoint...")
    try:
        resp = requests.get(NL_RESULTS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text))

        draws = []
        for _, row in df.iterrows():
            draw_date = pd.to_datetime(row['DrawDate'], dayfirst=True)
            draws.append({
                'date': draw_date.strftime('%Y%m%d'),
                'date_display': draw_date.strftime('%Y-%m-%d'),
                'n1': int(row['Ball 1']),
                'n2': int(row['Ball 2']),
                'n3': int(row['Ball 3']),
                'n4': int(row['Ball 4']),
                'n5': int(row['Ball 5']),
                's1': int(row['Lucky Star 1']),
                's2': int(row['Lucky Star 2']),
                'machine': row.get('Machine', None),
                'ball_set': row.get('Ball Set', None),
            })

        print(f"✓ Fetched {len(draws)} draws from NL CSV endpoint")
        return draws

    except Exception as e:
        print(f"✗ CSV endpoint failed: {e}")
        return None


def fetch_latest_from_page():
    """
    Fallback: scrape the NL results page for the most recent draw.
    Returns a list with one draw dict, or None on failure.
    """
    print("Fetching from NL results page (fallback)...")
    try:
        resp = requests.get(NL_RESULTS_PAGE, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Look for the main number elements (NL site uses specific classes)
        # This selector may need updating if the NL site changes
        main_balls = soup.select('.main_ball, .ball, [class*="main"]')
        star_balls = soup.select('.lucky_star, .star, [class*="star"]')

        if not main_balls or not star_balls:
            print("✗ Could not find ball elements on page — site may have changed")
            return None

        main_numbers = [int(el.get_text(strip=True)) for el in main_balls[:5]]
        lucky_stars = [int(el.get_text(strip=True)) for el in star_balls[:2]]

        today = datetime.now()
        draws = [{
            'date': today.strftime('%Y%m%d'),
            'date_display': today.strftime('%Y-%m-%d'),
            'n1': main_numbers[0], 'n2': main_numbers[1], 'n3': main_numbers[2],
            'n4': main_numbers[3], 'n5': main_numbers[4],
            's1': lucky_stars[0], 's2': lucky_stars[1],
        }]

        print(f"✓ Scraped latest draw from NL results page")
        return draws

    except Exception as e:
        print(f"✗ Page scraping failed: {e}")
        return None


def fetch_results():
    """Fetch results using CSV endpoint first, falling back to page scrape."""
    draws = fetch_latest_from_csv_endpoint()
    if draws is None:
        draws = fetch_latest_from_page()
    if draws is None:
        print("✗ Could not fetch results from any source")
        return None
    return draws


def get_latest_draw(draws):
    """Get the most recent draw from a list of draws."""
    if not draws:
        return None
    return sorted(draws, key=lambda d: d['date'], reverse=True)[0]


def update_csv(draws):
    """
    Append any new draws to the historical CSV.
    Deduplicates by date.
    """
    if not draws:
        return 0

    existing = pd.read_csv(CSV_PATH)
    existing_dates = set(str(d) for d in existing['date'].values)

    new_rows = []
    for draw in draws:
        if draw['date'] not in existing_dates:
            new_rows.append({
                'date': int(draw['date']),
                'n1': draw['n1'], 'n2': draw['n2'], 'n3': draw['n3'],
                'n4': draw['n4'], 'n5': draw['n5'],
                's1': draw['s1'], 's2': draw['s2'],
            })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated = pd.concat([existing, new_df], ignore_index=True)
        updated = updated.sort_values('date').reset_index(drop=True)
        updated.to_csv(CSV_PATH, index=False)
        print(f"✓ Added {len(new_rows)} new draw(s) to CSV")
    else:
        print("✓ CSV already up to date")

    return len(new_rows)


def update_history_with_actuals(latest_draw):
    """
    Fill in actual results for any pending entries in history-predictions.json.
    Computes match counts.
    """
    if not latest_draw:
        return

    if not os.path.exists(HISTORY_PATH):
        print("✓ No history file to update")
        return

    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)

    actual_main = sorted([latest_draw['n1'], latest_draw['n2'], latest_draw['n3'],
                          latest_draw['n4'], latest_draw['n5']])
    actual_stars = sorted([latest_draw['s1'], latest_draw['s2']])
    draw_date = f"{latest_draw['date'][:4]}-{latest_draw['date'][4:6]}-{latest_draw['date'][6:]}"

    updated = False
    for entry in history:
        if entry['actual_main'] is not None:
            continue

        # Match if the draw date matches or is within 1 day of the prediction date
        entry_date = datetime.strptime(entry['draw_date'], '%Y-%m-%d')
        draw_dt = datetime.strptime(draw_date, '%Y-%m-%d')
        if abs((entry_date - draw_dt).days) <= 1:
            entry['actual_main'] = actual_main
            entry['actual_stars'] = actual_stars

            # Match counts against ensemble predictions
            ens = entry.get('ensemble', {})
            ens_main = ens.get('main', [])
            ens_stars = ens.get('stars', [])
            entry['main_matched'] = len(set(ens_main) & set(actual_main))
            entry['stars_matched'] = len(set(ens_stars) & set(actual_stars))

            updated = True
            print(f"✓ Updated history entry for {entry['draw_date']}: "
                  f"{entry['main_matched']}/5 main, {entry['stars_matched']}/2 stars")

    if updated:
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    else:
        print("✓ No pending history entries matched this draw date")


def main():
    """Fetch latest results, update CSV and history."""
    print("=" * 50)
    print("EuroMLions — Fetching Results")
    print("=" * 50)

    draws = fetch_results()
    if draws is None:
        sys.exit(1)

    latest = get_latest_draw(draws)
    print(f"\nLatest draw: {latest['date_display']}")
    print(f"  Main: {latest['n1']}, {latest['n2']}, {latest['n3']}, {latest['n4']}, {latest['n5']}")
    print(f"  Stars: {latest['s1']}, {latest['s2']}")

    update_csv(draws)
    update_history_with_actuals(latest)

    print("\n✓ Fetch complete")
    return latest


if __name__ == '__main__':
    main()
