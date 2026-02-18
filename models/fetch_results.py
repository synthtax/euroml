"""
EuroMLions — Fetch Latest EuroMillions Results

Scrapes euro-millions.com for the latest draw results.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'historical-kaggle-clean.csv')
HISTORY_PATH = os.path.join(DATA_DIR, 'history-predictions.json')

EUROMILLIONS_URL = 'https://www.euro-millions.com/results/{date}'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def get_recent_draw_dates(count=4):
    """Get the last `count` EuroMillions draw dates (Tuesdays and Fridays)."""
    dates = []
    d = datetime.now()
    while len(dates) < count:
        if d.weekday() in [1, 4]:  # Tuesday=1, Friday=4
            dates.append(d)
        d -= timedelta(days=1)
    return dates


def fetch_draw(draw_date):
    """
    Fetch a single draw from euro-millions.com.
    draw_date: datetime object.
    Returns a draw dict or None.
    """
    date_str = draw_date.strftime('%d-%m-%Y')
    url = EUROMILLIONS_URL.format(date=date_str)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # The ascending-order ball list has clean selectors
        container = soup.find('ul', id='ballsAscending')
        if not container:
            print(f"  ✗ No ball container found for {date_str}")
            return None

        main_els = container.select('li.resultBall.ball')
        star_els = container.select('li.resultBall.lucky-star')

        if len(main_els) < 5 or len(star_els) < 2:
            print(f"  ✗ Incomplete results for {date_str}: "
                  f"{len(main_els)} main, {len(star_els)} stars")
            return None

        main = sorted([int(el.get_text(strip=True)) for el in main_els[:5]])
        stars = sorted([int(el.get_text(strip=True)) for el in star_els[:2]])

        return {
            'date': draw_date.strftime('%Y%m%d'),
            'date_display': draw_date.strftime('%Y-%m-%d'),
            'n1': main[0], 'n2': main[1], 'n3': main[2],
            'n4': main[3], 'n5': main[4],
            's1': stars[0], 's2': stars[1],
        }

    except Exception as e:
        print(f"  ✗ Failed to fetch {date_str}: {e}")
        return None


def fetch_results():
    """Fetch recent draws from euro-millions.com."""
    print("Fetching from euro-millions.com...")
    dates = get_recent_draw_dates(count=4)

    draws = []
    for d in dates:
        print(f"  Checking {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})...")
        draw = fetch_draw(d)
        if draw:
            draws.append(draw)
            print(f"  ✓ {draw['n1']}, {draw['n2']}, {draw['n3']}, "
                  f"{draw['n4']}, {draw['n5']} + "
                  f"{draw['s1']}, {draw['s2']}")

    if draws:
        print(f"✓ Fetched {len(draws)} draw(s)")
    else:
        print("✗ Could not fetch results from any source")

    return draws if draws else None


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
    if not draws:
        print("✗ No draws found from any source")
        sys.exit(1)

    latest = get_latest_draw(draws)
    if not latest:
        print("✗ Could not determine latest draw")
        sys.exit(1)

    print(f"\nLatest draw: {latest['date_display']}")
    print(f"  Main: {latest['n1']}, {latest['n2']}, {latest['n3']}, {latest['n4']}, {latest['n5']}")
    print(f"  Stars: {latest['s1']}, {latest['s2']}")

    update_csv(draws)
    update_history_with_actuals(latest)

    print("\n✓ Fetch complete")
    return latest


if __name__ == '__main__':
    main()
