"""
BuildDB.py
----------
Baut die SQLite-Datenbank "strompreis.db" auf bzw. aktualisiert sie.

Die App plant, wann die E-Fahrzeuge einer Firma am günstigsten laden. Dafür
braucht sie Strompreise für die nächsten Tage. Die Börse veröffentlicht aber nur
die Preise bis morgen. Übermorgen und den Tag danach sagt ein ML-Modell voraus.
Diese Datenbank liefert die Trainingsdaten fürs Modell und speichert alles,
womit die App arbeitet.

Marktdaten (bei jedem Lauf aktualisiert, ältere Preise werden wiederverwendet):
  strompreise   Day-Ahead-Preis Schweiz pro Stunde      (Energy-Charts API)
  wetter        Durchschnitt mehrerer Schweizer Städte   (Open-Meteo API)
  ml_daten      Preise + Wetter + Features fürs ML-Training
App-Tabellen (nur beim ersten Lauf mit Beispieldaten gefüllt, danach nie überschrieben):
  tarife, fahrzeugtypen, fahrzeuge, fahrplan, einstellungen, prognosen, ladeplan

Ausführen nach ca. 13:00 Uhr. Vorher sind die Preise für morgen noch nicht
veröffentlicht, und dem Modell "übermorgen" fehlt sein wichtigstes Feature.
Die App kann das auch automatisch erledigen (siehe Anleitung 2 ganz unten).

Datenquellen: api.energy-charts.info (Fraunhofer ISE) und open-meteo.com, beide CC BY 4.0
KI-Hinweis: Mit Claude (Anthropic) erstellt, von uns geprüft und angepasst.

Installation:   pip install -r requirements.txt
Ausführen:      python BuildDB.py
"""

import sqlite3
import threading
import time

import holidays
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
DATENBANK = "strompreis.db"

# Ab 2024: ein ganzes Jahr zum Trainieren plus Testmonate, damit das Modell
# jede Jahreszeit mindestens einmal sieht.
START = "2024-01-01"

EUR_ZU_CHF = 0.94                          # Annahme: fester Wechselkurs (API liefert Euro)

# SQLite kennt keinen eigenen Datumstyp, darum speichern wir Zeiten als Text
# in einem festen, sortierbaren Format. Alle Zeiten in der DB sind in UTC.
ZEITFORMAT = "%Y-%m-%d %H:%M"

# Der Börsenpreis gilt für die ganze Schweiz. Das Wetter an einem einzelnen Ort
# wäre zu zufällig, darum nehmen wir den Durchschnitt mehrerer Regionen.
WETTER_STANDORTE = {
    "Zürich": (47.37, 8.54),
    "Bern": (46.95, 7.45),
    "Genf": (46.20, 6.14),
    "Basel": (47.56, 7.59),
    "Lugano": (46.00, 8.95),
}

# Wetterwerte, die den Preis beeinflussen:
#   Temperatur: Kälte -> mehr Heizen -> mehr Nachfrage -> höherer Preis
#   Wind:       viel Wind -> viel Windstrom -> tieferer Preis
#   Sonne:      viel Sonne -> viel Solarstrom -> tiefere Mittagspreise
WETTER_VARIABLEN = {
    "temperature_2m": "temperatur",        # °C
    "wind_speed_10m": "wind",              # km/h
    "shortwave_radiation": "sonne",        # W/m²
}

# Die APIs sperren uns bei zu vielen Abfragen ("429 Too Many Requests").
# Darum: wenige grosse Abfragen, Pausen dazwischen und bereits gespeicherte
# Preise wiederverwenden statt jedes Mal alles neu zu laden.
PREIS_BLOCK_TAGE = 90                      # Tage pro Preis-Abfrage
PAUSE_SEKUNDEN = 10                        # Pause zwischen Abfragen (Energy-Charts erlaubt nur wenige pro Minute)
PREISE_NEU_LADEN_TAGE = 14                 # letzte Tage immer neu laden, falls Preise korrigiert wurden


def heute() -> pd.Timestamp:
    """
    Heutiges Datum (Mitternacht, Schweizer Zeit).
    Eine Funktion statt einer festen Variable, weil die App tagelang laufen kann.
    Eine Variable würde nur beim Start berechnet und wäre am nächsten Tag falsch.
    """
    return pd.Timestamp.now(tz="Europe/Zurich").normalize()


def hole_json(url: str, params: dict, timeout: int = 60):
    """
    Fragt eine API ab und gibt die Antwort als JSON zurück.
    Stossen wir trotzdem ans Limit (429), wartet die Funktion und versucht es
    erneut, statt das ganze Skript abzubrechen. Andere Fehler brechen sofort ab,
    damit nicht mit falschen Daten weitergerechnet wird.
    """
    for versuch in range(5):
        antwort = requests.get(url, params=params, timeout=timeout)
        if antwort.status_code != 429:
            antwort.raise_for_status()     # andere Fehler: abbrechen
            return antwort.json()
        retry_after = antwort.headers.get("Retry-After", "")
        warten = int(retry_after) if retry_after.isdigit() else 30 * 2 ** versuch
        print(f"    API-Limit erreicht, warte {warten} s ...")
        time.sleep(warten)
    antwort.raise_for_status()


# ---------------------------------------------------------------------------
# 1. Strompreise
# Der Preis ist das, was das Modell vorhersagen soll (Label). Frühere Preise
# dienen ausserdem als Hinweis auf zukünftige (Features).
# ---------------------------------------------------------------------------
def lade_gespeicherte_preise() -> pd.DataFrame:
    """
    Liest die Preise, die schon in der Datenbank stehen (leer beim ersten Lauf).
    Alte Preise ändern sich nicht mehr. Wenn wir sie wiederverwenden, braucht
    ein täglicher Lauf nur noch eine einzige Preis-Abfrage.
    """
    try:
        with sqlite3.connect(DATENBANK) as verbindung:
            alt = pd.read_sql("SELECT zeit_utc, preis_rp_kwh FROM strompreise", verbindung)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame(columns=["zeit_utc", "preis_rp_kwh"])
    alt["zeit_utc"] = pd.to_datetime(alt["zeit_utc"], format=ZEITFORMAT, utc=True)
    return alt


def lade_strompreise() -> pd.DataFrame:
    """
    Day-Ahead-Preise der Schweiz bis und mit morgen, eine Zeile pro Stunde.
    Lädt nur, was in der Datenbank fehlt, plus die letzten 14 Tage.
    Rückgabe: zeit_utc, preis_rp_kwh
    """
    # Bis morgen, weil diese Preise schon bekannt sind (ab ca. 13 Uhr).
    # Sie sind das wichtigste Feature für die Prognose von übermorgen.
    ende = (heute() + pd.Timedelta(days=1)).tz_localize(None)

    alt = lade_gespeicherte_preise()
    beginn = pd.Timestamp(START)
    if not alt.empty:
        letzte = alt["zeit_utc"].max().tz_convert(None).normalize()
        beginn = max(beginn, letzte - pd.Timedelta(days=PREISE_NEU_LADEN_TAGE))
        print(f"  {len(alt)} Stunden aus der Datenbank, lade ab {beginn:%Y-%m-%d} neu")

    # In Blöcken von PREIS_BLOCK_TAGE abfragen
    teile = []
    block_start = beginn
    while block_start <= ende:
        block_ende = min(block_start + pd.Timedelta(days=PREIS_BLOCK_TAGE - 1), ende)
        print(f"  Strompreise {block_start:%Y-%m-%d} bis {block_ende:%Y-%m-%d} ...")
        daten = hole_json("https://api.energy-charts.info/price",
                          {"bzn": "CH", "start": block_start.strftime("%Y-%m-%d"),
                           "end": block_ende.strftime("%Y-%m-%d")})
        teile.append(pd.DataFrame({
            "zeit_utc": pd.to_datetime(daten["unix_seconds"], unit="s", utc=True),
            "preis_eur_mwh": daten["price"],
        }))
        block_start = block_ende + pd.Timedelta(days=1)
        if block_start <= ende:
            time.sleep(PAUSE_SEKUNDEN)

    neu = pd.concat(teile).dropna().drop_duplicates("zeit_utc")

    # Seit Oktober 2025 gibt es Preise pro 15 Minuten. Modell und App rechnen in
    # Stunden, also mitteln wir die vier Werte einer Stunde.
    neu = neu.set_index("zeit_utc").resample("h").mean().dropna().reset_index()

    # Rp./kWh ist die Einheit auf jeder Schweizer Stromrechnung.
    # 1 MWh = 1000 kWh und 1 EUR = 100 Cent -> durch 10, dann in Franken.
    neu["preis_rp_kwh"] = (neu["preis_eur_mwh"] / 10 * EUR_ZU_CHF).round(3)
    neu = neu[["zeit_utc", "preis_rp_kwh"]]

    # Alt + neu zusammenführen, bei Überschneidung gilt der neue Wert
    preise = neu if alt.empty else pd.concat([alt, neu]).drop_duplicates("zeit_utc", keep="last")
    return preise.sort_values("zeit_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Wetter
# Wind, Sonne und Temperatur beeinflussen, wie viel Strom produziert und
# gebraucht wird, und damit den Preis.
# ---------------------------------------------------------------------------
def lade_wetter_alle_orte(url: str, params: dict) -> pd.DataFrame:
    """
    Lädt das Wetter für alle Standorte mit einer einzigen Abfrage.
    Open-Meteo erlaubt mehrere Orte pro Abfrage (Koordinaten mit Komma getrennt).
    Eine Abfrage statt fünf schont das API-Limit.
    """
    breiten = ",".join(str(b) for b, _ in WETTER_STANDORTE.values())
    laengen = ",".join(str(l) for _, l in WETTER_STANDORTE.values())
    # "timezone": "GMT" -> Zeiten in UTC, passend zu den Strompreisen
    daten = hole_json(url, {"latitude": breiten, "longitude": laengen,
                            "hourly": ",".join(WETTER_VARIABLEN), "timezone": "GMT", **params})
    if isinstance(daten, dict):            # bei nur einem Ort kommt keine Liste zurück
        daten = [daten]
    teile = []
    for ort in daten:
        tabelle = pd.DataFrame(ort["hourly"]).rename(columns=WETTER_VARIABLEN)
        tabelle["zeit_utc"] = pd.to_datetime(tabelle.pop("time"), utc=True)
        teile.append(tabelle)
    return pd.concat(teile, ignore_index=True)


def lade_wetter() -> pd.DataFrame:
    """
    Stündliches Wetter als Durchschnitt aller Standorte, aus zwei Quellen:
      Archiv   = gemessene Werte fürs Training (ca. 5 Tage im Rückstand)
      Prognose = füllt die Lücke zum Archiv und die nächsten 3 Tage
    Zum Trainieren brauchen wir die Vergangenheit, zum Vorhersagen die
    Wetterprognose für übermorgen und den Tag danach.
    """
    print("  Wetter-Archiv (alle Orte) ...")
    archiv = lade_wetter_alle_orte(
        "https://archive-api.open-meteo.com/v1/archive",
        {"start_date": START, "end_date": (heute() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")})
    time.sleep(PAUSE_SEKUNDEN)
    print("  Wetter-Prognose (alle Orte) ...")
    prognose = lade_wetter_alle_orte(
        "https://api.open-meteo.com/v1/forecast", {"past_days": 7, "forecast_days": 4})

    spalten = list(WETTER_VARIABLEN.values())
    archiv = archiv.groupby("zeit_utc")[spalten].mean().dropna()
    prognose = prognose.groupby("zeit_utc")[spalten].mean().dropna()
    # Gemessene Werte sind genauer, darum hat das Archiv Vorrang. Die Prognose
    # füllt nur die Stunden, die im Archiv noch fehlen.
    prognose = prognose[prognose.index > archiv.index.max()]
    wetter = pd.concat([archiv, prognose]).round(2).reset_index()

    # Markiert Stunden, deren Wetter nur vorhergesagt ist. Wichtig für die
    # Auswertung: Im Training steckt gemessenes Wetter, bei der echten
    # Vorhersage nur eine Wetterprognose.
    wetter["ist_prognose"] = (wetter["zeit_utc"] > pd.Timestamp.now(tz="UTC")).astype(int)
    return wetter


# ---------------------------------------------------------------------------
# 3. ML-Daten
# Alles, was das Modell braucht, in einer Tabelle. Das ML-Skript muss nur
# noch diese Tabelle lesen.
# ---------------------------------------------------------------------------
def baue_ml_daten(preise: pd.DataFrame, wetter: pd.DataFrame) -> pd.DataFrame:
    """
    Eine Zeile pro Stunde, vom Start bis 3 Tage in die Zukunft.
      Zeilen mit Preis  -> Training und Test
      Zeilen ohne Preis -> diese Stunden sagt das Modell voraus

    Zwei Modelle, damit beide den bekannten Preis von morgen nutzen:
      "übermorgen":  preis_vor_24h + preis_vor_168h
      "Tag danach":  preis_vor_48h + preis_vor_168h
    Das Modell "Tag danach" darf preis_vor_24h NICHT nutzen: Das wäre der Preis
    von übermorgen, den man bei der Prognose noch nicht kennt. Im Training wäre
    die Spalte trotzdem gefüllt, und das Modell sähe zu gut aus (Datenleck).
    """
    # Die leeren Zukunftszeilen sind genau die Stunden, für die die App eine
    # Prognose braucht.
    alle_stunden = pd.date_range(preise["zeit_utc"].min(),
                                 heute().tz_convert("UTC") + pd.Timedelta(days=4),
                                 freq="h", inclusive="left")
    daten = pd.DataFrame({"zeit_utc": alle_stunden})
    daten = daten.merge(preise, on="zeit_utc", how="left").merge(wetter, on="zeit_utc", how="left")

    # Kalender-Features in Schweizer Zeit, weil sich der Tagesrhythmus
    # (aufstehen, arbeiten, kochen) nach der lokalen Uhr richtet.
    lokal = daten["zeit_utc"].dt.tz_convert("Europe/Zurich")
    daten["zeit_lokal"] = lokal.dt.strftime(ZEITFORMAT)
    daten["stunde"] = lokal.dt.hour                   # nachts billig, morgens/abends teuer
    daten["wochentag"] = lokal.dt.day_of_week         # 0 = Montag
    daten["wochenende"] = (daten["wochentag"] >= 5).astype(int)  # weniger Industrie/Büros
    daten["monat"] = lokal.dt.month                   # Winter teurer

    # An Feiertagen ist die Nachfrage tief wie an einem Sonntag. Kanton Zürich,
    # weil viele Feiertage (Karfreitag, Stephanstag ...) kantonal geregelt sind.
    feiertage = holidays.country_holidays("CH", subdiv="ZH", years=range(2024, heute().year + 2))
    daten["feiertag"] = lokal.dt.date.map(lambda tag: int(tag in feiertage))

    # Frühere Preise als Features. Strompreise ändern sich selten sprunghaft,
    # der Preis zur gleichen Stunde gestern/letzte Woche ist ein starker Hinweis.
    #   24 h / 48 h: Preis von morgen für die Modelle "übermorgen" / "Tag danach"
    #   168 h:       gleiche Stunde eine Woche früher (Wochenrhythmus)
    # Umsetzung: Preistabelle um X Stunden nach vorne schieben und anhängen.
    for stunden in (24, 48, 168):
        frueher = preise.assign(zeit_utc=preise["zeit_utc"] + pd.Timedelta(hours=stunden))
        daten = daten.merge(frueher.rename(columns={"preis_rp_kwh": f"preis_vor_{stunden}h"}),
                            on="zeit_utc", how="left")

    daten["zeit_utc"] = daten["zeit_utc"].dt.strftime(ZEITFORMAT)
    return daten[["zeit_utc", "zeit_lokal", "preis_rp_kwh",
                  "stunde", "wochentag", "wochenende", "feiertag", "monat",
                  "temperatur", "wind", "sonne", "ist_prognose",
                  "preis_vor_24h", "preis_vor_48h", "preis_vor_168h"]]


# ---------------------------------------------------------------------------
# 4. App-Tabellen
# Die App muss sich merken, welche Fahrzeuge es gibt, wann sie fahren, welcher
# Tarif gilt und welche Ladepläne und Prognosen berechnet wurden.
# ---------------------------------------------------------------------------
TABELLEN_APP = """
-- ML-Vorhersagen mit Erstellungszeitpunkt (für den Vergleich Prognose vs. echter Preis)
CREATE TABLE IF NOT EXISTS prognosen (
    erstellt_am      TEXT NOT NULL,
    zeit_utc         TEXT NOT NULL,     -- vorhergesagte Stunde
    prognose_rp_kwh  REAL NOT NULL,
    PRIMARY KEY (erstellt_am, zeit_utc)
);

-- Der Börsenpreis ist schweizweit gleich, der Kunde zahlt aber zusätzlich Netz,
-- Abgaben und Marge des lokalen Versorgers: Endpreis = Börsenpreis + Aufschlag.
-- Mit Hoch-/Niedertarif kann sich auch der Ladeplan ändern.
CREATE TABLE IF NOT EXISTS tarife (
    tarif_id             INTEGER PRIMARY KEY,
    name                 TEXT NOT NULL,
    aufschlag_rp_kwh     REAL NOT NULL,
    ht_aufschlag_rp_kwh  REAL,          -- Aufschlag im Hochtarif
    ht_start             INTEGER,       -- Hochtarif ab Stunde
    ht_ende              INTEGER,       -- Hochtarif bis Stunde (exklusiv)
    quelle               TEXT
);

-- Technische Daten pro Fahrzeugmodell
CREATE TABLE IF NOT EXISTS fahrzeugtypen (
    typ_id               INTEGER PRIMARY KEY,
    name                 TEXT NOT NULL,
    akku_kwh             REAL NOT NULL,
    verbrauch_kwh_100km  REAL NOT NULL,
    ladeleistung_kw      REAL NOT NULL
);

-- Flotte. Ankunft/Abfahrt bestimmen, WANN geladen werden kann,
-- Akkustand und Reserve, WIE VIEL mindestens geladen werden muss.
CREATE TABLE IF NOT EXISTS fahrzeuge (
    fahrzeug_id             INTEGER PRIMARY KEY,
    name                    TEXT NOT NULL,
    typ_id                  INTEGER NOT NULL REFERENCES fahrzeugtypen(typ_id),
    ankunft_depot           TEXT NOT NULL,   -- "18:00"
    abfahrt_depot           TEXT NOT NULL,   -- "07:00"
    akkustand_prozent       REAL NOT NULL,
    mindestreserve_prozent  REAL NOT NULL DEFAULT 30
);

-- km pro Wochentag. Daraus berechnet die App, wie viel Strom ein Fahrzeug
-- braucht. 0 km = steht den ganzen Tag im Depot und kann laden (z. B. Sonntag).
CREATE TABLE IF NOT EXISTS fahrplan (
    fahrzeug_id  INTEGER NOT NULL REFERENCES fahrzeuge(fahrzeug_id) ON DELETE CASCADE,
    wochentag    INTEGER NOT NULL CHECK (wochentag BETWEEN 0 AND 6),  -- 0 = Montag
    km           REAL NOT NULL,
    PRIMARY KEY (fahrzeug_id, wochentag)
);

-- Einstellungen, die der Nutzer in der App ändert
CREATE TABLE IF NOT EXISTS einstellungen (
    schluessel  TEXT PRIMARY KEY,
    wert        TEXT NOT NULL
);

-- Gespeicherte Ladepläne. Daraus entsteht die Anzeige "lädt gerade".
CREATE TABLE IF NOT EXISTS ladeplan (
    erstellt_am  TEXT NOT NULL,
    fahrzeug_id  INTEGER NOT NULL REFERENCES fahrzeuge(fahrzeug_id) ON DELETE CASCADE,
    zeit_utc     TEXT NOT NULL,
    kwh          REAL NOT NULL,
    PRIMARY KEY (erstellt_am, fahrzeug_id, zeit_utc)
);
"""

# Beispieldaten (Paketdienst), damit die App beim ersten Öffnen nicht leer ist
# und man sofort sieht, wie sie funktioniert.
# Tarif-Aufschläge sind PLATZHALTER -> echte Werte bei der ElCom nachschauen
BEISPIEL_TARIFE = [
    ("Nur Börsenpreis", 0.0, None, None, None, "-"),
    ("Beispiel: fixer Aufschlag", 15.0, None, None, None, "PLATZHALTER"),
    ("Beispiel: Hoch-/Niedertarif", 12.0, 18.0, 7, 20, "PLATZHALTER"),
]
# name, akku_kwh, verbrauch_kwh_100km, ladeleistung_kw (Richtwerte)
BEISPIEL_FAHRZEUGTYPEN = [
    ("E-Lieferwagen klein", 45, 20, 11),
    ("E-Transporter gross", 80, 28, 11),
    ("Elektro-Dreirad", 7, 8, 3.7),
]
# name, typ_id, ankunft, abfahrt, akkustand %, reserve %, km Mo..So
# Sonntag 0 km (keine Zustellung) -> dort kann die App günstig voll laden.
BEISPIEL_FLOTTE = [
    ("Lieferwagen 1", 1, "18:00", "07:00", 55, 30, [70, 60, 60, 60, 65, 35, 0]),
    ("Lieferwagen 2", 1, "18:30", "07:30", 40, 30, [80, 70, 70, 70, 75, 40, 0]),
    ("Transporter 1", 2, "17:30", "06:30", 60, 30, [110, 95, 95, 95, 100, 50, 0]),
    ("Dreirad 1", 3, "16:00", "08:00", 50, 25, [30, 25, 25, 25, 28, 15, 0]),
]
BEISPIEL_EINSTELLUNGEN = [
    ("anzahl_ladestationen", "4"),
    ("gewaehlter_tarif", "2"),
]


def erstelle_app_tabellen(verbindung) -> None:
    """
    Legt die App-Tabellen an und füllt sie nur beim ersten Mal mit Beispieldaten.
    Das Skript läuft täglich. Dabei dürfen die Eingaben der Nutzer (eigene
    Fahrzeuge, Tarife) nicht überschrieben werden.
    """
    verbindung.executescript(TABELLEN_APP)  # "IF NOT EXISTS": nur anlegen, wenn sie fehlen
    cur = verbindung.cursor()

    if cur.execute("SELECT COUNT(*) FROM tarife").fetchone()[0] > 0:
        return                             # schon befüllt -> Nutzereingaben nicht anfassen

    cur.executemany("""INSERT INTO tarife (name, aufschlag_rp_kwh, ht_aufschlag_rp_kwh,
                       ht_start, ht_ende, quelle) VALUES (?, ?, ?, ?, ?, ?)""", BEISPIEL_TARIFE)
    cur.executemany("""INSERT INTO fahrzeugtypen (name, akku_kwh, verbrauch_kwh_100km,
                       ladeleistung_kw) VALUES (?, ?, ?, ?)""", BEISPIEL_FAHRZEUGTYPEN)
    for name, typ, ankunft, abfahrt, stand, reserve, km_pro_tag in BEISPIEL_FLOTTE:
        cur.execute("""INSERT INTO fahrzeuge (name, typ_id, ankunft_depot, abfahrt_depot,
                       akkustand_prozent, mindestreserve_prozent) VALUES (?, ?, ?, ?, ?, ?)""",
                    (name, typ, ankunft, abfahrt, stand, reserve))
        fahrzeug_id = cur.lastrowid        # ID des eben angelegten Fahrzeugs, für den Fahrplan
        cur.executemany("INSERT INTO fahrplan VALUES (?, ?, ?)",
                        [(fahrzeug_id, tag, km) for tag, km in enumerate(km_pro_tag)])
    cur.executemany("INSERT INTO einstellungen VALUES (?, ?)", BEISPIEL_EINSTELLUNGEN)


# ---------------------------------------------------------------------------
# 5. Datenbank schreiben
# ---------------------------------------------------------------------------
def baue_datenbank() -> None:
    """Lädt die Marktdaten und schreibt alles in die Datenbank (erstellt sie, falls nötig)."""
    print("1/4 Lade Strompreise ...")
    preise = lade_strompreise()
    print("2/4 Lade Wetter ...")
    wetter = lade_wetter()
    print("3/4 Berechne ML-Daten ...")
    ml_daten = baue_ml_daten(preise, wetter)

    print("4/4 Schreibe Datenbank ...")
    with sqlite3.connect(DATENBANK) as verbindung:
        # SQLite prüft Verknüpfungen nur, wenn man es einschaltet. So bleibt z. B.
        # kein Fahrplan für ein gelöschtes Fahrzeug übrig.
        verbindung.execute("PRAGMA foreign_keys = ON")

        # Marktdaten-Tabellen komplett ersetzen. Das ist sicher, weil "preise"
        # die alten Preise aus der Datenbank bereits enthält.
        for tabelle, df in [("strompreise", preise), ("wetter", wetter)]:
            df.assign(zeit_utc=df["zeit_utc"].dt.strftime(ZEITFORMAT)).to_sql(
                tabelle, verbindung, if_exists="replace", index=False)
        ml_daten.to_sql("ml_daten", verbindung, if_exists="replace", index=False)
        erstelle_app_tabellen(verbindung)

    print(f"Fertig: {DATENBANK} ({len(ml_daten)} Stunden in ml_daten)")


# ---------------------------------------------------------------------------
# 6. Automatische Aktualisierung
# Die App ruft beim Öffnen aktualisiere_falls_noetig() auf. So muss niemand
# daran denken, dieses Skript jeden Tag von Hand zu starten.
# ---------------------------------------------------------------------------
PREISE_VEROEFFENTLICHT_UM = 13             # ab dann gibt es die Preise für morgen
WARTEN_ZWISCHEN_VERSUCHEN_MIN = 30         # Mindestabstand zwischen zwei Versuchen

_letzter_versuch = None                    # Zeitpunkt des letzten Versuchs (solange die App läuft)
_sperre = threading.Lock()                 # falls mehrere Nutzer die App gleichzeitig öffnen


def ist_aktuell() -> bool:
    """
    Prüft, ob die Datenbank alle Preise enthält, die es schon geben sollte:
      vor 13 Uhr: bis Ende heute
      ab 13 Uhr:  bis Ende morgen (dann sind die Preise für morgen veröffentlicht)
    Schnell (nur eine SQL-Abfrage), darum kann die App das bei jedem Laden prüfen.
    """
    jetzt = pd.Timestamp.now(tz="Europe/Zurich")
    tage = 2 if jetzt.hour >= PREISE_VEROEFFENTLICHT_UM else 1
    # letzte Stunde, die vorhanden sein muss (z. B. morgen 23:00 Schweizer Zeit), in UTC
    noetig_bis = (heute() + pd.Timedelta(days=tage) - pd.Timedelta(hours=1)).tz_convert("UTC")
    try:
        with sqlite3.connect(DATENBANK) as verbindung:
            letzte = verbindung.execute("SELECT MAX(zeit_utc) FROM strompreise").fetchone()[0]
    except sqlite3.Error:
        return False                       # Datenbank oder Tabelle fehlt
    return letzte is not None and letzte >= noetig_bis.strftime(ZEITFORMAT)


def aktualisiere_falls_noetig() -> str | None:
    """
    Aktualisiert die Datenbank, falls sie nicht aktuell ist.
    Rückgabe: None, wenn alles aktuell ist, sonst ein Hinweistext für die App.

    Wirft nie einen Fehler (z. B. bei fehlendem Internet), damit die App mit
    den alten Daten weiterläuft statt abzustürzen.
    Versucht es höchstens alle 30 Minuten: Manchmal erscheinen die Preise erst
    nach 13 Uhr, und sonst würde jeder Klick in der App eine API-Abfrage auslösen.
    """
    global _letzter_versuch
    if ist_aktuell():
        return None
    with _sperre:
        if ist_aktuell():                  # inzwischen von jemand anderem aktualisiert
            return None
        jetzt = pd.Timestamp.now(tz="Europe/Zurich")
        if (_letzter_versuch is not None
                and jetzt - _letzter_versuch < pd.Timedelta(minutes=WARTEN_ZWISCHEN_VERSUCHEN_MIN)):
            return "Die neuesten Strompreise sind noch nicht verfügbar. Es werden die letzten bekannten Daten verwendet."
        _letzter_versuch = jetzt
        try:
            baue_datenbank()
        except Exception as fehler:        # absichtlich breit: die App darf hier nie abstürzen
            return f"Aktualisierung fehlgeschlagen ({fehler}). Es werden die letzten bekannten Daten verwendet."
    if not ist_aktuell():
        return "Die Strompreise für morgen sind noch nicht veröffentlicht. Neuer Versuch in 30 Minuten."
    return None


# Von Hand starten: python BuildDB.py
if __name__ == "__main__":
    baue_datenbank()


# ===========================================================================
# ANLEITUNG 1: ML-MODELL  (Datei noch zu erstellen, z. B. ml_modell.py)
# ===========================================================================
#
# ZIEL
#   Börsenpreis für übermorgen und den Tag danach vorhersagen und in die Tabelle
#   "prognosen" schreiben. Solange sie leer ist, nimmt die App den Preis der
#   Vorwoche als Platzhalter.
#
# EINGABE: Tabelle "ml_daten" (eine Zeile pro Stunde, Zeiten in UTC)
#   Label:     preis_rp_kwh
#   Features:  stunde, wochentag, wochenende, feiertag, monat,
#              temperatur, wind, sonne, preis_vor_24h, preis_vor_48h, preis_vor_168h
#   Keine Features: zeit_utc, zeit_lokal, ist_prognose
#
# ZWEI MODELLE (wegen Datenleck)
#   "übermorgen":  ohne preis_vor_48h
#   "Tag danach":  ohne preis_vor_24h
#
# TRAINING
#   - Nur Zeilen mit preis_rp_kwh, Zeilen mit fehlenden Features weglassen.
#   - Nach Zeit aufteilen, nicht zufällig (z. B. letzte 3 Monate = Test).
#   - Mit Baseline "Preis = preis_vor_168h" vergleichen (MAE in Rp./kWh).
#     Das Modell muss besser sein.
#   - Vorschlag: sklearn RandomForestRegressor oder HistGradientBoostingRegressor
#     (scikit-learn in requirements.txt ergänzen).
#
# VORHERSAGE: Zeilen ohne preis_rp_kwh
#   preis_vor_24h vorhanden             -> Modell "übermorgen"
#   preis_vor_24h leer, _48h vorhanden  -> Modell "Tag danach"
#   Das Wetter ist dort eine Prognose, im Training gemessen -> echte Genauigkeit
#   etwas schlechter als im Test.
#
# AUSGABE: Tabelle "prognosen"
#   erstellt_am (Text, ZEITFORMAT), zeit_utc (wie in ml_daten), prognose_rp_kwh
#   Anhängen (INSERT OR REPLACE / if_exists="append"), NIE "replace":
#   Alte Prognosen braucht der Tab "Prognose-Güte".
#
# ABLAUF: täglich nach BuildDB.py laufen lassen.
#
#
# ===========================================================================
# ANLEITUNG 2: APP
# ===========================================================================
#
# INSTALLATION
#   pip install -r requirements.txt
#
# STARTEN
#   python -m streamlit run app.py
#   ("python -m", weil streamlit.exe unter Windows oft nicht im PATH liegt)
#   Alle Dateien im selben Ordner wie strompreis.db.
#
# AUTOMATISCHE AKTUALISIERUNG
#   Die App kann die Datenbank selbst aktuell halten. Dann muss niemand
#   BuildDB.py von Hand starten. In app.py VOR der Prüfung
#   "Datenbank nicht gefunden" einfügen:
#
#       import BuildDB
#
#       with st.spinner("Prüfe, ob neue Strompreise verfügbar sind ..."):
#           hinweis = BuildDB.aktualisiere_falls_noetig()
#       if hinweis:
#           st.info(hinweis)
#
#   Verhalten:
#   - Datenbank aktuell: nur eine Prüfung (ca. 0.01 s), keine API-Abfrage.
#   - Preise fehlen: Aktualisierung in wenigen Sekunden.
#   - Datenbank fehlt: wird erstellt (beim ersten Mal 2-3 Minuten).
#   - Kein Internet / Preise noch nicht veröffentlicht: App läuft mit alten
#     Daten weiter, zeigt einen Hinweis, neuer Versuch frühestens nach 30 Min.
#   - Mehrere Nutzer gleichzeitig: nur eine Aktualisierung läuft.
#   - "Aktuell" heisst: vor 13 Uhr Preise bis Ende heute, danach bis Ende morgen.
#   Die ML-Prognosen werden dabei NICHT aktualisiert (ml_modell.py separat).
#
# AUFBAU
#   app.py          Oberfläche (Streamlit)
#   ladeplanung.py  Rechenlogik: Preise, Endpreis mit Tarif, Ladeplan
#
# TABS
#   Ladeplan        wann welches Fahrzeug lädt
#   Strompreise     Preise der nächsten Tage mit Quelle (Börse / ML / Platzhalter)
#   Flotte          Fahrzeuge und Fahrpläne bearbeiten
#   Prognose-Güte   ML-Prognose vs. echter Preis (sobald Prognosen vorhanden)
#   Tarife          Aufschläge bearbeiten (Beispielwerte sind Platzhalter)
#
# WICHTIG
#   Nutzereingaben bleiben bei jeder Aktualisierung erhalten. Nur das Löschen
#   von strompreis.db setzt alles auf die Beispieldaten zurück.
