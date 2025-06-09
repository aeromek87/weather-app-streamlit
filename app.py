# ────────────── Imports ──────────────
import os, base64, requests, time
from io import BytesIO
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
import streamlit as st
from streamlit_js_eval import get_geolocation

# ────────────── Session defaults ──────────────
defaults = {
    "coords"       : None,   # {"latitude":..,"longitude":..}
    "coords_source": None,   # "GPS" or "IP"
    "geo_wait"     : False,  # waiting spinner flag
    "typed_btn"    : False,
    "auto_btn"     : False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ────────────── Configuration ──────────────
GEONAMES_USERNAME = st.secrets["geonames"]["username"]
GEOCODER = Nominatim(user_agent="weather_app", timeout=10)

# simple 1-request-per-second wrapper
nominatim_reverse = RateLimiter(GEOCODER.reverse, min_delay_seconds=1)
nominatim_geocode = RateLimiter(GEOCODER.geocode, min_delay_seconds=1)


# ────────────── Logo ──────────────
def _img_b64(path):
    buf = BytesIO(); Image.open(path).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

if os.path.exists("img/logo.png"):
    st.markdown(
        f"""
        <div style="display:flex;align-items:flex-end;gap:20px;min-height:70px;
                    margin-top:-40px;margin-bottom:10px">
            <img src="data:image/png;base64,{_img_b64('img/logo.png')}"
                 style="max-height:90px">
            <span style="font-size:22px;font-family:Segoe UI,Helvetica,Arial">
                Reliable weather.<br>Simply delivered.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────── Helper utilities ──────────────
def choose_weather_icon(cloud, rain, snow, is_night=False):
    if snow > 0.1:
        return "1_cloud_snow.png"  # same for day/night
    if rain > 0.1:
        return "3_moon_rainy_cloud.png" if is_night else "3_sun_rainy_cloud.png" if cloud < 0.5 else "2_cloud_rain.png"
    if cloud > 0.9:
        return "4_cloudy.png"
    if cloud > 0.5:
        return "5_smallMoon_cloud.png" if is_night else "5_smallSun_cloud.png"
    if cloud > 0.2:
        return "6_bigMoon_cloud.png" if is_night else "6_bigSun_cloud.png"
    return "7_full_moon.png" if is_night else "7_full_sun.png"


def axis_formatter(tz):
    def _fmt(x, _):
        dt = mdates.num2date(x, tz=tz)
        return dt.strftime("%a") if dt.hour == 0 else dt.strftime("%H:%M")
    return _fmt

def date_axis_formatter(tz):
    def _fmt(x, _):
        dt = mdates.num2date(x, tz=tz)
        if dt.hour == 0:
            return dt.strftime("%a\n%d-%m")  # Two lines: Day + date
        else:
            return dt.strftime("%H:%M")
    return _fmt

def geonames_search(place_name, max_rows=5):
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": place_name,
        "maxRows": max_rows,
        "featureClass": "P",  # or "T" or try both if needed
        "username": GEONAMES_USERNAME,
        "lang": "en",
    }
    try:
        resp = requests.get(url, params=params, timeout=10).json()
        if resp.get("geonames"):
            return [
                {
                    "label": f"{g['name']}, {g.get('countryName', '')}",
                    "lat": float(g["lat"]),
                    "lon": float(g["lng"])
                }
                for g in resp["geonames"]
            ]
    except Exception:
        return []
    return []


def geonames_nearby_feature(lat, lon):
    url = "http://api.geonames.org/findNearbyJSON"
    params = {
        "lat": lat,
        "lng": lon,
        "featureClass": "T",  # Topographic features
        "radius": 7,         # km
        "maxRows": 1,
        "username": GEONAMES_USERNAME,
        "lang": "en",
    }
    try:
        resp = requests.get(url, params=params, timeout=10).json()
        if resp.get("geonames"):
            g = resp["geonames"][0]
            return {
                "name": g.get("name"),
                "country": g.get("countryName"),
                "lat": float(g["lat"]),
                "lon": float(g["lng"]),
                "label": f'{g.get("name")}, {g.get("countryName")}'
            }
    except Exception:
        return None
    return None

def geonames_suggest_terrain(query, max_rows=10):
    url = "http://api.geonames.org/searchJSON"
    params = dict(q=query, maxRows=max_rows, featureClass="T",
                  lang="en", orderby="relevance", username=GEONAMES_USERNAME)
    try:
        data = requests.get(url, params=params, timeout=10).json()
        return [
            dict(
                label=f"{g['name']}, {g.get('countryName','')}",
                lat=float(g["lat"]),
                lon=float(g["lng"]),
                display=f"{_suggest_icon(g.get('fcode') or g.get('featureCode'))} "
                        f"{g['name']}, {g.get('countryName','')}",
                src="geonames"
            )
            for g in data.get("geonames", [])
            if g.get("countryName")  # <── keep only if country present
        ]


    except Exception:
        return []

def nominatim_suggest(query, max_rows=10):
    """Placenames/addresses from Nominatim (OpenStreetMap)."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = dict(q=query, format="json", addressdetails=1,
                      limit=max_rows, accept_language="en")
        data = requests.get(url, params=params, timeout=10,
                            headers={"User-Agent": "weather_app"}).json()
        results = []
        for item in data:
            adr = item.get("address", {})
            street = " ".join(filter(None, [adr.get("road"), adr.get("house_number")]))
            city   = adr.get("city") or adr.get("town") or adr.get("village") or adr.get("hamlet") or ""
            prov   = adr.get("state") or adr.get("region") or ""
            ctry   = adr.get("country") or ""
            if not ctry:
                continue  # Skip entries with no country
            label  = ", ".join(x for x in (street, city, prov, ctry) if x)

            if not label:
                label = item.get("display_name", "").split(",")[0]  # fallback to first part
            
            display = f"{_suggest_icon(item.get('type') or adr.get('type') or adr.get('place_type'))} {label}"
            results.append(
                dict(
                    label=label,
                    display=display,
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    kind=item.get("type", ""),
                    kind_class=item.get("class", ""),
                    importance=item.get("importance", 0),
                    place_rank=item.get("place_rank", 0),
                    osm_type=item.get("osm_type", ""),
                    osm_id=item.get("osm_id", ""),
                    raw=item  # optionally include the full response for debugging
                )
            )

        return results
    except Exception:
        return []

def _suggest_icon(kind: str) -> str:
    kind = (kind or "").lower()

    # GeoNames fcodes (short codes)
    if kind in ("mt", "pk", "rdg", "hll", "col", "pass"):
        return "⛰️"
    if kind in ("val", "gorge", "cyn"):
        return "🏞️"
    if kind in ("ppl", "pplc", "ppla", "pplg", "pplx"):
        return "🏙️"
    if kind in ("lk", "lkn", "rsv", "stm", "dtch"):
        return "🌊"
    if kind in ("isl", "isls"):
        return "🏝️"

    # Nominatim "type" strings
    if kind in ("mountain", "peak", "hill", "ridge", "volcano", "col", "pass"):
        return "⛰️"
    if kind in ("valley", "gorge", "canyon"):
        return "🏞️"
    if kind in ("city", "town", "village", "hamlet", "municipality", "locality", "suburb"):
        return "🏙️"
    if kind in ("lake", "reservoir", "river", "stream", "bay"):
        return "🌊"
    if kind in ("island", "islet", "archipelago"):
        return "🏝️"
    if kind in ("administrative", "region", "state", "province"):
        return "📍"

    return "📍"


def filter_best_nominatim(results, min_place_rank=15):
    """
    Keep, for each city name, the best (node preferred, else highest importance)
    and drop all entries whose place_rank is below the threshold.
    """
    # Step 1 ─ discard low-rank admin areas
    filtered = [r for r in results if r.get("place_rank", 0) >= min_place_rank]

    # Step 2 ─ group by base city label (= text before first comma, case-insensitive)
    buckets = {}
    for r in filtered:
        base = r["label"].split(",")[0].strip().lower()
        buckets.setdefault(base, []).append(r)

    best = []
    for base, group in buckets.items():
        # Prefer node over relation
        nodes   = [g for g in group if g.get("osm_type") == "node"]
        choose_from = nodes or group

        # Pick highest importance (or, tie-break by place_rank)
        choose_from.sort(key=lambda g: (-g.get("importance", 0),
                                        -g.get("place_rank", 0)))
        best.append(choose_from[0])

    return best


def reverse_open_meteo(lat, lon, *, timeout=5):
    """Fast reverse geocoder with generous limits (no key required)."""
    url = ("https://geocoding-api.open-meteo.com/v1/reverse"
           f"?latitude={lat}&longitude={lon}&count=1&language=en")
    try:
        data = requests.get(url, timeout=timeout).json()
        r = data["results"][0]
        city = r.get("name") or ""
        admin = r.get("admin1") or ""
        country = r.get("country") or ""
        label = ", ".join(x for x in (city, admin, country) if x)
        return {
            "label": label,
            "raw": r
        }
    except Exception:
        return None


def safe_reverse(lat, lon):
    """
    ➊ Try Open-Meteo (never rate-limited)
    ➋ Try GeoNames (you already have the username)
    ➌ Fall back to Nominatim (may 429 on Streamlit Cloud)
    """
    # ➊ Open-Meteo
    r = reverse_open_meteo(lat, lon)
    if r and r["label"]:
        return r

    # ➋ GeoNames – your helper already exists
    r = geonames_nearby_feature(lat, lon)
    if r:
        return r

    # ➌ Nominatim with retries
    for _ in range(2):                       # two quick retries
        try:
            rev = nominatim_reverse((lat, lon),
                                    language="en", addressdetails=True)
            if rev and rev.raw:
                adr = rev.raw.get("address", {})
                street = " ".join(filter(None, [adr.get("road"),
                                                adr.get("house_number")]))
                city   = (adr.get("city") or adr.get("town")
                          or adr.get("village") or adr.get("hamlet") or "")
                admin  = adr.get("state") or adr.get("region") or ""
                country = adr.get("country") or ""
                label = ", ".join(x for x in (street, city, admin, country) if x)
                return {"label": label, "raw": rev.raw}
        except (GeocoderUnavailable, GeocoderTimedOut):
            time.sleep(1)                    # polite back-off
    return None


# ───────── GEO BUTTONS + LAYOUT ─────────
st.session_state.setdefault("coords",   None)
st.session_state.setdefault("src",      None)
st.session_state.setdefault("gps_wait", False)
st.session_state.setdefault("gps_t0",   0.0)

st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)


# # ROW 1: IP + GPS + Detected forecast button
# col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
# with col1:
#     ip_clicked = st.button("🌐 Get IP location")
# with col2:
#     gps_clicked = st.button("📡 Get GPS location")
# with col3:
#     if st.button("Forecast (your location)"):
#         st.session_state["auto_btn"] = True

# ROW 1: GPS + Detected forecast button (IP button removed)
col1, col2 = st.columns([0.7, 0.3])
with col1:
    gps_clicked = st.button("📡 Get GPS location")
with col2:
    if st.button("Show forecast (GPS)"):
        st.session_state["auto_btn"] = True


# GPS logic
if gps_clicked:
    st.session_state.update(
        gps_wait=True,
        gps_t0=time.time(),
        coords=None,
        src=None,
    )


if st.session_state["gps_wait"]:
    placeholder = st.empty()

    with placeholder.container():
        st.markdown(
            """
            <div style="margin-bottom: -10rem; font-size: 1rem;">
                ⏳ Waiting for GPS location...
            </div>
            """,
            unsafe_allow_html=True
        )

        loc = get_geolocation()  # Call the frontend JS component

    if isinstance(loc, dict) and "coords" in loc:
        coords = loc["coords"]
        st.session_state.update(coords=coords, src="GPS location", gps_wait=False)
        placeholder.empty()  # Remove message and component after success
        
        info = safe_reverse(coords["latitude"], coords["longitude"])
        st.session_state["label"] = info["label"] if info else None

    else:
        st.session_state["gps_wait"] = True  # Still waiting, rerun will check again


# IP logic
# if ip_clicked and not st.session_state["gps_wait"]:
#     try:
#         data = requests.get("https://ipinfo.io/json", timeout=5).json()
#         lat, lon = map(float, (data.get("loc") or ",").split(","))
#         st.session_state.update(coords={"latitude": lat, "longitude": lon}, src="IP fallback")
#     except Exception:
#         st.session_state["src"] = "fail"

# ROW 2: Info display
coords = st.session_state["coords"]
src    = st.session_state["src"]
label  = st.session_state.get("label")  # may be None initially

if coords and src != "fail":
    label_str = f" ({label})" if label else ""
    if label:
        st.info(f"**Detected location near:** {label} "
            f"({coords['latitude']:.4f}, {coords['longitude']:.4f})")
    else:
        st.info(f"Detected location: "
            f"({coords['latitude']:.4f}, {coords['longitude']:.4f})")
elif src == "fail":
    st.error("Could not determine location.")
else:
    st.info("No location yet - click the 📡 button above or search for a location below.")

# Save lat/lon for forecast
lat_dev = coords["latitude"]  if coords else None
lon_dev = coords["longitude"] if coords else None


st.markdown("<div style='height:45px;'></div>", unsafe_allow_html=True)


# ROW 3 – full-width text box
st.text_input(
    "Search for a location (e.g. city, mountain, or address):",
    "Hoogezand",
    key="typed_place"
)

typed_query = st.session_state["typed_place"].strip()

# Suggest when the user has typed at least 3 characters
suggestions = []
if len(typed_query) >= 3:
    raw_nom = nominatim_suggest(typed_query, 20)        # ask for 20 to catch the node
    nom = filter_best_nominatim(raw_nom)                # 💡 new filter
    ter = geonames_suggest_terrain(typed_query, 10)
    suggestions = nom + ter

# debug:
# for i, s in enumerate(suggestions):
#     st.write(f"🔎 Suggestion {i + 1}:")
#     st.json(s)
    
query_lc = typed_query.lower()

# Prioritize exact label matches or startswith
def rank(s):
    label = s["label"].lower()
    if label == query_lc:
        return 0
    elif label.startswith(query_lc):
        return 1
    elif query_lc in label:
        return 2
    return 3

suggestions.sort(key=rank)


# Deduplicate by label while preserving order
seen_coords = set()
deduped = []
for s in suggestions:
    key = (round(s["lat"], 5), round(s["lon"], 5))  # 5 decimals ≈ 1 meter
    if key not in seen_coords:
        seen_coords.add(key)
        deduped.append(s)
suggestions = deduped

# Show a dropdown only if we have suggestions
if suggestions:
    labels = [s["display"] for s in suggestions]

    col_sel, col_btn = st.columns([0.7, 0.3])
    with col_sel:
        idx = st.selectbox("Select a location", labels, key="loc_choice")
        chosen = suggestions[labels.index(idx)]
        st.session_state["chosen_loc"] = chosen

    with col_btn:
        st.markdown(
            """
            <style>
            .align-button-bottom {
                display: flex;
                align-items: flex-end;
                height: 100%;
                margin-top: 1.8rem;
            }
            </style>
            <div class="align-button-bottom">
            """,
            unsafe_allow_html=True
        )
        if st.button("Show forecast (selected)"):
            st.session_state["typed_btn"] = True
        st.markdown("</div>", unsafe_allow_html=True)


# ────────────── Forecast builder ──────────────
def build_forecast(lat: float, lon: float, label: str):
    # ——— Configuration identical to your second script ———
    forecast_hours    = 44
    rain_threshold_mm = 0.0
    tz_local_str      = "Europe/Amsterdam"
    local_tz          = pytz.timezone(tz_local_str)
    title_fs          = 16
    label_fs          = 16

    # ---------------- Fetch weather ----------------
    def fetch_icon_ensemble_data(lat, lon, model_name, forecast_hours):
        url = (
            "https://ensemble-api.open-meteo.com/v1/ensemble"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,apparent_temperature,precipitation,rain,snowfall,cloud_cover,"
            "wind_speed_10m,wind_gusts_10m,wind_direction_10m"
            f"&models={model_name}&forecast_hours={forecast_hours}"
            "&snowfall_unit=cm"
            "&timeformat=unixtime&timezone=GMT"
        )
        try:
            return requests.get(url, timeout=30).json()
        except Exception:
            return {}
    
    with st.spinner("Fetching weather data…"):
        model_used = None
        for model_name in ["icon_d2", "icon_eu", "icon_global"]:
            data = fetch_icon_ensemble_data(lat, lon, model_name, forecast_hours)
            if "hourly" in data:
                model_used = model_name
                used_model = model_name.upper().replace("_", "-")
                break
    
        if model_used is None:
            st.error("❌ Weather forecast not available for this location.")
            return

        # build time axis
        gmt_times = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in data["hourly"]["time"]]
        times     = [t.astimezone(local_tz) for t in gmt_times]

        # precipitation probability
        member_keys   = sorted(k for k in data["hourly"] if k.startswith("precipitation_member"))
        member_matrix = np.array([data["hourly"][k] for k in member_keys])
        p_precipitation = (member_matrix > rain_threshold_mm).sum(axis=0) / member_matrix.shape[0]

        # rain probability
        member_keys   = sorted(k for k in data["hourly"] if k.startswith("rain_member"))
        member_matrix = np.array([data["hourly"][k] for k in member_keys])      
        p_rain        = (member_matrix > rain_threshold_mm).sum(axis=0) / member_matrix.shape[0]

        df = pd.DataFrame({
            "time":                 times,
            "temperature_2m":       data["hourly"]["temperature_2m"],
            "apparent_temperature": data["hourly"]["apparent_temperature"],
            "cloud_cover":          np.array(data["hourly"]["cloud_cover"]) / 100.0,
            "precipitation":        data["hourly"]["precipitation"],
            "rain":                 data["hourly"]["rain"],
            "snowfall":             data["hourly"]["snowfall"],
            "wind_speed_10m":       data["hourly"]["wind_speed_10m"],
            "wind_gusts_10m":       data["hourly"]["wind_gusts_10m"],
            "wind_direction_10m":   data["hourly"]["wind_direction_10m"],
            "precipitation_prob":   p_precipitation,
            "rain_prob":            p_rain,
        })

        # sunrise / sunset for night shading
        solar_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=sunrise,sunset"
            f"&timezone={tz_local_str}"
        )
        solar_data = requests.get(solar_url, timeout=10).json()
        sunrises = pd.to_datetime(solar_data["daily"]["sunrise"]).tz_localize(local_tz)
        sunsets  = pd.to_datetime(solar_data["daily"]["sunset"]).tz_localize(local_tz)
    

    # --------------------- Plot ---------------------
    with st.spinner("Processing the data…"):
        fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    
        # 1) Temperature
        ax_t = axs[0]
        # 1 — primary temperature on host axis
        ax_t.plot(df["time"],
                  df["temperature_2m"],
                  lw=2, color="blue", label="Temperature", zorder=4)
        
        # 2 — create twin axis, but DON’T plot on it
        ax_t2 = ax_t.twinx()
        
        # 3 — apparent-temperature line is *also* plotted on ax_t
        ax_t.plot(df["time"],
                  df["apparent_temperature"],
                  lw=2, color="darkgreen", label="Apparent Temp", zorder=6)
        
        # 4 — keep the limits identical and give ax_t2 its label/ticks
        t_min = min(df["temperature_2m"].min(), df["apparent_temperature"].min())
        t_max = max(df["temperature_2m"].max(), df["apparent_temperature"].max())
        for a in (ax_t, ax_t2):
            a.set_ylim(t_min - 1, t_max + 4)
        
        ax_t.set_ylabel("Temperature [°C]",      fontsize=title_fs, color="blue")
        ax_t2.set_ylabel("Apparent Temp [°C]",   fontsize=title_fs, color="darkgreen")
        ax_t.tick_params(axis="y", labelsize=label_fs, colors="blue")
        ax_t2.tick_params(axis="y", labelsize=label_fs, colors="darkgreen")
        
        # 5 — integer tick labels on both
        int_fmt = FuncFormatter(lambda x, _: f"{int(x)}")
        ax_t.yaxis.set_major_formatter(int_fmt)
        ax_t2.yaxis.set_major_formatter(int_fmt)
        
        ax_t.set_zorder(ax_t2.get_zorder() + 1)   # Bring ax_t above twin
        ax_t.patch.set_visible(False)             # Allow ax_t2 background to show through
        
        # Grid
        ax_t.grid(True, zorder=0)                                                    
    
        for x, temp, cloud, rain, snow in zip(df["time"], df["temperature_2m"], df["cloud_cover"], df["rain"], df["snowfall"]):
            # Determine if current time is night (between sunset and next sunrise)
            is_night = any(sunset <= x < sunrises[i+1] for i, sunset in enumerate(sunsets[:-1]))
            icon_file = choose_weather_icon(cloud, rain, snow, is_night)
            icon_path = os.path.join("icons", icon_file)
            if os.path.exists(icon_path):
                img = mpimg.imread(icon_path)
                ab = AnnotationBbox(OffsetImage(img, zoom=0.05), (x, temp + 1.5),
                                    frameon=False, zorder=5)
                ax_t.add_artist(ab)
    
    
        # 2) Rain
        bar_w = 0.02
        for x, mm, p in zip(df["time"], df["precipitation"], df["precipitation_prob"]):
            if p >= 0.05:
                axs[1].bar(x, mm, width=bar_w * 2, color="steelblue", zorder=3)
                prob_str = f"{p:.1f}".rstrip("0").rstrip(".") if p != 0 else "0"
                axs[1].text(x, mm + 0.01, prob_str, ha="center", va="bottom",
                            fontsize=10, zorder=4)
        axs[1].set_ylabel("Precipitation [mm]", fontsize=title_fs)
        axs[1].set_ylim(bottom=0)  # ← this line ensures y-axis never drops below 0
        axs[1].grid(True, zorder=0)
    
        # 3) Wind
        ax_ws = axs[2]
        ax_gst = ax_ws.twinx()
        
        # Colors
        wind_speed_color = "teal"
        wind_gust_color  = "darkorange"
        
        # Plot lines (solid)
        ax_ws.plot(df["time"], df["wind_speed_10m"], lw=2,
                   color=wind_speed_color, label="Wind Speed", zorder=4)
        ax_ws.set_zorder(ax_gst.get_zorder() + 1)  # keep ax_ws in front of twin
        ax_ws.patch.set_visible(False)
        
        # Plot gusts directly on host axis to keep it on top
        ax_ws.plot(df["time"], df["wind_gusts_10m"], lw=2,
                   color=wind_gust_color, label="Wind Gusts", zorder=6)
        
        # Shared limits
        w_raw_min = min(df["wind_speed_10m"].min(), df["wind_gusts_10m"].min())
        w_raw_max = max(df["wind_speed_10m"].max(), df["wind_gusts_10m"].max())
        
        # Apply padding (20% of range)
        padding = (w_raw_max - w_raw_min) * 0.2
        w_min = max(0, w_raw_min - padding)  # don't allow negative wind
        w_max = w_raw_max + padding
        
        for ax in (ax_ws, ax_gst):
            ax.set_ylim(w_min, w_max)
    
        
        # Axis labels with color
        ax_ws.set_ylabel("Wind Speed [km/h]", fontsize=title_fs, color=wind_speed_color)
        ax_gst.set_ylabel("Wind Gusts [km/h]", fontsize=title_fs, color=wind_gust_color)
        ax_ws.tick_params(axis="y", labelsize=label_fs, colors=wind_speed_color)
        ax_gst.tick_params(axis="y", labelsize=label_fs, colors=wind_gust_color)
        
        # Integer tick formatting
        ax_ws.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax_gst.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        
        # Wind direction arrows (unchanged)
        u = -np.sin(np.radians(df["wind_direction_10m"]))
        v = -np.cos(np.radians(df["wind_direction_10m"]))
        xnum = mdates.date2num(df["time"])
        ax_ws.quiver(xnum, np.full_like(u, w_max * 0.9), u, v,
                     scale_units="width", scale=30, width=0.004,
                     headlength=4, headaxislength=4, pivot="middle",
                     color="dimgray", zorder=2)
        
        # Grid
        ax_ws.grid(True, zorder=0)
    
    
        # Night shading
        for i in range(len(sunsets) - 1):
            for ax in axs:
                ax.axvspan(sunsets[i], sunrises[i + 1],
                           facecolor="lightgray", alpha=0.4, zorder=0)
    
        # Shared x-axis limits for all
        for ax in axs:
            ax.set_xlim(df["time"].iloc[0], df["time"].iloc[-1])
            ax.tick_params(axis="x", labelsize=label_fs)
        
        # Apply formatters separately
        for i, ax in enumerate(axs):
            if i == 2:
                ax.xaxis.set_major_formatter(FuncFormatter(date_axis_formatter(local_tz)))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter(local_tz)))
    
        for ax in axs:
            ax.tick_params(axis="both", labelsize=label_fs)
        ax_gst.tick_params(axis="y", labelsize=label_fs)
    
        # Add annotation above the first subplot
        fig.text(0.01, 0.995,
                 f"Forecast for: {label}",          # full address, commas and all
                 fontsize=17, fontweight="bold",    # <-- bold the first line
                 ha="left", va="top")
        
        fig.text(0.01, 0.970,
                 f"Model: {used_model}",            # second line, regular weight
                 fontsize=15,
                 ha="left", va="top")
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
    
        # ——— Output to Streamlit ———
        st.pyplot(fig)


# ────────────── Button handlers ──────────────
if st.session_state["typed_btn"]:
    st.session_state["typed_btn"] = False

    # Use the dropdown-chosen location if it exists
    chosen = st.session_state.get("chosen_loc")
    if chosen:
        build_forecast(chosen["lat"], chosen["lon"], chosen["label"])
        st.stop()          # ← halt this run; no Python return needed

    # Otherwise fall back to your original logic …
    user_input = st.session_state["typed_place"].strip()

    geonames_result = geonames_search(user_input)
    if geonames_result:
        label = geonames_result["label"]
        st.session_state["label"] = label
        build_forecast(geonames_result["lat"], geonames_result["lon"], label)
    else:
        with st.spinner("Resolving place name…"):
            loc = GEOCODER.geocode(
                user_input, addressdetails=True, language="en")
        if loc:
            adr = loc.raw.get("address", {})
            street   = " ".join(filter(None, [adr.get("road"), adr.get("house_number")]))
            city     = adr.get("city") or adr.get("town") or adr.get("village") or adr.get("hamlet") or ""
            province = adr.get("state") or adr.get("region") or ""
            country  = adr.get("country") or ""
            label = ", ".join(x for x in (street, city, province, country) if x)
            st.session_state["label"] = label
            build_forecast(loc.latitude, loc.longitude, label)
        else:
            st.error("❌ Couldn’t geocode that place – be more specific or check the spelling.")



elif st.session_state["auto_btn"]:
    st.session_state["auto_btn"] = False
    if lat_dev is not None:
        with st.spinner("Resolving your location…"):
            info = safe_reverse(lat_dev, lon_dev)
        
        if info and info["label"]:
            st.session_state["label"] = info["label"]
        else:
            # Fallback: try GeoNames if address is too sparse
            geo = geonames_nearby_feature(lat_dev, lon_dev)
            if geo:
                label = geo["label"]
            else:
                label = f"{lat_dev:.4f}, {lon_dev:.4f}"
            st.session_state["label"] = label
        build_forecast(lat_dev, lon_dev, label)
    else:
        st.error("❌ No coordinates available – Get GPS location first.")
