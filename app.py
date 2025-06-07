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
def choose_weather_icon(cloud, rain, snow):
    if snow > 0.1: return "cloud_snow.png"
    if rain > 0.1: return "sun_behind_rainy_cloud.png" if cloud < 0.5 else "cloud_rain.png"
    if cloud > 0.9: return "cloudy.png"
    if cloud > 0.7: return "smalSun_Bigcloud.png"
    if cloud > 0.5: return "bigSun_smallCloud.png"
    if cloud > 0.2: return "bigSun_smallCloud.png"
    return "full_sun.png"


def time_formatter_factory(tz):
    def _fmt(x, _):
        dt = mdates.num2date(x, tz=tz)
        return dt.strftime("%H:%M\n%a") if dt.hour == 0 else dt.strftime("%H:%M")
    return _fmt


def geonames_search(place_name):
    username = st.secrets["geonames"]["username"]
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": place_name,
        "maxRows": 1,
        "featureClass": "T",  # Terrain features (mountains, hills, etc.)
        "username": username,
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

def geonames_nearby_feature(lat, lon):
    username = st.secrets["geonames"]["username"]
    url = "http://api.geonames.org/findNearbyJSON"
    params = {
        "lat": lat,
        "lng": lon,
        "featureClass": "T",  # Topographic features
        "radius": 10,         # km
        "maxRows": 1,
        "username": username,
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
# Note: "Get IP location" button is hidden
col1, col2 = st.columns([0.7, 0.3])
with col1:
    gps_clicked = st.button("📡 Get GPS location")
with col2:
    if st.button("Forecast (your location)"):
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
        st.session_state.update(coords=loc["coords"], src="GPS location", gps_wait=False)
        placeholder.empty()  # Remove message and component after success
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

if coords and src != "fail":
    st.info(f"Detected **{src}** → "
            f"{coords['latitude']:.4f}, {coords['longitude']:.4f}")
elif src == "fail":
    st.error("Could not determine location.")
else:
    st.info("No location yet — click a button above or type location below.")

# Save lat/lon for forecast
lat_dev = coords["latitude"]  if coords else None
lon_dev = coords["longitude"] if coords else None


st.markdown("<div style='height:45px;'></div>", unsafe_allow_html=True)


# ROW 3: Typed address input and button
col_txt, col_btn = st.columns([0.7, 0.3])

with col_txt:
    st.text_input("Enter a city or address:", "Hoogezand", key="typed_place")

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
    if st.button("Forecast (typed location)"):
        st.session_state["typed_btn"] = True
    st.markdown("</div>", unsafe_allow_html=True)



# ────────────── Forecast builder ──────────────
def build_forecast(lat: float, lon: float, label: str):
    # ——— Configuration identical to your second script ———
    forecast_hours    = 44
    model             = "icon_d2"
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
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # 1) Temperature
    ax_t = axs[0]
    ax_t.plot(df["time"], df["temperature_2m"],           lw=2, color="blue",  label="Temperature", zorder=4)
    ax_t.plot(df["time"], df["apparent_temperature"],     lw=2, ls="--", color="brown", label="Feels Like", zorder=4)
    t_min = min(df["temperature_2m"].min(), df["apparent_temperature"].min())
    t_max = max(df["temperature_2m"].max(), df["apparent_temperature"].max())
    ax_t.set_ylim(t_min - 1, t_max + 4)
    ax_t.set_ylabel("Temperature / Apparent [°C]", fontsize=title_fs)
    ax_t.grid(True, zorder=0)

    for x, temp, cloud, rain, snow in zip(df["time"], df["temperature_2m"], df["cloud_cover"], df["rain"], df["snowfall"]):
        icon_file = choose_weather_icon(cloud, rain, snow)
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
    axs[1].grid(True, zorder=0)

    # 3) Wind
    ax_ws  = axs[2]
    ax_gst = ax_ws.twinx()
    ax_ws.plot(df["time"], df["wind_speed_10m"], color="green",  label="Wind Speed", zorder=3)
    ax_gst.plot(df["time"], df["wind_gusts_10m"], ls="--", color="brown", label="Wind Gust", zorder=3)

    y_min = min(df["wind_speed_10m"].min(), df["wind_gusts_10m"].min())
    y_max = max(df["wind_speed_10m"].max(), df["wind_gusts_10m"].max()) * 1.1
    ax_ws.set_ylim(y_min, y_max)
    ax_gst.set_ylim(y_min, y_max)
    ax_ws.set_ylabel("Wind Speed [km/h]", color="green",  fontsize=title_fs)
    ax_gst.set_ylabel("Wind Gusts [km/h]", color="brown", fontsize=title_fs)

    u = -np.sin(np.radians(df["wind_direction_10m"]))
    v = -np.cos(np.radians(df["wind_direction_10m"]))
    xnum = mdates.date2num(df["time"])
    ax_ws.quiver(xnum, np.full_like(u, y_max * 0.9), u, v,
                 scale_units="width", scale=30, width=0.004,
                 headlength=4, headaxislength=4, pivot="middle",
                 color="dimgray", zorder=2)

    ax_ws.grid(True, zorder=0)
    ax_ws.set_zorder(ax_gst.get_zorder() + 1)  # keep grid above twin axis
    ax_ws.patch.set_visible(False)

    # Night shading
    for i in range(len(sunsets) - 1):
        for ax in axs:
            ax.axvspan(sunsets[i], sunrises[i + 1],
                       facecolor="lightgray", alpha=0.4, zorder=0)

    # Shared x-axis formatting
    axs[-1].set_xlim(df["time"].iloc[0], df["time"].iloc[-1])
    axs[-1].xaxis.set_major_formatter(FuncFormatter(time_formatter_factory(local_tz)))

    for ax in axs:
        ax.tick_params(axis="both", labelsize=label_fs)
    ax_gst.tick_params(axis="y", labelsize=label_fs)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # ——— Output to Streamlit ———
    st.markdown(f"##### Forecast for: {label} ({used_model})")
    st.pyplot(fig)


# ────────────── Button handlers ──────────────
if st.session_state["typed_btn"]:
    st.session_state["typed_btn"] = False
    user_input = st.session_state["typed_place"].strip()

    # Try GeoNames first
    geonames_result = geonames_search(user_input)
    if geonames_result:
        label = geonames_result["label"]
        build_forecast(geonames_result["lat"], geonames_result["lon"], label)
    else:
        # fallback to Nominatim
        with st.spinner("Resolving place name…"):
            loc = Nominatim(user_agent="weather_app").geocode(
                user_input, addressdetails=True)
        if loc:
            adr = loc.raw.get("address", {})
            street   = " ".join(filter(None, [adr.get("road"), adr.get("house_number")]))
            city     = adr.get("city") or adr.get("town") or adr.get("village") or adr.get("hamlet") or ""
            province = adr.get("state") or adr.get("region") or ""
            label    = ", ".join(x for x in (street, city, province) if x)
            build_forecast(loc.latitude, loc.longitude, label)
        else:
            st.error("❌ Couldn’t geocode that place – be more specific or check the spelling.")

elif st.session_state["auto_btn"]:
    st.session_state["auto_btn"] = False
    if lat_dev is not None:
        with st.spinner("Resolving your location…"):
            rev = Nominatim(user_agent="weather_app").reverse(
                (lat_dev, lon_dev), language="en", addressdetails=True)
        if rev and hasattr(rev, "raw"):
            adr = rev.raw.get("address", {})
            street   = " ".join(filter(None, [adr.get("road"), adr.get("house_number")]))
            city     = adr.get("city") or adr.get("town") or adr.get("village") or adr.get("hamlet") or ""
            province = adr.get("state") or adr.get("region") or ""
        
            if any([street, city, province]):
                label = ", ".join(x for x in (street, city, province) if x)
            else:
                # Fallback: try GeoNames if address is too sparse
                geo = geonames_nearby_feature(lat_dev, lon_dev)
                if geo:
                    label = geo["label"]
                else:
                    label = f"{lat_dev:.4f}, {lon_dev:.4f}"
        build_forecast(lat_dev, lon_dev, label)
    else:
        st.error("❌ No coordinates available – Get GPS location first.")
