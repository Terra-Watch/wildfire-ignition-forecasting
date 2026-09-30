"""Download the sealed-season sources (fire detections, hourly weather, vegetation index). Credentials come from the
environment (.env); nothing secret is printed. Safe to re-run: finished files are skipped."""
from wildfire.data import downloads, vegetation

if __name__ == "__main__":
    print("fire detections", downloads.fetch_firms())
    print("weather", downloads.fetch_weather())
    print("vegetation", vegetation.fetch_vegetation())
