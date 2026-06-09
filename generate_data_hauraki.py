#!/usr/bin/env python3
"""
生成 Fish Finder 网页版所需的JSON数据
SST网格 + 等温线 + 热点 → data.json
"""

import os, sys, json, datetime, urllib.request
import numpy as np
from scipy.ndimage import gaussian_filter

LON_MIN, LON_MAX = 174.5, 176.5
LAT_MIN, LAT_MAX = -37.5, -35.5

LANDMARKS = {
    'Auckland': {'lat': -36.84, 'lon': 174.77, 'icon': 'port'},
    'Coromandel': {'lat': -36.76, 'lon': 175.50, 'icon': 'port'},
    'Great Barrier Island': {'lat': -36.20, 'lon': 175.37, 'icon': 'target'},
    'Cape Colville': {'lat': -36.47, 'lon': 175.35, 'icon': 'cape'},
    'Anchorite Rock': {'lat': -36.60, 'lon': 175.10, 'icon': 'target'},
    'Motuihe Island': {'lat': -36.78, 'lon': 175.07, 'icon': 'cape'},
    'Waiheke Island': {'lat': -36.80, 'lon': 175.07, 'icon': 'cape'},
    'Cuvier Island': {'lat': -36.43, 'lon': 175.77, 'icon': 'target'},
}

# 主要钓鱼地标
THREE_KINGS_ISLANDS = {}

SPECIES = {
    'Snapper': {'min': 12, 'max': 20, 'cn': '鲷鱼'},
    'Kingfish': {'min': 16, 'max': 24, 'cn': '青甘'},
    'Hapuku': {'min': 10, 'max': 18, 'cn': '石斑/Hapuku'},
    'Kahawai': {'min': 14, 'max': 22, 'cn': '卡哈外'},
}


def download_sst(date_str):
    out = f"/tmp/sst_hauraki_{date_str}.csv"
    if os.path.exists(out) and os.path.getsize(out) > 500:
        return out
    url = (
        f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csv"
        f"?analysed_sst[({date_str}T09:00:00Z)]"
        f"[({LAT_MIN}):5:({LAT_MAX})]"
        f"[({LON_MIN}):5:({LON_MAX})]"
    )
    print(f"  下载SST ({date_str})...")
    req = urllib.request.Request(url, headers={'User-Agent': 'FishFinder/0.5'})
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = resp.read().decode('utf-8')
    with open(out, 'w') as f:
        f.write(data)
    return out


def parse_csv(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    lat_col = lon_col = val_col = None
    for i, h in enumerate(header):
        h = h.strip().lower()
        if 'latitude' in h: lat_col = i
        elif 'longitude' in h: lon_col = i
        elif val_col is None and i >= 2: val_col = i

    lats_r, lons_r, vals_r = [], [], []
    for line in lines[2:]:
        parts = line.strip().split(',')
        try:
            lat, lon = float(parts[lat_col]), float(parts[lon_col])
            val = float(parts[val_col]) if parts[val_col].strip() not in ('NaN','') else None
            lats_r.append(lat); lons_r.append(lon); vals_r.append(val)
        except: continue

    ulats = sorted(set(lats_r))
    ulons = sorted(set(lons_r))
    grid = [[None]*len(ulons) for _ in range(len(ulats))]
    lat_idx = {v:i for i,v in enumerate(ulats)}
    lon_idx = {v:i for i,v in enumerate(ulons)}
    for lat, lon, val in zip(lats_r, lons_r, vals_r):
        grid[lat_idx[lat]][lon_idx[lon]] = val
    return ulats, ulons, grid


def download_bathy():
    """下载SRTM15+水深数据 (15弧秒分辨率, ~460m)"""
    out = '/tmp/bathy_srtm15_hauraki.csv'
    if os.path.exists(out) and os.path.getsize(out) > 500:
        return out
    # SRTM15+ via ERDDAP, stride=4 (~1 arc-min, ~1.8km) for contour generation
    url = (
        f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/srtm15plus.csv"
        f"?z[({LAT_MIN}):4:({LAT_MAX})]"
        f"[({LON_MIN}):4:({LON_MAX})]"
    )
    print(f"  下载水深 (SRTM15+, 1弧分)...")
    req = urllib.request.Request(url, headers={'User-Agent': 'FishFinder/1.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read().decode('utf-8')
    with open(out, 'w') as f:
        f.write(data)
    return out


def generate_bathymetry_contours():
    """生成等深线"""
    try:
        bathy_file = download_bathy()
        blats, blons, bgrid = parse_csv(bathy_file)
        np_bathy = np.array([[v if v is not None else np.nan for v in row] for row in bgrid])
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        lon_mesh, lat_mesh = np.meshgrid(blons, blats)
        # 等深线级别: 更细致，利用SRTM15+高分辨率
        levels = [-3000, -2000, -1500, -1000, -500, -200, -100, -50, 0]
        
        fig, ax = plt.subplots()
        cs = ax.contour(lon_mesh, lat_mesh, np_bathy, levels=levels)
        
        contours = []
        for i, level in enumerate(cs.levels):
            paths = cs.allsegs[i] if hasattr(cs, 'allsegs') else []
            if not paths:
                # fallback for older matplotlib
                try:
                    coll = cs.collections[i]
                    paths = [p.vertices for p in coll.get_paths()]
                except:
                    continue
            for seg in paths:
                if hasattr(seg, 'tolist'):
                    coords = [[round(float(v[0]), 4), round(float(v[1]), 4)] for v in seg]
                else:
                    coords = [[round(float(v[0]), 4), round(float(v[1]), 4)] for v in seg]
                if len(coords) > 2:
                    contours.append({
                        'depth': int(level),
                        'coords': coords,
                    })
        plt.close(fig)
        print(f"  ✅ 等深线: {len(contours)} 条")
        return contours
    except Exception as e:
        print(f"  ⚠️ 等深线生成失败: {e}")
        return []


def download_chlorophyll(date_str):
    """下载VIIRS NRT叶绿素数据 (gapfilled, 9km)"""
    out = f'/tmp/chl_hauraki_{date_str}.csv'
    if os.path.exists(out) and os.path.getsize(out) > 500:
        return out
    ds = 'nesdisVHNnoaaSNPPnoaa20NRTchlaGapfilledDaily'
    url = (
        f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/{ds}.csv"
        f"?chlor_a[({date_str}T12:00:00Z)][(0.0)]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )
    print(f"  下载叶绿素 (VIIRS NRT, {date_str})...")
    req = urllib.request.Request(url, headers={'User-Agent': 'FishFinder/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode('utf-8')
        with open(out, 'w') as f:
            f.write(data)
        return out
    except Exception as e:
        print(f"  ⚠️ 叶绿素下载失败: {e}")
        # Try 1 day earlier
        from datetime import timedelta
        d = datetime.date.fromisoformat(date_str) - timedelta(days=1)
        url2 = url.replace(date_str, d.isoformat())
        print(f"  重试 {d.isoformat()}...")
        try:
            req2 = urllib.request.Request(url2, headers={'User-Agent': 'FishFinder/1.0'})
            with urllib.request.urlopen(req2, timeout=60) as resp2:
                data2 = resp2.read().decode('utf-8')
            out2 = f'/tmp/chl_hauraki_{d.isoformat()}.csv'
            with open(out2, 'w') as f:
                f.write(data2)
            return out2
        except Exception as e2:
            print(f"  ⚠️ 叶绿素仍然失败: {e2}")
            return None


def parse_chl_csv(filepath):
    """解析叶绿素CSV (有额外的time和altitude列)"""
    with open(filepath) as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    lat_col = lon_col = val_col = None
    for i, h in enumerate(header):
        h = h.strip().lower()
        if 'latitude' in h: lat_col = i
        elif 'longitude' in h: lon_col = i
        elif 'chlor' in h: val_col = i

    lats_r, lons_r, vals_r = [], [], []
    for line in lines[2:]:
        parts = line.strip().split(',')
        try:
            lat, lon = float(parts[lat_col]), float(parts[lon_col])
            v = parts[val_col].strip()
            val = float(v) if v not in ('NaN', '') else None
            lats_r.append(lat); lons_r.append(lon); vals_r.append(val)
        except: continue

    ulats = sorted(set(lats_r))
    ulons = sorted(set(lons_r))
    grid = [[None]*len(ulons) for _ in range(len(ulats))]
    lat_idx = {v:i for i,v in enumerate(ulats)}
    lon_idx = {v:i for i,v in enumerate(ulons)}
    for lat, lon, val in zip(lats_r, lons_r, vals_r):
        grid[lat_idx[lat]][lon_idx[lon]] = val
    return ulats, ulons, grid


def generate(date_str=None):
    if date_str is None:
        date_str = (datetime.date.today() - datetime.timedelta(days=2)).isoformat()

    print(f"🐟 生成Fish Finder网页数据 — {date_str}")
    sst_file = download_sst(date_str)
    lats, lons, grid = parse_csv(sst_file)

    # numpy grid for analysis
    np_grid = np.array([[v if v is not None else np.nan for v in row] for row in grid])
    if np.nanmean(np_grid) > 200:
        np_grid -= 273.15
        grid = [[round(v-273.15,2) if v else None for v in row] for row in grid]
    else:
        grid = [[round(v,2) if v else None for v in row] for row in grid]

    # Gradient
    filled = np.where(np.isnan(np_grid), np.nanmean(np_grid), np_grid)
    smoothed = gaussian_filter(filled, sigma=1.5)
    gy, gx = np.gradient(smoothed)
    grad = np.sqrt(gx**2 + gy**2)
    grad[np.isnan(np_grid)] = np.nan

    # Score
    score = np.zeros_like(np_grid)
    mx = np.nanmax(grad)
    if mx > 0: score += (grad/mx) * 60
    temp_s = np.zeros_like(np_grid)
    for sp in SPECIES.values():
        temp_s += ((np_grid >= sp['min']) & (np_grid <= sp['max'])).astype(float)
    temp_s /= len(SPECIES)
    score += temp_s * 40
    score[np.isnan(np_grid)] = np.nan
    smx = np.nanmax(score)
    if smx > 0: score = score/smx*100

    # SST heatmap points (for Leaflet heatmap layer)
    sst_points = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            v = np_grid[i, j]
            if not np.isnan(v):
                sst_points.append({
                    'lat': round(lat, 3),
                    'lon': round(lon, 3),
                    'sst': round(float(v), 2),
                    'score': round(float(score[i,j]), 1) if not np.isnan(score[i,j]) else 0,
                    'grad': round(float(grad[i,j]), 4) if not np.isnan(grad[i,j]) else 0,
                })

    # Isotherms (contour lines as GeoJSON)
    sst_smooth = gaussian_filter(filled, sigma=2)
    isotherms = []
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        cs = ax.contour(lon_mesh, lat_mesh, sst_smooth,
                        levels=np.arange(14, 19, 0.5))
        for coll, level in zip(cs.collections, cs.levels):
            for path in coll.get_paths():
                verts = path.vertices
                coords = [[round(float(v[0]),4), round(float(v[1]),4)] for v in verts]
                if len(coords) > 2:
                    isotherms.append({
                        'temp': round(float(level), 1),
                        'coords': coords
                    })
        plt.close(fig)
    except Exception as e:
        print(f"  等温线生成失败: {e}")

    # Hotspots
    sc = score.copy()
    sc[np.isnan(sc)] = -1
    top_idx = np.argsort(sc.flatten())[-10:][::-1]
    hotspots = []
    akl = LANDMARKS['Auckland']
    cor = LANDMARKS['Coromandel']
    for rank, idx in enumerate(top_idx, 1):
        row, col = np.unravel_index(idx, score.shape)
        lat, lon = lats[row], lons[col]
        s = score[row, col]
        t = np_grid[row, col]
        if np.isnan(s): continue
        dist_akl = np.sqrt(((lat-akl['lat'])*60)**2 + ((lon-akl['lon'])*60*np.cos(np.radians(lat)))**2)
        dist_cor = np.sqrt(((lat-cor['lat'])*60)**2 + ((lon-cor['lon'])*60*np.cos(np.radians(lat)))**2)

        species_match = [f"{v['cn']} {k}" for k, v in SPECIES.items()
                         if v['min'] <= t <= v['max']]

        # Gradient strength label
        grad_pct = np.nanpercentile(grad, 90)
        is_front = bool(grad[row, col] > grad_pct)

        hotspots.append({
            'rank': rank,
            'lat': round(float(lat), 3),
            'lon': round(float(lon), 3),
            'sst': round(float(t), 1),
            'score': round(float(s), 0),
            'dist_akl_nm': round(float(dist_akl), 0),
            'dist_cor_nm': round(float(dist_cor), 0),
            'species': species_match,
            'is_thermal_front': is_front,
        })

    # Stats
    stats = {
        'sst_min': round(float(np.nanmin(np_grid)), 1),
        'sst_max': round(float(np.nanmax(np_grid)), 1),
        'sst_mean': round(float(np.nanmean(np_grid)), 1),
        'grid_size': f"{len(lats)}x{len(lons)}",
        'total_points': len(sst_points),
    }

    # Species suitability
    species_info = []
    for k, v in SPECIES.items():
        pct = np.nansum((np_grid >= v['min']) & (np_grid <= v['max'])) / np.sum(~np.isnan(np_grid)) * 100
        species_info.append({
            'name': k,
            'cn': v['cn'],
            'temp_min': v['min'],
            'temp_max': v['max'],
            'coverage_pct': round(pct, 0),
        })

    # Chlorophyll
    chl_points = []
    chl_date = date_str
    try:
        chl_file = download_chlorophyll(date_str)
        if chl_file:
            clats, clons, cgrid = parse_chl_csv(chl_file)
            for i, lat in enumerate(clats):
                for j, lon in enumerate(clons):
                    v = cgrid[i][j]
                    if v is not None and v > 0:
                        chl_points.append({
                            'lat': round(lat, 3),
                            'lon': round(lon, 3),
                            'chl': round(v, 3),
                        })
            print(f"  ✅ 叶绿素: {len(chl_points)} 点")
    except Exception as e:
        print(f"  ⚠️ 叶绿素处理失败: {e}")

    # Bathymetry contours
    bathymetry = generate_bathymetry_contours()

    # Build output
    output = {
        'date': date_str,
        'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'region': {
            'lon_min': LON_MIN, 'lon_max': LON_MAX,
            'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
        },
        'stats': stats,
        'species': species_info,
        'landmarks': LANDMARKS,
        'landmarks_extra': THREE_KINGS_ISLANDS,
        'hotspots': hotspots,
        'isotherms': isotherms,
        'bathymetry': bathymetry,
        'chlorophyll': chl_points,
        'sst_points': sst_points,
        'lats': [round(l, 3) for l in lats],
        'lons': [round(l, 3) for l in lons],
    }

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'data_hauraki.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))
    print(f"  ✅ data.json: {os.path.getsize(out_path)//1024}KB")
    return out_path


if __name__ == '__main__':
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    generate(date_str)
