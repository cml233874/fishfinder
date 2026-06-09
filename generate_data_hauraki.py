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

    # 构建水深掩膜 — 只保留真实海洋区域（水深 < -5m）
    ocean_mask = np.ones_like(np_grid, dtype=bool)  # default: all valid
    try:
        bathy_file = download_bathy()
        blats, blons, bgrid = parse_csv(bathy_file)
        np_bathy = np.array([[v if v is not None else 0 for v in row] for row in bgrid])
        # 插值水深到SST网格
        from scipy.interpolate import RegularGridInterpolator
        bathy_interp = RegularGridInterpolator(
            (np.array(blats), np.array(blons)), np_bathy,
            method='nearest', bounds_error=False, fill_value=0
        )
        lat_mesh_sst, lon_mesh_sst = np.meshgrid(lats, lons, indexing='ij')
        pts = np.column_stack([lat_mesh_sst.ravel(), lon_mesh_sst.ravel()])
        bathy_on_sst = bathy_interp(pts).reshape(np_grid.shape)
        # 陆地/浅滩掩膜: 水深 > -10m（即陆地或<10m浅水）排除
        ocean_mask = bathy_on_sst < -20  # 更严格：排除 < 20m 浅水区
        print(f"  ✅ 海洋掩膜: {ocean_mask.sum()} / {ocean_mask.size} 点 ({100*ocean_mask.sum()/ocean_mask.size:.0f}%)")
    except Exception as e:
        print(f"  ⚠️ 海洋掩膜失败，跳过: {e}")

    # Gradient
    filled = np.where(np.isnan(np_grid), np.nanmean(np_grid), np_grid)
    smoothed = gaussian_filter(filled, sigma=1.5)
    gy, gx = np.gradient(smoothed)
    grad = np.sqrt(gx**2 + gy**2)
    grad[np.isnan(np_grid)] = np.nan
    grad[~ocean_mask] = np.nan  # 陆地排除

    # Score — 只在真实海洋区域评分
    score = np.zeros_like(np_grid)
    mx = np.nanmax(grad)
    if mx > 0: score += (grad/mx) * 60
    temp_s = np.zeros_like(np_grid)
    for sp in SPECIES.values():
        temp_s += ((np_grid >= sp['min']) & (np_grid <= sp['max'])).astype(float)
    temp_s /= len(SPECIES)
    score += temp_s * 40
    score[np.isnan(np_grid)] = np.nan
    score[~ocean_mask] = np.nan  # 陆地排除
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

    # ======================================================
    # 双重断层叠加区 (Thermal + Bathymetric Front Overlap)
    # 条件：SST梯度 > 75百分位 AND 水深在关键断层附近（40-120m之间）
    # ======================================================
    dual_front_zones = []
    try:
        from scipy.interpolate import RegularGridInterpolator
        bathy_file_df = download_bathy()
        blats_df, blons_df, bgrid_df = parse_csv(bathy_file_df)
        np_bathy_df = np.array([[v if v is not None else 0 for v in row] for row in bgrid_df])
        bathy_interp_df = RegularGridInterpolator(
            (np.array(blats_df), np.array(blons_df)), np_bathy_df,
            method='nearest', bounds_error=False, fill_value=0
        )
        lat_mesh_df, lon_mesh_df = np.meshgrid(lats, lons, indexing='ij')
        pts_df = np.column_stack([lat_mesh_df.ravel(), lon_mesh_df.ravel()])
        bathy_on_sst_df = bathy_interp_df(pts_df).reshape(np_grid.shape)

        # 水深断层：40-150m之间（礁石边缘、大陆架边坡）
        bathy_front_mask = (bathy_on_sst_df >= -150) & (bathy_on_sst_df <= -40)

        # SST热力断层：梯度 > 70百分位
        grad_threshold = np.nanpercentile(grad[ocean_mask], 70) if ocean_mask.any() else 0
        thermal_front_mask = grad > grad_threshold

        # 双重叠加区
        dual_mask = bathy_front_mask & thermal_front_mask & ocean_mask & ~np.isnan(np_grid)

        # 找叠加区中高分点作为标注
        dual_score = score.copy()
        dual_score[~dual_mask] = np.nan
        valid_dual = np.argwhere(~np.isnan(dual_score))

        if len(valid_dual) > 0:
            # 用网格化方式找代表性点（防止聚堆），分成4x4网格各取最高分
            lat_bins = np.linspace(min(lats), max(lats), 5)
            lon_bins = np.linspace(min(lons), max(lons), 5)
            for li in range(4):
                for lj in range(4):
                    zone_mask = dual_mask.copy()
                    zone_mask &= (lat_mesh_df >= lat_bins[li]) & (lat_mesh_df < lat_bins[li+1])
                    zone_mask &= (lon_mesh_df >= lon_bins[lj]) & (lon_mesh_df < lon_bins[lj+1])
                    if not zone_mask.any(): continue
                    zone_score = score.copy()
                    zone_score[~zone_mask] = np.nan
                    if np.nanmax(zone_score) < 20: continue
                    best = np.nanargmax(zone_score)
                    row_d, col_d = np.unravel_index(best, zone_score.shape)
                    depth_val = float(bathy_on_sst_df[row_d, col_d])
                    sst_val = float(np_grid[row_d, col_d])
                    grad_val = float(grad[row_d, col_d])
                    dual_front_zones.append({
                        'lat': round(float(lats[row_d]), 3),
                        'lon': round(float(lons[col_d]), 3),
                        'sst': round(sst_val, 1),
                        'depth': round(depth_val, 0),
                        'score': round(float(score[row_d, col_d]), 0),
                        'grad': round(grad_val, 4),
                    })
        print(f"  ✅ 双重断层叠加区: {len(dual_front_zones)} 个标注点")
    except Exception as e:
        print(f"  ⚠️ 双重断层计算失败: {e}")

    # Hotspots — 额外过滤：距陆地太近的点排除（缓冲5nm ≈ 0.083度）
    SHORE_BUFFER_DEG = 0.08  # ~5nm
    # 用水深掩膜腐蚀得到深水区掩膜
    from scipy.ndimage import binary_erosion
    eroded_mask = binary_erosion(ocean_mask, iterations=3) if ocean_mask.any() else ocean_mask

    sc = score.copy()
    sc[~eroded_mask] = np.nan
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
        'dual_front_zones': dual_front_zones,
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
